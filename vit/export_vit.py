"""Exports the Qwen3-VL-4B vision tower with EditScore's merger LoRA merged, to ONNX.

The HEF that llama.cpp's mtmd loads takes a raw image and returns four streams, so
patchify and normalisation live inside the graph rather than in a preprocessor. Shapes are
fixed because a dataflow part compiles fixed shapes.

Only the three deepstack mergers carry EditScore -- the ViT blocks are stock -- so the merge
touches six linears and nothing else.
"""
import argparse, glob, json, os
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

BASE = "Qwen/Qwen3-VL-4B-Instruct"
LORA = "EditScore/EditScore-Qwen3-VL-4B-Instruct"


def load_tower(dtype=torch.float32):
    cfg = AutoConfig.from_pretrained(BASE).vision_config
    cfg._attn_implementation = "eager"
    tower = Qwen3VLVisionModel(cfg).to(dtype).eval()

    d = snapshot_download(BASE, allow_patterns=["*.json", "*.safetensors"])
    sd = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        for k, v in load_file(f).items():
            if ".visual." in k:
                sd[k.split(".visual.", 1)[1]] = v.to(dtype)
    missing, unexpected = tower.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing
    return tower, cfg


def merge_lora(tower, scale_report):
    ad = snapshot_download(LORA)
    lw = load_file(os.path.join(ad, "adapter_model.safetensors"))
    acfg = json.load(open(os.path.join(ad, "adapter_config.json")))
    scale = acfg["lora_alpha"] / acfg["r"]

    pairs = {}
    for k, v in lw.items():
        if ".visual." not in k:
            continue
        mod = k.split(".visual.", 1)[1].rsplit(".lora_", 1)[0]
        pairs.setdefault(mod, {})["A" if ".lora_A." in k else "B"] = v.float()

    n = 0
    for mod, ab in sorted(pairs.items()):
        lin = tower.get_submodule(mod)
        delta = (ab["B"] @ ab["A"]) * scale
        assert delta.shape == lin.weight.shape, (mod, delta.shape, lin.weight.shape)
        before = lin.weight.detach().clone()
        lin.weight.data += delta.to(lin.weight.dtype)
        scale_report.append((mod, float(delta.abs().max()),
                             float((lin.weight - before).abs().sum())))
        n += 1
    assert n == 6, "expected 6 merger linears, merged %d" % n
    return scale, n


class Tower(torch.nn.Module):
    """image uint8-range float [1,3,H,W] -> 4 streams, each [1, oh, ow, 2560]."""

    def __init__(self, tower, cfg, h, w):
        super().__init__()
        self.t = tower
        self.p = cfg.patch_size
        self.m = cfg.spatial_merge_size
        self.tp = cfg.temporal_patch_size
        self.gh, self.gw = h // self.p, w // self.p
        self.oh, self.ow = h // (self.p * self.m), w // (self.p * self.m)
        self.register_buffer("grid", torch.tensor([[1, self.gh, self.gw]], dtype=torch.long))

    def patchify(self, x):
        b, c, _, _ = x.shape
        p, m = self.p, self.m
        x = x.reshape(b, c, self.gh // m, m, p, self.gw // m, m, p)
        x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)
        x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, self.tp, -1, -1)
        return x.reshape(b * self.gh * self.gw, c * self.tp * p * p)

    def forward(self, image):
        x = image / 127.5 - 1.0
        out = self.t(self.patchify(x), self.grid)
        # pooler_output is the merged image embedding; last_hidden_state is pre-merger
        streams = [out.pooler_output] + list(out.deepstack_features)
        return tuple(s.reshape(1, self.oh, self.ow, -1) for s in streams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="qwen3vl_vit.onnx")
    ap.add_argument("--opset", type=int, default=16)
    a = ap.parse_args()

    tower, cfg = load_tower()
    report = []
    scale, n = merge_lora(tower, report)
    print("merged %d merger linears at lora_alpha/r = %.1f" % (n, scale))
    for mod, mx, tot in report:
        print("   %-46s max|delta| %.5f   sum|change| %.1f" % (mod, mx, tot))

    net = Tower(tower, cfg, a.size, a.size).eval()
    img = torch.randint(0, 256, (1, 3, a.size, a.size)).float()
    with torch.no_grad():
        ys = net(img)
    print("\ninput  %s" % list(img.shape))
    for i, y in enumerate(ys):
        print("output %d %-22s mean %+.4f std %.4f"
              % (i, str(list(y.shape)), y.mean(), y.std()))

    names = ["image_embeddings", "deepstack_layer_1", "deepstack_layer_2", "deepstack_layer_3"]
    torch.onnx.export(net, (img,), a.out, input_names=["image"], output_names=names,
                      opset_version=a.opset, dynamo=False)
    print("\nwrote %s (%.1f MB)" % (a.out, os.path.getsize(a.out) / 1e6))
    json.dump({"patch_size": cfg.patch_size, "spatial_merge_size": cfg.spatial_merge_size,
               "encoder_output_layers_names_suffixes": {k: k for k in names}},
              open("hailo-config.json", "w"), indent=2)
    print("wrote hailo-config.json")


if __name__ == "__main__":
    main()
