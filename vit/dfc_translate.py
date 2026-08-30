"""Translate probe: will the Dataflow Compiler accept the Qwen3-VL vision tower at all?

No transformer has been through translate here, so this runs before any calibration or
compile spend. It reports node and operator counts on success, and the first refusal on
failure -- which is the useful output either way.
"""
import sys, time, traceback
from hailo_sdk_client import ClientRunner

ONNX = "/work/qwen3vl_vit.onnx"
OUTS = ["image_embeddings", "deepstack_layer_1", "deepstack_layer_2", "deepstack_layer_3"]

r = ClientRunner(hw_arch="hailo10h")
t = time.time()
try:
    r.translate_onnx_model(ONNX, "qwen3vl_vit",
                           start_node_names=["image"], end_node_names=OUTS,
                           net_input_shapes={"image": [1, 3, 512, 512]})
except Exception as e:
    print("TRANSLATE FAILED after %.1f s" % (time.time() - t))
    traceback.print_exc()
    sys.exit(1)

print("translate ok in %.1f s" % (time.time() - t))
hn = r.get_hn()
import json
g = json.loads(hn) if isinstance(hn, str) else hn
layers = g.get("layers", {})
kinds = {}
for name, spec in layers.items():
    kinds[spec.get("type", "?")] = kinds.get(spec.get("type", "?"), 0) + 1
print("nodes %d, operators %d" % (len(layers), len(kinds)))
for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
    print("   %-28s %d" % (k, v))
r.save_har("/work/qwen3vl_vit.har")
print("saved /work/qwen3vl_vit.har")
