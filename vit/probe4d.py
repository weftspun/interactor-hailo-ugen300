"""Is a 4D-expressed attention block translatable at all?

The tower is rank 2/3 throughout and the parser wants 4D NCHW, so the fix is to re-express
tokens as a spatial axis and linears as 1x1 convolutions. That is only worth doing if the
two ops with no convolutional equivalent survive: LayerNorm on a 4D tensor, and a MatMul
between two activations rather than against a constant. This builds the smallest block
containing both.
"""
import numpy as np, torch, torch.nn as nn, traceback
from hailo_sdk_client import ClientRunner

S, C, H = 256, 64, 4          # tokens, channels, heads


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(1, C)          # LayerNorm over channels, 4D-safe
        self.qkv = nn.Conv2d(C, 3 * C, 1)
        self.proj = nn.Conv2d(C, C, 1)

    def forward(self, x):                        # x: [1, C, 1, S]
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(1, H, C // H, S).permute(0, 1, 3, 2)   # [1,H,S,d]
        k = k.reshape(1, H, C // H, S)                        # [1,H,d,S]
        v = v.reshape(1, H, C // H, S).permute(0, 1, 3, 2)
        a = torch.softmax(torch.matmul(q, k) / (C // H) ** 0.5, dim=-1)
        o = torch.matmul(a, v).permute(0, 1, 3, 2).reshape(1, C, 1, S)
        return x + self.proj(o)


m = Block().eval()
x = torch.randn(1, C, 1, S)
torch.onnx.export(m, (x,), "/work/probe4d.onnx", input_names=["x"], output_names=["y"],
                  opset_version=17, dynamo=False)
print("exported probe4d.onnx")

r = ClientRunner(hw_arch="hailo10h")
try:
    r.translate_onnx_model("/work/probe4d.onnx", "probe4d",
                           start_node_names=["x"], end_node_names=["y"],
                           net_input_shapes={"x": [1, C, 1, S]})
except Exception:
    print("4D BLOCK FAILED TO TRANSLATE")
    traceback.print_exc()
    raise SystemExit(1)
print("4D block translated -- LayerNorm and activation-by-activation MatMul both parse")
try:
    r.optimize(np.random.rand(64, 1, S, C).astype(np.float32))
    print("optimize ok -- the block also quantises")
except Exception as e:
    print("translate ok but OPTIMIZE failed: %s" % type(e).__name__)
    traceback.print_exc()
