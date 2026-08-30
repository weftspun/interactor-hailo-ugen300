"""The smallest real attention block, batched and unbatched, all the way to a HEF.

Same weights, same maths, same opset. The only difference is whether a batch dimension is
present, which is the one thing the model zoo has and my Qwen3-VL export did not. If the
batched arm compiles and the unbatched arm fails, the batch dimension is the cause rather
than something that merely travels with it.
"""
import collections, time, traceback
import numpy as np, onnx, torch, torch.nn as nn
from hailo_sdk_client import ClientRunner

S, C, H = 64, 64, 4
D = C // H


class Batched(nn.Module):
    def __init__(s):
        super().__init__()
        s.n = nn.LayerNorm(C); s.qkv = nn.Linear(C, 3 * C); s.proj = nn.Linear(C, C)
        s.n2 = nn.LayerNorm(C); s.fc1 = nn.Linear(C, 4 * C); s.fc2 = nn.Linear(4 * C, C)

    def forward(s, x):                                    # [1, S, C]
        q, k, v = s.qkv(s.n(x)).split(C, dim=-1)
        q = q.reshape(1, S, H, D).permute(0, 2, 1, 3)
        k = k.reshape(1, S, H, D).permute(0, 2, 3, 1)
        v = v.reshape(1, S, H, D).permute(0, 2, 1, 3)
        a = torch.softmax(torch.matmul(q, k) * D ** -0.5, dim=-1)
        o = torch.matmul(a, v).permute(0, 2, 1, 3).reshape(1, S, C)
        x = x + s.proj(o)
        return x + s.fc2(torch.nn.functional.gelu(s.fc1(s.n2(x))))


class Unbatched(nn.Module):
    """The same block the way Qwen3VLVisionModel writes it: a packed sequence, no batch."""

    def __init__(s, src):
        super().__init__()
        for a in ("n", "qkv", "proj", "n2", "fc1", "fc2"):
            setattr(s, a, getattr(src, a))

    def forward(s, x):                                    # [S, C]
        q, k, v = s.qkv(s.n(x)).split(C, dim=-1)
        q = q.reshape(S, H, D).permute(1, 0, 2)
        k = k.reshape(S, H, D).permute(1, 2, 0)
        v = v.reshape(S, H, D).permute(1, 0, 2)
        a = torch.softmax(torch.matmul(q, k) * D ** -0.5, dim=-1)
        o = torch.matmul(a, v).permute(1, 0, 2).reshape(S, C)
        x = x + s.proj(o)
        return x + s.fc2(torch.nn.functional.gelu(s.fc1(s.n2(x))))


torch.manual_seed(1171)
base = Batched().eval()
arms = [("batched", base, (1, S, C)), ("unbatched", Unbatched(base).eval(), (S, C))]

results = {}
for tag, mod, shape in arms:
    path = "/work/rb_%s.onnx" % tag
    torch.onnx.export(mod, (torch.randn(*shape),), path, input_names=["x"],
                      output_names=["y"], opset_version=17, dynamo=False)
    ops = collections.Counter(n.op_type for n in onnx.load(path).graph.node)
    lin = "Gemm" if ops.get("Gemm") else "MatMul"
    r = ClientRunner(hw_arch="hailo10h")
    try:
        r.translate_onnx_model(path, "rb_" + tag, start_node_names=["x"],
                               end_node_names=["y"], net_input_shapes={"x": list(shape)})
        verdict, runner = "translates", r
    except Exception as e:
        verdict, runner = "FAILS (%s)" % type(e).__name__, None
        results[tag + "_err"] = traceback.format_exc().strip().splitlines()[-1]
    print("%-10s rank %d  linears as %-6s  softmax %d  matmul %d  -> %s"
          % (tag, len(shape), lin, ops.get("Softmax", 0), ops.get("MatMul", 0), verdict))
    results[tag] = runner

if results.get("unbatched_err"):
    print("   unbatched died on: %s" % results["unbatched_err"][:110])

r = results.get("batched")
if r is None:
    raise SystemExit("batched arm did not translate; the flow stops here")

print("\ncontinuing the batched arm through the flow")
t = time.time()
import json
_hn = r.get_hn()
_hn = json.loads(_hn) if isinstance(_hn, str) else _hn
_in = [l for l in _hn["layers"].values() if l.get("type") == "input_layer"]
_shape = _in[0]["output_shapes"][0]
print("  HN input shape %s" % _shape)
_calib = [64] + list(_shape[1:])
print("  calibration set %s" % _calib)
r.optimize(np.random.rand(*_calib).astype(np.float32))
print("  optimize ok   %.1f s" % (time.time() - t))
t = time.time()
hef = r.compile()
print("  compile  ok   %.1f s, %d bytes" % (time.time() - t, len(hef)))
open("/work/rb_batched.hef", "wb").write(hef)
print("  wrote rb_batched.hef")
