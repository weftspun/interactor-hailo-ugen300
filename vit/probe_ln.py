"""Is the fused LayerNormalization the thing the parser rejects, or is it fine?

The opset-17 run died at LayerNormalization and the opset-16 run died earlier, at Concat,
so the decomposed form was never actually reached. That is not evidence either way. This
puts the same rank-3 block through both opsets and asks the parser directly.
"""
import collections, io, traceback
import onnx, torch, torch.nn as nn
from hailo_sdk_client import ClientRunner


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.n = nn.LayerNorm(64)
        s.l = nn.Linear(64, 64)

    def forward(s, x):
        return s.l(s.n(x))


for opset in (16, 17):
    path = "/work/ln_op%d.onnx" % opset
    torch.onnx.export(Block().eval(), (torch.randn(1, 197, 64),), path,
                      input_names=["x"], output_names=["y"],
                      opset_version=opset, dynamo=False)
    ops = collections.Counter(n.op_type for n in onnx.load(path).graph.node)
    fused = "LayerNormalization" in ops
    r = ClientRunner(hw_arch="hailo10h")
    try:
        r.translate_onnx_model(path, "ln%d" % opset, start_node_names=["x"],
                               end_node_names=["y"], net_input_shapes={"x": [1, 197, 64]})
        verdict = "TRANSLATES"
    except Exception as e:
        verdict = "fails: %s" % type(e).__name__
    print("opset %d  fused=%-5s  %-38s %s"
          % (opset, fused, str(dict(ops))[:38], verdict))
