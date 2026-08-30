"""Fold the static shape plumbing, then translate.

Every Shape/Concat/Unsqueeze feeding a Reshape is constant once the input size is fixed,
and it was a Concat with no resolvable format that stopped the first translate. Simplifying
first is cheaper than teaching the parser about rank-1 shape tensors.
"""
import collections, json, sys, time, traceback
import onnx
from onnxsim import simplify

m = onnx.load("/work/qwen3vl_vit_op16.onnx")
before = collections.Counter(n.op_type for n in m.graph.node)
t = time.time()
ms, ok = simplify(m, overwrite_input_shapes={"image": [1, 3, 512, 512]})
print("simplify %s in %.1f s" % ("ok" if ok else "REPORTED FAILURE", time.time() - t))
after = collections.Counter(n.op_type for n in ms.graph.node)
print("nodes %d -> %d" % (sum(before.values()), sum(after.values())))
for k in sorted(set(before) | set(after), key=lambda k: -(before[k] + after[k]))[:14]:
    if before[k] != after[k]:
        print("   %-14s %5d -> %5d" % (k, before[k], after[k]))
onnx.save(ms, "/work/qwen3vl_vit_op16_sim.onnx", save_as_external_data=False)
print("wrote qwen3vl_vit_op16_sim.onnx")

from hailo_sdk_client import ClientRunner
OUTS = ["image_embeddings", "deepstack_layer_1", "deepstack_layer_2", "deepstack_layer_3"]
r = ClientRunner(hw_arch="hailo10h")
t = time.time()
try:
    r.translate_onnx_model("/work/qwen3vl_vit_op16_sim.onnx", "qwen3vl_vit",
                           start_node_names=["image"], end_node_names=OUTS,
                           net_input_shapes={"image": [1, 3, 512, 512]})
except Exception:
    print("TRANSLATE FAILED after %.1f s" % (time.time() - t))
    traceback.print_exc()
    sys.exit(1)
print("translate ok in %.1f s" % (time.time() - t))
r.save_har("/work/qwen3vl_vit.har")
print("saved har")
