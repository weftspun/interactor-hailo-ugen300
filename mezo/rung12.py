"""Rungs 1 and 2: compile a graph I authored, twice, differing only in weights.

Everything measured on this device so far used efficientnet_lite0.hef, which arrived from a
previous session. Nothing here has yet gone author -> DFC -> HEF -> silicon. This writes two
HEFs from the same architecture with different weights, so rung 4 has something to compare.
"""
import time
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from hailo_sdk_client import ClientRunner

C, O = 64, 16
LOG = open("/work/rung12.txt", "w", buffering=1)


def say(s):
    print(s, flush=True)
    LOG.write(s + "\n")


def build(seed, path):
    """Same architecture, different weights. NHWC 8x8 so it is a real spatial graph."""
    rng = np.random.default_rng(seed)
    W1 = (rng.standard_normal((C, 3, 3, 3)) * 0.1).astype(np.float32)
    W2 = (rng.standard_normal((O, C, 1, 1)) * 0.1).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("Conv", ["x", "W1"], ["h0"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
         helper.make_node("Relu", ["h0"], ["h1"]),
         helper.make_node("Conv", ["h1", "W2"], ["y"], kernel_shape=[1, 1])],
        "lad",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, O, 8, 8])],
        [numpy_helper.from_array(W1, "W1"), numpy_helper.from_array(W2, "W2")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)


say("seed   translate   optimize    compile     hef_MB   path")
say("-" * 62)
for seed in (1171, 2026):
    src = "/tmp/lad_%d.onnx" % seed
    build(seed, src)
    r = ClientRunner(hw_arch="hailo10h")

    t = time.time()
    r.translate_onnx_model(src, "lad", start_node_names=["x"], end_node_names=["y"],
                           net_input_shapes={"x": [1, 3, 8, 8]})
    t_tr = time.time() - t

    t = time.time()
    r.optimize(np.random.default_rng(7).standard_normal((256, 8, 8, 3)).astype(np.float32))
    t_op = time.time() - t

    t = time.time()
    hef = r.compile()
    t_co = time.time() - t

    out = "/work/lad_%d.hef" % seed
    with open(out, "wb") as f:
        f.write(hef)
    say("%4d %10.1fs %10.1fs %10.1fs %9.3f   %s"
        % (seed, t_tr, t_op, t_co, len(hef) / 1e6, out))

say("")
say("two HEFs written; same architecture, different weights")
