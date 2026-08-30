"""How large a frame will the part take, and what does the link do when it gets one.

The 16 KB block measured 0.48 ms/frame pipelined, nearly all of it per-transfer overhead
rather than the link. A 4K RGBA frame is 33.18 MB, about two thousand times larger, so it
should land in the opposite regime -- bandwidth-bound -- if it compiles at all. Compiling is
the first question: a dataflow part allocates activation buffers at compile time.

The graph is deliberately trivial, a 1x1 convolution from 4 channels to 1. Anything heavier
would confound a transport measurement with compute.
"""
import time, traceback
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
from hailo_sdk_client import ClientRunner

SIZES = [("512p", 512, 512), ("720p", 1280, 720), ("1080p", 1920, 1080), ("4K", 3840, 2160)]
CH_IN, CH_OUT = 4, 1


def build(w, h, path):
    W = (np.random.default_rng(0).standard_normal((CH_OUT, CH_IN, 1, 1)) * 0.1).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[1, 1])], "big",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, CH_IN, h, w])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, CH_OUT, h, w])],
        [numpy_helper.from_array(W, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.save(m, path)


print("%-7s %11s %11s %9s  %s" % ("size", "in MB", "out MB", "compile s", "result"))
print("-" * 62)
for tag, w, h in SIZES:
    src = "/work/big_%s.onnx" % tag
    build(w, h, src)
    mb_in = w * h * CH_IN / 1e6
    mb_out = w * h * CH_OUT / 1e6
    t = time.time()
    try:
        r = ClientRunner(hw_arch="hailo10h")
        r.translate_onnx_model(src, "big_" + tag, start_node_names=["x"], end_node_names=["y"],
                               net_input_shapes={"x": [1, CH_IN, h, w]})
        r.optimize(np.random.randint(0, 256, (8, h, w, CH_IN)).astype(np.float32))
        hef = r.compile()
        open("/work/big_%s.hef" % tag, "wb").write(hef)
        res = "ok, HEF %.2f MB" % (len(hef) / 1e6)
    except Exception as e:
        res = "FAILS: %s: %s" % (type(e).__name__, str(e).strip().splitlines()[0][:60])
    print("%-7s %11.2f %11.2f %9.1f  %s" % (tag, mb_in, mb_out, time.time() - t, res))
