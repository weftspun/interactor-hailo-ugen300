"""How long is one HEF compile? That is the generation clock for on-device evolution.

An idle-time evolutionary loop scores a population on the accelerator and breeds the next
generation on the host, and each generation costs one compile. Ten minutes a compile is
fine overnight; two hours is not. Nothing else about the design can be sized until this
number exists.

Also measures whether the quantised parameter set exposes scale factors, which is what QZO
perturbs, and how large a params round-trip is.
"""
import time
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from hailo_sdk_client import ClientRunner

LOG = open("/work/compile_clock.txt", "w", buffering=1)


def say(s):
    print(s, flush=True)
    LOG.write(s + "\n")


def build(width, depth):
    """A stack of 1x1 convs, so compile cost can be read against parameter count."""
    np.random.seed(1171)
    nodes, inits, prev = [], [], "x"
    for i in range(depth):
        w = (np.random.randn(width, width, 1, 1) * 0.05).astype(np.float32)
        inits.append(numpy_helper.from_array(w, "W%d" % i))
        nodes.append(helper.make_node("Conv", [prev, "W%d" % i], ["h%d" % i], kernel_shape=[1, 1]))
        nodes.append(helper.make_node("Relu", ["h%d" % i], ["a%d" % i]))
        prev = "a%d" % i
    nodes[-1].output[0] = "y"
    g = helper.make_graph(
        nodes, "stack",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, width, 1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, width, 1, 1])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.save(m, "/tmp/stack.onnx")
    return sum(int(np.prod(i.dims)) for i in inits)


say("width depth     params   translate   optimize    compile    hef_MB")
say("-" * 68)
for width, depth in ((256, 2), (512, 4), (768, 8)):
    n = build(width, depth)
    r = ClientRunner(hw_arch="hailo10h")

    t = time.time()
    r.translate_onnx_model("/tmp/stack.onnx", "stack", start_node_names=["x"],
                           end_node_names=["y"], net_input_shapes={"x": [1, width, 1, 1]})
    t_tr = time.time() - t

    t = time.time()
    r.optimize(np.random.randn(256, 1, 1, width).astype(np.float32))
    t_op = time.time() - t

    t = time.time()
    hef = r.compile()
    t_co = time.time() - t

    say("%5d %5d %10d %10.1fs %10.1fs %10.1fs %9.2f"
        % (width, depth, n, t_tr, t_op, t_co, len(hef) / 1e6))

    if width == 256:
        # Does the quantised parameter set expose scales? That is what QZO perturbs.
        r.save_params("/work/q.npz", params_kind="quantized")
        z = np.load("/work/q.npz", allow_pickle=True)
        keys = list(z.keys())
        say("")
        say("    quantised params: %d arrays" % len(keys))
        say("    keys: %s" % ", ".join(keys[:12]))
        scaleish = [k for k in keys if any(w in k.lower()
                                           for w in ("scale", "zp", "zero", "limvals", "shift"))]
        say("    scale-like entries: %s" % (scaleish[:10] or "NONE FOUND"))
        say("")
