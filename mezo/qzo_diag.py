"""Does load_params(TRANSLATED) change what infer() computes? QZO is dead here if not.

The QZO loop reported a gradient of exactly 0.000e+00 at every step over 30 steps. Either
the perturbation never reached the inference path, or it was cast away. This separates the
two: scale the float scales by a large factor, not a small one, and see whether the output
moves at all. A 50% change in a scale that the graph uses cannot produce an identical output.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext
from hailo_sdk_client.hailo_archive.hailo_archive import ParamsKinds

C, H, O, B = 256, 256, 64, 8
LOG = open("/work/qzo_diag.txt", "w", buffering=1)


def say(s):
    print(s, flush=True)
    LOG.write(s + "\n")


np.random.seed(1171)
W1 = (np.random.randn(H, C, 1, 1) * 0.05).astype(np.float32)
W2 = (np.random.randn(O, H, 1, 1) * 0.05).astype(np.float32)
g = helper.make_graph(
    [helper.make_node("Conv", ["x", "W1"], ["h0"], kernel_shape=[1, 1]),
     helper.make_node("Relu", ["h0"], ["h1"]),
     helper.make_node("Conv", ["h1", "W2"], ["y"], kernel_shape=[1, 1])],
    "tiny",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, 1, 1])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, O, 1, 1])],
    [numpy_helper.from_array(W1, "W1"), numpy_helper.from_array(W2, "W2")])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 9
onnx.save(m, "/tmp/tiny.onnx")

runner = ClientRunner(hw_arch="hailo10h")
runner.translate_onnx_model("/tmp/tiny.onnx", "tiny", start_node_names=["x"],
                            end_node_names=["y"], net_input_shapes={"x": [1, C, 1, 1]})
runner.load_model_script("quantization_param({*}, precision_mode=a16_w16)\n")
runner.optimize(np.random.randn(1024, 1, 1, C).astype(np.float32))

runner.save_params("/work/q.npz", ParamsKinds.TRANSLATED)
z = np.load("/work/q.npz", allow_pickle=True)
keys = list(z.keys())
FLOAT_SCALES = [k for k in keys if z[k].dtype.kind == "f" and "scale" in k.lower()]
KERNELS = [k for k in keys if k.endswith("kernel:0")]
say("float scale arrays: %s" % FLOAT_SCALES)
say("kernels (discrete): %s -> %s" % (KERNELS, [str(z[k].dtype) for k in KERNELS]))

np.random.seed(99)
X = np.random.randn(B, 1, 1, C).astype(np.float32)


def out():
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        return np.asarray(runner.infer(ctx, X)).reshape(B, -1)


base = out()
say("")
say("baseline mean=%+.6f std=%.6f" % (base.mean(), base.std()))

for factor, label in ((1.5, "scales x1.5"), (0.5, "scales x0.5"), (10.0, "scales x10")):
    d = {k: z[k] for k in keys}
    for k in FLOAT_SCALES:
        d[k] = (z[k] * factor).astype(z[k].dtype)
    np.savez("/work/mod.npz", **d)
    runner.load_params("/work/mod.npz", params_kind=ParamsKinds.TRANSLATED)
    o = out()
    say("%-14s max|d|=%.6e   mean=%+.6f   %s"
        % (label, np.abs(o - base).max(), o.mean(),
           "CHANGED" if np.abs(o - base).max() > 0 else "IDENTICAL -- load_params is inert"))

# Control: perturbing the discrete kernel MUST change the output. If this is also inert,
# load_params does not reach the inference path at all and nothing about scales is proven.
d = {k: z[k] for k in keys}
for k in KERNELS:
    d[k] = (z[k].astype(np.int64) // 2).astype(z[k].dtype)
np.savez("/work/mod.npz", **d)
runner.load_params("/work/mod.npz", params_kind=ParamsKinds.TRANSLATED)
o = out()
say("")
say("CONTROL, kernels halved: max|d|=%.6e   %s"
    % (np.abs(o - base).max(),
       "changed, so load_params does reach inference"
       if np.abs(o - base).max() > 0 else "IDENTICAL -- load_params never reaches inference"))
