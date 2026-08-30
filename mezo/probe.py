"""Is a Hailo-quantised graph locally flat? MeZO dies if it is.

MeZO estimates a gradient from L(t+eps*z) - L(t-eps*z). If quantisation buckets both
perturbations into the same output that difference is exactly zero, and the optimiser
converges to nothing while every log line looks healthy. This finds the epsilon where
the quantised graph stops responding, with the float graph as the control.

1x1 convs rather than MatMul: the DFC is a vision compiler and wants NCHW spatial dims.
"""
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext

np.random.seed(1171)
C, H, O = 256, 256, 64
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
onnx.checker.check_model(m)
onnx.save(m, "/tmp/tiny.onnx")

runner = ClientRunner(hw_arch="hailo10h")
runner.translate_onnx_model("/tmp/tiny.onnx", "tiny",
                            start_node_names=["x"], end_node_names=["y"],
                            net_input_shapes={"x": [1, C, 1, 1]})
runner.optimize(np.random.randn(1024, 1, 1, C).astype(np.float32))
print("translated and quantised, 1024 calibration samples\n")

def out(xs, kind):
    with runner.infer_context(kind) as ctx:
        return np.asarray(runner.infer(ctx, xs)).reshape(len(xs), -1)

x0 = np.random.randn(1, 1, 1, C).astype(np.float32)
for kind, label in ((InferenceContext.SDK_NATIVE, "float"),
                    (InferenceContext.SDK_QUANTIZED, "quantised")):
    b, r = out(x0, kind)[0], out(x0, kind)[0]
    print(f"{label:10s} mean={b.mean():+.5f} std={b.std():.5f}  "
          f"determinism max|repeat-base|={np.abs(r-b).max():.3e}")

print(f"\n{'epsilon':>10}  {'float d':>12}  {'quantised d':>13}   verdict")
print("-" * 62)
z = np.random.randn(1, 1, 1, C).astype(np.float32); z /= np.linalg.norm(z)
for e in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
    pair = np.concatenate([x0 + e * z, x0 - e * z]).astype(np.float32)
    df = np.abs(np.diff(out(pair, InferenceContext.SDK_NATIVE), axis=0)).max()
    dq = np.abs(np.diff(out(pair, InferenceContext.SDK_QUANTIZED), axis=0)).max()
    print(f"{e:10.0e}  {df:12.3e}  {dq:13.3e}   "
          f"{'signal' if dq > 0 else 'FLAT -- MeZO gets nothing'}")
