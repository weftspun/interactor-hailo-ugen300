"""Does an AGGREGATED loss move under small perturbation on a quantised graph?

The first probe measured max|d output| on 64 outputs from one sample and found a
staircase. That is not what MeZO consumes. MeZO consumes a scalar loss summed over a
batch, where individual quantisation steps partly cancel and the sum has far finer
granularity. This measures that scalar, and reports how many of the batch's outputs
moved at all -- a sum over many terms can respond while every single term is flat.
"""
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext

np.random.seed(1171)
C, H, O, B = 256, 256, 64, 256
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
runner.optimize(np.random.randn(1024, 1, 1, C).astype(np.float32))

def out(xs, kind):
    with runner.infer_context(kind) as ctx:
        return np.asarray(runner.infer(ctx, xs)).reshape(len(xs), -1)

X = np.random.randn(B, 1, 1, C).astype(np.float32)
# A low-loss regime: the target IS the quantised model's own output, so we sit at a
# minimum the way a pretrained model does, which is the regime Gao's argument is about.
T = out(X, InferenceContext.SDK_QUANTIZED)

def loss(xs, kind):
    return float(np.mean((out(xs, kind) - T) ** 2))

print(f"batch {B}, loss = mean squared error against the model's own output\n")
print(f"{'epsilon':>9}  {'float dL':>12}  {'quant dL':>12}  {'outputs moved':>14}")
print("-" * 56)
Z = np.random.randn(*X.shape).astype(np.float32)
Z /= np.linalg.norm(Z.reshape(B, -1), axis=1).reshape(B, 1, 1, 1)
for e in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
    p, mns = (X + e * Z).astype(np.float32), (X - e * Z).astype(np.float32)
    dF = abs(loss(p, InferenceContext.SDK_NATIVE) - loss(mns, InferenceContext.SDK_NATIVE))
    qp, qm = out(p, InferenceContext.SDK_QUANTIZED), out(mns, InferenceContext.SDK_QUANTIZED)
    dQ = abs(float(np.mean((qp - T) ** 2)) - float(np.mean((qm - T) ** 2)))
    moved = int((np.abs(qp - qm) > 0).sum())
    print(f"{e:9.0e}  {dF:12.4e}  {dQ:12.4e}  {moved:8d}/{B*O:<6d}")
