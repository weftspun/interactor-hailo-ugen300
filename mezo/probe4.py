"""MeZO viability at a16w16, full optimisation, and a realistically small loss.

Probe 3 put the target at the model's own output, an exact minimum where the true
gradient is ~0 and no estimator can win. That was harsher than MeZO's regime, which is
low-but-nonzero loss on a pretrained model. Here the target is offset so the gradient is
small and real, and the DFC has a GPU so optimisation runs at its proper level.
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
mm = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
mm.ir_version = 9
onnx.save(mm, "/tmp/tiny.onnx")

runner = ClientRunner(hw_arch="hailo10h")
runner.translate_onnx_model("/tmp/tiny.onnx", "tiny", start_node_names=["x"],
                            end_node_names=["y"], net_input_shapes={"x": [1, C, 1, 1]})
runner.load_model_script("quantization_param({*}, precision_mode=a16_w16)\n")
runner.optimize(np.random.randn(1024, 1, 1, C).astype(np.float32))

def out(xs, kind):
    with runner.infer_context(kind) as ctx:
        return np.asarray(runner.infer(ctx, xs)).reshape(len(xs), -1)

X = np.random.randn(B, 1, 1, C).astype(np.float32)
base_q, base_f = out(X, InferenceContext.SDK_QUANTIZED), out(X, InferenceContext.SDK_NATIVE)
# Small but REAL loss: offset the target so the gradient is nonzero, as on a pretrained model.
noise = (np.random.randn(*base_f.shape) * base_f.std() * 0.10).astype(np.float32)
Tq, Tf = base_q + noise, base_f + noise
print(f"a16w16, full optimisation. loss at start: "
      f"quant={np.mean((base_q-Tq)**2):.4e}  float={np.mean((base_f-Tf)**2):.4e}\n")

def zo(eps, Z):
    p, mn = (X + eps * Z).astype(np.float32), (X - eps * Z).astype(np.float32)
    qp, qm = out(p, InferenceContext.SDK_QUANTIZED), out(mn, InferenceContext.SDK_QUANTIZED)
    fp, fm = out(p, InferenceContext.SDK_NATIVE), out(mn, InferenceContext.SDK_NATIVE)
    return (float((np.mean((qp-Tq)**2) - np.mean((qm-Tq)**2)) / (2*eps)),
            float((np.mean((fp-Tf)**2) - np.mean((fm-Tf)**2)) / (2*eps)))

N = 48
print(f"{'eps':>7}  {'sign agree':>11}  {'pearson r':>10}  {'|gq| med':>11}  {'|gf| med':>11}  ratio")
print("-" * 72)
for eps in (3e-2, 1e-2, 3e-3, 1e-3, 1e-4):
    gq, gf = [], []
    for _ in range(N):
        Z = np.random.randn(*X.shape).astype(np.float32)
        Z /= np.linalg.norm(Z.reshape(B, -1), axis=1).reshape(B, 1, 1, 1)
        a, b = zo(eps, Z); gq.append(a); gf.append(b)
    gq, gf = np.array(gq), np.array(gf)
    agree = float((np.sign(gq) == np.sign(gf)).mean())
    r = float(np.corrcoef(gq, gf)[0, 1]) if gq.std() > 0 and gf.std() > 0 else float("nan")
    mq, mf = np.median(np.abs(gq)), np.median(np.abs(gf))
    print(f"{eps:7.0e}  {agree:10.1%}  {r:+10.3f}  {mq:11.3e}  {mf:11.3e}  {mq/mf:7.1f}x")
