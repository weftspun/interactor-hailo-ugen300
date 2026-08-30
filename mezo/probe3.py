"""Best case for MeZO: a16w16. Does the ZO estimate agree with the float one?

Probe 2 showed the aggregated loss moves at every epsilon, so it is not flat -- but the
difference did not scale with epsilon, which means it may be dominated by discretisation
rather than by slope. A difference that moves for the wrong reason is worse than one that
does not move, because it looks like progress. Correlation against the float estimate on
the same z is what separates them.
"""
import numpy as np, onnx, sys
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
try:
    runner.load_model_script("quantization_param({*}, precision_mode=a16_w16)\n")
    print("precision: a16_w16 requested")
except Exception as e:
    print("model script rejected:", type(e).__name__, str(e)[:120]); sys.exit(1)
runner.optimize(np.random.randn(1024, 1, 1, C).astype(np.float32))

def out(xs, kind):
    with runner.infer_context(kind) as ctx:
        return np.asarray(runner.infer(ctx, xs)).reshape(len(xs), -1)

X = np.random.randn(B, 1, 1, C).astype(np.float32)
Tq = out(X, InferenceContext.SDK_QUANTIZED)
Tf = out(X, InferenceContext.SDK_NATIVE)

def zo(eps, Z):
    p, mn = (X + eps * Z).astype(np.float32), (X - eps * Z).astype(np.float32)
    qp, qm = out(p, InferenceContext.SDK_QUANTIZED), out(mn, InferenceContext.SDK_QUANTIZED)
    fp, fm = out(p, InferenceContext.SDK_NATIVE), out(mn, InferenceContext.SDK_NATIVE)
    gq = (np.mean((qp - Tq) ** 2) - np.mean((qm - Tq) ** 2)) / (2 * eps)
    gf = (np.mean((fp - Tf) ** 2) - np.mean((fm - Tf) ** 2)) / (2 * eps)
    return float(gq), float(gf)

N = 24
for eps in (1e-2, 1e-3, 1e-4):
    gq, gf = [], []
    for _ in range(N):
        Z = np.random.randn(*X.shape).astype(np.float32)
        Z /= np.linalg.norm(Z.reshape(B, -1), axis=1).reshape(B, 1, 1, 1)
        a, b = zo(eps, Z)
        gq.append(a); gf.append(b)
    gq, gf = np.array(gq), np.array(gf)
    agree = float((np.sign(gq) == np.sign(gf)).mean())
    r = float(np.corrcoef(gq, gf)[0, 1]) if gq.std() > 0 and gf.std() > 0 else float("nan")
    print(f"eps={eps:.0e}  n={N}  sign agreement={agree:5.1%}  pearson r={r:+.3f}  "
          f"|gq| median={np.median(np.abs(gq)):.3e}  |gf| median={np.median(np.abs(gf)):.3e}")
