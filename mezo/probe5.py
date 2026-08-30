"""MeZO viability: a16w16, full optimisation, small-but-real loss, batched.

Probe 4 opened one infer_context per call -- 960 GPU sessions for 960 inferences, which is
session setup with arithmetic hidden inside it. Here every perturbation for one epsilon
goes through in a single call: 10 contexts instead of 960.

The target is offset from the model's own output so the loss is small but nonzero, which
is MeZO's actual regime; probe 3 sat at an exact minimum where no estimator can win.
"""
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext

np.random.seed(1171)
C, H, O, B, N = 256, 256, 64, 64, 48
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
print("optimised at a16_w16\n", flush=True)

def out(xs, kind):
    with runner.infer_context(kind) as ctx:
        return np.asarray(runner.infer(ctx, xs)).reshape(len(xs), -1)

X = np.random.randn(B, 1, 1, C).astype(np.float32)
bq, bf = out(X, InferenceContext.SDK_QUANTIZED), out(X, InferenceContext.SDK_NATIVE)
noise = (np.random.randn(*bf.shape) * bf.std() * 0.10).astype(np.float32)
Tq, Tf = bq + noise, bf + noise
print(f"start loss  quant={np.mean((bq-Tq)**2):.4e}  float={np.mean((bf-Tf)**2):.4e}\n", flush=True)

Zs = np.random.randn(N, *X.shape).astype(np.float32)
Zs /= np.linalg.norm(Zs.reshape(N, B, -1), axis=2).reshape(N, B, 1, 1, 1)

print(f"{'eps':>7}  {'sign agree':>11}  {'pearson r':>10}  {'|gq| med':>11}  {'|gf| med':>11}  ratio", flush=True)
print("-" * 72, flush=True)
for eps in (3e-2, 1e-2, 3e-3, 1e-3, 1e-4):
    # every draw, both signs, one batch
    big = np.concatenate([X + eps * Zs[i] for i in range(N)] +
                         [X - eps * Zs[i] for i in range(N)]).astype(np.float32)
    q = out(big, InferenceContext.SDK_QUANTIZED).reshape(2 * N, B, -1)
    f = out(big, InferenceContext.SDK_NATIVE).reshape(2 * N, B, -1)
    gq = (((q[:N] - Tq) ** 2).mean((1, 2)) - ((q[N:] - Tq) ** 2).mean((1, 2))) / (2 * eps)
    gf = (((f[:N] - Tf) ** 2).mean((1, 2)) - ((f[N:] - Tf) ** 2).mean((1, 2))) / (2 * eps)
    agree = float((np.sign(gq) == np.sign(gf)).mean())
    r = float(np.corrcoef(gq, gf)[0, 1]) if gq.std() > 0 and gf.std() > 0 else float("nan")
    mq, mf = np.median(np.abs(gq)), np.median(np.abs(gf))
    print(f"{eps:7.0e}  {agree:10.1%}  {r:+10.3f}  {mq:11.3e}  {mf:11.3e}  "
          f"{mq/mf if mf else float('inf'):7.1f}x", flush=True)
