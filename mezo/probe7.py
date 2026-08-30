"""Rank-based ES instead of MeZO: does quantisation preserve ORDER when it ruins magnitude?

MeZO consumes [L+ - L-]/2e, a magnitude, and probe 6 showed the quantised magnitude wrong by
10^5. Rank-based evolution strategies consume only which member of a population scored
better. If the quantised graph orders a population the same way the float graph does, ES
adapts where MeZO cannot -- the ruined quantity is not the one being used.

Spearman is the statistic, not Pearson. Reported for both emulation contexts, because they
disagree and neither is an oracle for the other.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext

C, H, O, B, P = 256, 256, 64, 64, 96
LOG = open("/work/results_es.txt", "w", buffering=1)


def say(s):
    print(s, flush=True)
    LOG.write(s + "\n")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


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


def out(xs, kind):
    with runner.infer_context(kind) as ctx:
        return np.asarray(runner.infer(ctx, xs)).reshape(len(xs), -1)


np.random.seed(99)
X = np.random.randn(B, 1, 1, C).astype(np.float32)
bf = out(X, InferenceContext.SDK_NATIVE)
noise = (np.random.randn(*bf.shape) * bf.std() * 0.10).astype(np.float32)
Tf = bf + noise

for ctx_kind in (InferenceContext.SDK_QUANTIZED, InferenceContext.SDK_BIT_EXACT):
    try:
        Tq = out(X, ctx_kind) + noise
        say("")
        say("=== context=%s, precision=a16_w16, population=%d ===" % (ctx_kind.name, P))
        say("      eps   spearman   top-1 match   top-10 overlap   sign agree")
        for eps in (3e-2, 1e-2, 1e-3, 1e-4):
            Z = np.random.randn(P, *X.shape).astype(np.float32)
            Z /= np.linalg.norm(Z.reshape(P, B, -1), axis=2).reshape(P, B, 1, 1, 1)
            big = np.concatenate([X + eps * Z[i] for i in range(P)]).astype(np.float32)
            q = out(big, ctx_kind).reshape(P, B, -1)
            f = out(big, InferenceContext.SDK_NATIVE).reshape(P, B, -1)
            Lq = ((q - Tq) ** 2).mean((1, 2))
            Lf = ((f - Tf) ** 2).mean((1, 2))
            rho = spearman(Lq, Lf)
            top1 = int(np.argmin(Lq) == np.argmin(Lf))
            ov = len(set(np.argsort(Lq)[:10]) & set(np.argsort(Lf)[:10])) / 10.0
            # the ES update direction: does it point the same way?
            wq = (Lq - Lq.mean()) / (Lq.std() + 1e-30)
            wf = (Lf - Lf.mean()) / (Lf.std() + 1e-30)
            sa = float((np.sign(wq) == np.sign(wf)).mean())
            say("   %7.0e   %+8.3f   %11s   %13.0f%%   %9.1f%%"
                % (eps, rho, "yes" if top1 else "no", ov * 100, sa * 100))
    except Exception as e:
        say("")
        say("=== context=%s FAILED %s: %s" % (ctx_kind.name, type(e).__name__, str(e)[:160]))
