"""Three probes: does the MeZO result survive bit-exact emulation and deployable precisions?

Probe 5 used SDK_QUANTIZED at a16_w16 -- the quantisation model at the highest precision.
SDK_BIT_EXACT is the context whose name claims agreement with the device, and a16_w4 and
a8_w8 are what a real deployment would run. If the correlation holds across all three the
result is about the hardware; if it only holds at a16_w16 under SDK_QUANTIZED it was about
the emulator's most forgiving corner.

Results go to /work/results.txt as well as stdout, because a stdout filter discarded the
first two probes of the previous attempt.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext

C, H, O, B, N = 256, 256, 64, 64, 48
LOG = open("/work/results.txt", "w", buffering=1)


def say(s):
    print(s, flush=True)
    LOG.write(s + "\n")


def build():
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


def run(precision, ctx_kind, label):
    build()
    r = ClientRunner(hw_arch="hailo10h")
    r.translate_onnx_model("/tmp/tiny.onnx", "tiny", start_node_names=["x"],
                           end_node_names=["y"], net_input_shapes={"x": [1, C, 1, 1]})
    r.load_model_script("quantization_param({*}, precision_mode=%s)\n" % precision)
    r.optimize(np.random.randn(1024, 1, 1, C).astype(np.float32))

    def out(xs, kind):
        with r.infer_context(kind) as ctx:
            return np.asarray(r.infer(ctx, xs)).reshape(len(xs), -1)

    np.random.seed(99)
    X = np.random.randn(B, 1, 1, C).astype(np.float32)
    bq, bf = out(X, ctx_kind), out(X, InferenceContext.SDK_NATIVE)
    noise = (np.random.randn(*bf.shape) * bf.std() * 0.10).astype(np.float32)
    Tq, Tf = bq + noise, bf + noise
    Zs = np.random.randn(N, *X.shape).astype(np.float32)
    Zs /= np.linalg.norm(Zs.reshape(N, B, -1), axis=2).reshape(N, B, 1, 1, 1)

    say("")
    say("=== %s: precision=%s, context=%s ===" % (label, precision, ctx_kind.name))
    say("    start loss quant=%.4e float=%.4e"
        % (np.mean((bq - Tq) ** 2), np.mean((bf - Tf) ** 2)))
    say("        eps     sign         r        |gq|        |gf|  ratio")
    for eps in (3e-2, 1e-2, 1e-3):
        big = np.concatenate([X + eps * Zs[i] for i in range(N)]
                             + [X - eps * Zs[i] for i in range(N)]).astype(np.float32)
        q = out(big, ctx_kind).reshape(2 * N, B, -1)
        f = out(big, InferenceContext.SDK_NATIVE).reshape(2 * N, B, -1)
        gq = (((q[:N] - Tq) ** 2).mean((1, 2)) - ((q[N:] - Tq) ** 2).mean((1, 2))) / (2 * eps)
        gf = (((f[:N] - Tf) ** 2).mean((1, 2)) - ((f[N:] - Tf) ** 2).mean((1, 2))) / (2 * eps)
        ag = float((np.sign(gq) == np.sign(gf)).mean())
        rr = float(np.corrcoef(gq, gf)[0, 1]) if gq.std() > 0 and gf.std() > 0 else float("nan")
        mq, mf = np.median(np.abs(gq)), np.median(np.abs(gf))
        ratio = mq / mf if mf else float("inf")
        say("    %7.0e  %6.1f%%  %+8.3f  %10.3e  %10.3e  %6.1fx"
            % (eps, ag * 100, rr, mq, mf, ratio))


for prec, label in (("a16_w16", "probe 1, best case"),
                    ("a16_w4", "probe 2, LLM deployment shape"),
                    ("a8_w8", "probe 3, DFC default")):
    try:
        run(prec, InferenceContext.SDK_BIT_EXACT, label)
    except Exception as e:
        say("")
        say("=== %s: %s FAILED %s: %s" % (label, prec, type(e).__name__, str(e)[:200]))
