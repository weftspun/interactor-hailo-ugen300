"""QZO on the Hailo emulator: perturb quantisation scales, never the discrete weights.

arXiv 2505.13430. MeZO perturbs weights, which on this hardware are integers, and probe 1
showed the response is a staircase. QZO perturbs the CONTINUOUS scale factors instead. The
weights stay fixed and discrete; only the multiplier moves, so the loss responds smoothly by
construction and the search space is a handful of parameters per layer rather than millions.

Step: g = clip([L(s+ez) - L(s-ez)] / 2e), then s <- s - lr*g*z, with z regenerated from a
seed rather than stored. Clipping is QZO's stabiliser -- a ZO estimate divided by a small
epsilon blows up, and probe 6 measured exactly that failure on this hardware.
"""
import time
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext
from hailo_sdk_client.hailo_archive.hailo_archive import ParamsKinds

C, H, O, B = 256, 256, 64, 64
LOG = open("/work/qzo_results.txt", "w", buffering=1)


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

# --- which parameter kind carries the scales? measured, not guessed --------------------
KIND = None
for name in ("HAILO_OPTIMIZED", "TRANSLATED", "FP_OPTIMIZED", "NATIVE"):
    kind = getattr(ParamsKinds, name)
    try:
        runner.save_params("/work/q_%s.npz" % name, kind)
        zz = np.load("/work/q_%s.npz" % name, allow_pickle=True)
        ks = list(zz.keys())
        hits = [k for k in ks if any(w in k.lower() for w in
                ("scale", "zp", "zero", "limvals", "shift", "range", "offset"))]
        say("%-16s %3d arrays  scale-like: %s" % (name, len(ks), hits[:6] or "none"))
        if hits and KIND is None:
            KIND = (name, kind)
    except Exception as e:
        say("%-16s unavailable: %s" % (name, str(e)[:70]))

if KIND is None:
    say("")
    say("No parameter kind exposes a scale. QZO has nothing continuous to perturb.")
    raise SystemExit(0)

say("")
say("using %s" % KIND[0])
runner.save_params("/work/q.npz", KIND[1])
PARAMS_KIND = KIND[1]
z = np.load("/work/q.npz", allow_pickle=True)
keys = list(z.keys())
for k in keys:
    a = z[k]
    say("    %-44s %-18s %s" % (k, str(a.shape), a.dtype))

SCALE_WORDS = ("scale", "zp", "zero", "limvals", "shift", "range", "offset")
scale_keys = [k for k in keys if any(w in k.lower() for w in SCALE_WORDS)]
say("")
say("scale-like keys: %s" % (scale_keys or "NONE -- QZO has nothing to perturb here"))

if not scale_keys:
    say("")
    say("Falling back: perturbing every float array that is small enough to be a scale,")
    say("since the naming may not carry the word.")
    scale_keys = [k for k in keys
                  if z[k].dtype.kind == "f" and z[k].size <= max(C, H, O) * 4]
    say("candidates by shape: %s" % (scale_keys or "NONE"))

if not scale_keys:
    say("QZO cannot proceed: no continuous parameter exposed in the quantised set.")
    raise SystemExit(0)


def out(xs):
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        return np.asarray(runner.infer(ctx, xs)).reshape(len(xs), -1)


np.random.seed(99)
X = np.random.randn(B, 1, 1, C).astype(np.float32)
base = out(X)
T = base + (np.random.randn(*base.shape) * base.std() * 0.10).astype(np.float32)


def loss():
    return float(np.mean((out(X) - T) ** 2))


params = {k: np.array(z[k], dtype=np.float64) for k in scale_keys}
others = {k: z[k] for k in keys if k not in scale_keys}


def write(pert):
    d = dict(others)
    for k in scale_keys:
        d[k] = (params[k] + pert.get(k, 0.0)).astype(z[k].dtype)
    np.savez("/work/mod.npz", **d)
    runner.load_params("/work/mod.npz", params_kind=PARAMS_KIND)


def draw(seed):
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal(params[k].shape) for k in scale_keys}


EPS, LR, CLIP, STEPS = 1e-2, 5e-3, 1.0, 30
write({})
say("")
say("QZO: %d scale arrays, %d values, eps=%g lr=%g clip=%g"
    % (len(scale_keys), sum(params[k].size for k in scale_keys), EPS, LR, CLIP))
say("")
say("step        loss       g_raw     g_clipped")
l0 = loss()
say("%4d  %10.6e" % (0, l0))
t0 = time.time()
for step in range(1, STEPS + 1):
    zs = draw(2000 + step)
    write({k: EPS * v for k, v in zs.items()})
    lp = loss()
    write({k: -EPS * v for k, v in zs.items()})
    lm = loss()
    graw = (lp - lm) / (2 * EPS)
    gcl = float(np.clip(graw, -CLIP, CLIP))
    for k in scale_keys:
        params[k] -= LR * gcl * zs[k]
    write({})
    if step % 5 == 0 or step == 1:
        say("%4d  %10.6e  %10.3e  %10.3e" % (step, loss(), graw, gcl))
say("")
say("start %.6e -> end %.6e   delta %+.3e   %.1fs for %d steps"
    % (l0, loss(), l0 - loss(), time.time() - t0, STEPS))
