"""Rungs 1-3 of the QZO-on-device ladder: does forcing a range move anything?

QZO perturbs quantisation scales while the discrete weights stay fixed. load_params was
proven inert -- halving the int16 kernels left the output bit-identical -- so the only
supported way to set a scale is force_range_out in the model script, applied before
optimize. That is a compile-time knob, which is fine: it is what an idle-time loop would
turn between generations anyway.

Rung 1  a forced range changes the compiled HEF bytes
Rung 2  it changes the emulator output, and does so smoothly as the range is swept
Rung 3  the HEFs are written for hailortcli to run on the device

The weights are asserted bit-identical across every variant, so any output difference is
attributable to the scale and nothing else.
"""
import hashlib
import time
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext
from hailo_sdk_client.hailo_archive.hailo_archive import ParamsKinds

C, O, B = 32, 16, 8
LOG = open("/work/rung123.txt", "w", buffering=1)


def say(s):
    print(s, flush=True)
    LOG.write(s + "\n")


rng = np.random.default_rng(1171)
W1 = (rng.standard_normal((C, 3, 3, 3)) * 0.1).astype(np.float32)
W2 = (rng.standard_normal((O, C, 1, 1)) * 0.1).astype(np.float32)
g = helper.make_graph(
    [helper.make_node("Conv", ["x", "W1"], ["h0"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
     helper.make_node("Relu", ["h0"], ["h1"]),
     helper.make_node("Conv", ["h1", "W2"], ["y"], kernel_shape=[1, 1])],
    "lad",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, O, 8, 8])],
    [numpy_helper.from_array(W1, "W1"), numpy_helper.from_array(W2, "W2")])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 9
onnx.checker.check_model(m)
onnx.save(m, "/tmp/lad.onnx")

CAL = np.random.default_rng(7).standard_normal((512, 8, 8, 3)).astype(np.float32)
X = np.random.default_rng(99).standard_normal((B, 8, 8, 3)).astype(np.float32)


def variant(force_out, tag):
    r = ClientRunner(hw_arch="hailo10h")
    r.translate_onnx_model("/tmp/lad.onnx", "lad", start_node_names=["x"],
                           end_node_names=["y"], net_input_shapes={"x": [1, 3, 8, 8]})
    script = "quantization_param({*}, precision_mode=a16_w16)\n"
    if force_out is not None:
        script += "quantization_param(lad/conv1, force_range_out=[%.8f, %.8f])\n" % force_out
    r.load_model_script(script)
    r.optimize(CAL)

    with r.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        y = np.asarray(r.infer(ctx, X)).reshape(B, -1)

    r.save_params("/work/p_%s.npz" % tag, ParamsKinds.TRANSLATED)
    z = np.load("/work/p_%s.npz" % tag, allow_pickle=True)
    kern = hashlib.sha256(b"".join(np.ascontiguousarray(z[k]).tobytes()
                                   for k in sorted(z.keys()) if k.endswith("kernel:0"))).hexdigest()[:16]
    scl = float(np.mean([z[k].mean() for k in z.keys()
                         if "output_scales" in k and "conv1" in k] or [np.nan]))
    hef = r.compile()
    open("/work/lad_%s.hef" % tag, "wb").write(hef)
    return y, kern, scl, hashlib.sha256(hef).hexdigest()[:16], len(hef)


base_y, base_kern, base_scl, base_hef, base_n = variant(None, "base")
say("baseline: kernels sha=%s  conv1 out_scale mean=%.6g  hef sha=%s  %d bytes"
    % (base_kern, base_scl, base_hef, base_n))
say("")
say("%10s  %-16s  %-16s  %12s  %12s  %s"
    % ("range x", "kernels sha", "hef sha", "out_scale", "max|dy|", "verdict"))
say("-" * 92)

# conv1's own learned range, scaled. Sweeping it is sweeping the scale, which is QZO's knob.
lo, hi = -3.0, 3.0
for mult in (1.00, 1.02, 1.05, 1.10, 1.25, 1.50):
    y, kern, scl, hsha, n = variant((lo * mult, hi * mult), "m%03d" % int(mult * 100))
    d = float(np.abs(y - base_y).max())
    same_w = "weights SAME" if kern == base_kern else "WEIGHTS CHANGED -- confound"
    moved = "hef differs" if hsha != base_hef else "HEF IDENTICAL"
    say("%10.2f  %-16s  %-16s  %12.6g  %12.4e  %s / %s"
        % (mult, kern, hsha, scl, d, same_w, moved))

say("")
say("HEFs written to /work/lad_*.hef for rung 3")
