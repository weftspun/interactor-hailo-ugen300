"""Rung 4, part one: compile three HEFs that differ in known, controlled ways.

  base     a graph with weights W
  weights  the same architecture with different weights          -> outputs must differ
  scale    the SAME weights, only a forced quantisation range    -> QZO's knob

The third is the one that matters. force_range_out changes a layer's scale while the
discrete weights stay bit-identical, which is exactly what QZO perturbs. If base and scale
produce different device outputs, the QZO knob reaches silicon; if they do not, it never
will and the compile-per-generation design is pointless.

The kernel hash is asserted across base and scale so any difference is attributable to the
scale and nothing else.
"""
import hashlib
import time

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.hailo_archive.hailo_archive import ParamsKinds

C, O = 32, 16
LOG = open("/work/rung4_build.txt", "w", buffering=1)


def say(s):
    print(s, flush=True)
    LOG.write(s + "\n")


def build_onnx(seed, path):
    rng = np.random.default_rng(seed)
    W1 = (rng.standard_normal((C, 3, 3, 3)) * 0.1).astype(np.float32)
    W2 = (rng.standard_normal((O, C, 1, 1)) * 0.1).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("Conv", ["x", "W1"], ["h0"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
         helper.make_node("Relu", ["h0"], ["h1"]),
         helper.make_node("Conv", ["h1", "W2"], ["y"], kernel_shape=[1, 1])],
        "lad",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, O, 16, 16])],
        [numpy_helper.from_array(W1, "W1"), numpy_helper.from_array(W2, "W2")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)


CAL = np.random.default_rng(7).standard_normal((512, 16, 16, 3)).astype(np.float32)


def compile_variant(seed, force_out, tag):
    src = "/tmp/lad_%s.onnx" % tag
    build_onnx(seed, src)
    r = ClientRunner(hw_arch="hailo10h")
    r.translate_onnx_model(src, "lad", start_node_names=["x"], end_node_names=["y"],
                           net_input_shapes={"x": [1, 3, 16, 16]})
    script = "quantization_param({*}, precision_mode=a16_w16)\n"
    if force_out is not None:
        script += "quantization_param(lad/conv1, force_range_out=[%.8f, %.8f])\n" % force_out
    r.load_model_script(script)
    t = time.time()
    r.optimize(CAL)
    t_op = time.time() - t

    r.save_params("/work/p_%s.npz" % tag, ParamsKinds.TRANSLATED)
    z = np.load("/work/p_%s.npz" % tag, allow_pickle=True)
    kern = hashlib.sha256(b"".join(np.ascontiguousarray(z[k]).tobytes()
                                   for k in sorted(z) if k.endswith("kernel:0"))).hexdigest()[:16]
    scales = [float(z[k].mean()) for k in z if "conv1" in k and "output_scales" in k]

    t = time.time()
    hef = r.compile()
    t_co = time.time() - t
    open("/work/lad_%s.hef" % tag, "wb").write(hef)
    return {"tag": tag, "kernels": kern, "scale": scales[0] if scales else float("nan"),
            "hef_sha": hashlib.sha256(hef).hexdigest()[:16], "bytes": len(hef),
            "optimize_s": t_op, "compile_s": t_co}


say("%-9s %-16s %-16s %12s %9s %8s %8s"
    % ("variant", "kernels sha", "hef sha", "conv1 scale", "hef bytes", "opt s", "comp s"))
say("-" * 88)
rows = []
for seed, force, tag in ((1171, None, "base"),
                         (2026, None, "weights"),
                         (1171, (-3.0 * 1.25, 3.0 * 1.25), "scale")):
    row = compile_variant(seed, force, tag)
    rows.append(row)
    say("%-9s %-16s %-16s %12.6g %9d %8.1f %8.1f"
        % (row["tag"], row["kernels"], row["hef_sha"], row["scale"],
           row["bytes"], row["optimize_s"], row["compile_s"]))

base, weights, scale = rows
say("")
say("base vs weights: kernels %s"
    % ("differ, as intended" if base["kernels"] != weights["kernels"] else "SAME -- broken setup"))
say("base vs scale:   kernels %s"
    % ("BIT-IDENTICAL, so any output difference is the scale alone"
       if base["kernels"] == scale["kernels"] else "DIFFER -- the scale test is confounded"))
say("base vs scale:   hef %s"
    % ("differs, the forced range reached the compiler"
       if base["hef_sha"] != scale["hef_sha"] else "IDENTICAL -- force_range_out did nothing"))
