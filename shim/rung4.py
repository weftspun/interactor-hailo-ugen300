"""Rung 4: do two HEFs differing only in weights give different device outputs?

That is the claim an on-device scoring loop rests on -- the host changes something, compiles,
and the silicon reports a different answer. Without it, evolution or QZO on the device is
scoring noise.

The `scale` variant is reported but NOT attributed: rung4_build showed force_range_out also
re-quantises the kernels, so its difference is scale AND weights together.
"""
import ctypes, hashlib, os
import numpy as np
from sigs_ctypes import bind

HERE = os.path.dirname(os.path.abspath(__file__))
os.add_dll_directory(r"C:\Program Files\HailoRT\bin")
_, F, _ = bind(os.path.join(HERE, "hailort_shim.dll"), os.path.join(HERE, "hailort_shim.sigs"))
F["hs_last_error"].restype = ctypes.c_char_p
F["hs_open"].restype = ctypes.c_void_p

def run(hef, seed):
    h = F["hs_open"](hef.encode())
    if not h:
        raise RuntimeError("%s: %s" % (hef, F["hs_last_error"]().decode()))
    h = ctypes.c_void_p(h)
    n_in, n_out = F["hs_input_size"](h), F["hs_output_size"](h)
    x = np.random.default_rng(seed).integers(0, 256, n_in, dtype=np.uint8)
    y = np.zeros(n_out, dtype=np.uint8)
    st = F["hs_infer"](h, x.ctypes.data_as(ctypes.c_void_p), n_in,
                       y.ctypes.data_as(ctypes.c_void_p), n_out, 10000)
    F["hs_close"](h)
    if st != 0:
        raise RuntimeError("infer status=%d: %s" % (st, F["hs_last_error"]().decode()))
    return n_in, n_out, y

SEED = 4242
res = {}
for tag in ("base", "weights", "scale"):
    p = os.path.join(HERE, "lad_%s.hef" % tag)
    n_in, n_out, y = run(p, SEED)
    res[tag] = y
    print("%-8s in %d out %d  mean %7.3f  sha %s"
          % (tag, n_in, n_out, y.mean(), hashlib.sha256(y.tobytes()).hexdigest()[:16]))

print()
d_w = int((res["base"] != res["weights"]).sum())
print("base vs weights: %d of %d bytes differ -> %s"
      % (d_w, res["base"].size,
         "the device reflects a weight change" if d_w else "IDENTICAL -- it does not"))
d_s = int((res["base"] != res["scale"]).sum())
print("base vs scale:   %d of %d bytes differ  (CONFOUNDED: kernels also re-quantised)"
      % (d_s, res["base"].size))
_, _, again = run(os.path.join(HERE, "lad_base.hef"), SEED)
print("base rerun:      %s"
      % ("bit-identical" if np.array_equal(res["base"], again) else "DIFFERS"))
