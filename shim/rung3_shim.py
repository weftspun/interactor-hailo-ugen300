"""Rung 3: a chosen input through the UGen300, and the output read back.

Bound from hailort_shim.sigs through sigs_ctypes.bind -- the same declarative-ABI
convention the workspace uses for iceoryx2 and openvr, now over a shim because the
Hailo-10H rejects HailoRT's own flat C API and only supports the C++ InferModel.

Three claims, each falsifiable:
  different inputs give different outputs   -- the device computed on what I sent
  the same input twice gives the same bytes -- it is deterministic
  a wrong-sized buffer is refused           -- the size check is real, not decorative
"""
import ctypes
import hashlib
import os
import sys

import numpy as np
from sigs_ctypes import bind

HERE = os.path.dirname(os.path.abspath(__file__))
os.add_dll_directory(r"C:\Program Files\HailoRT\bin")
_, F, _ = bind(os.path.join(HERE, "hailort_shim.dll"),
               os.path.join(HERE, "hailort_shim.sigs"))
F["hs_last_error"].restype = ctypes.c_char_p
F["hs_open"].restype = ctypes.c_void_p


def err():
    return F["hs_last_error"]().decode(errors="replace")


def main(hef):
    h = F["hs_open"](hef.encode())
    if not h:
        print("hs_open failed: %s" % err())
        return 1
    h = ctypes.c_void_p(h)
    n_in = F["hs_input_size"](h)
    n_out = F["hs_output_size"](h)
    print("opened %s" % os.path.basename(hef))
    print("frame sizes: in %d bytes, out %d bytes" % (n_in, n_out))

    def infer(seed):
        x = np.random.default_rng(seed).integers(0, 256, n_in, dtype=np.uint8)
        y = np.zeros(n_out, dtype=np.uint8)
        st = F["hs_infer"](h, x.ctypes.data_as(ctypes.c_void_p), n_in,
                           y.ctypes.data_as(ctypes.c_void_p), n_out, 10000)
        if st != 0:
            raise RuntimeError("hs_infer status=%d: %s" % (st, err()))
        return y

    a, b = infer(1171), infer(2026)
    for tag, y in (("seed 1171", a), ("seed 2026", b)):
        print("%-10s output mean %7.3f  argmax %4d  sha %s"
              % (tag, y.mean(), int(y.argmax()), hashlib.sha256(y.tobytes()).hexdigest()[:16]))

    differ = not np.array_equal(a, b)
    print("")
    print("different inputs -> %s"
          % ("different outputs, the device computed on what I sent" if differ
             else "IDENTICAL OUTPUT -- the device is not seeing my input"))

    a2 = infer(1171)
    print("same input twice -> %s"
          % ("bit-identical, deterministic" if np.array_equal(a, a2) else "DIFFERS"))

    # Negative control: a wrong-sized buffer must be refused, or the size check is
    # decoration and a real mismatch would read past the end of an array instead.
    y = np.zeros(n_out, dtype=np.uint8)
    x_short = np.zeros(max(1, n_in - 1), dtype=np.uint8)
    st = F["hs_infer"](h, x_short.ctypes.data_as(ctypes.c_void_p), n_in - 1,
                       y.ctypes.data_as(ctypes.c_void_p), n_out, 10000)
    print("undersized input -> %s"
          % ("refused, status=%d" % st if st != 0
             else "ACCEPTED -- the size check does not work"))

    F["hs_close"](h)
    print("closed")
    return 0 if differ else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else os.path.join(HERE, "efficientnet_lite0.hef")))
