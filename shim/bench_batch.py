"""Does queueing more frames buy anything, and where does it stop buying.

One synchronous inference measured 1.4 ms, nearly all of it a USB round trip rather than
compute. If that is right, throughput should climb with queue depth until it hits either the
device's own async queue size or the link, and per-frame latency should fall toward the
compute time. If it does not climb, the round trip is not the cost and the theory is wrong.

The control is depth 1 through the same batch entry point, so the comparison is not against
a different code path.
"""
import ctypes, os, statistics, sys, time
import numpy as np
from sigs_ctypes import bind

HERE = os.path.dirname(os.path.abspath(__file__))
os.add_dll_directory(r"C:\Program Files\HailoRT\bin")
_, F, _ = bind(os.path.join(HERE, "hailort_shim.dll"), os.path.join(HERE, "hailort_shim.sigs"))
F["hs_last_error"].restype = ctypes.c_char_p
F["hs_open_ex"].restype = ctypes.c_void_p


def main(hef, depths=(1, 2, 4, 8, 16, 32, 64), reps=20):
    h = F["hs_open_ex"](hef.encode(), 1)
    if not h:
        print("open failed:", F["hs_last_error"]().decode()); return 1
    h = ctypes.c_void_p(h)
    n_in, n_out = F["hs_input_size"](h), F["hs_output_size"](h)
    q = F["hs_queue_size"](h)
    print("%s: in %d B, out %d B, device async queue depth %d" % (hef, n_in, n_out, q))
    print()
    print("%6s %11s %13s %12s %10s" % ("depth", "wall ms", "ms/frame", "frames/s", "vs depth 1"))
    print("-" * 58)

    base = None
    for d in depths:
        xin = np.ascontiguousarray(
            np.random.default_rng(1171).standard_normal((d, n_in // 4)).astype(np.float32))
        y = np.zeros(d * n_out // 4, dtype=np.float32)

        def run():
            st = F["hs_infer_batch"](h, xin.ctypes.data_as(ctypes.c_void_p),
                                     y.ctypes.data_as(ctypes.c_void_p), d, 20000)
            if st != 0:
                raise RuntimeError("status=%d %s" % (st, F["hs_last_error"]().decode()))

        run()
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); run(); ts.append((time.perf_counter() - t) * 1e3)
        wall = statistics.median(ts)
        per = wall / d
        fps = 1000.0 / per
        if base is None:
            base = per
        print("%6d %11.3f %13.4f %12.0f %9.2fx" % (d, wall, per, fps, base / per))

    F["hs_close"](h)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "rb_normal.hef"))
