"""Throughput against frame size, uint8 -- what a real RGBA frame costs to move.

The small block was overhead-bound. If the link is the limit at large frames, MB/s should
flatten and ms/frame should grow in proportion to the bytes. Reported as achieved MB/s so
the two regimes are comparable on one axis.
"""
import ctypes, os, statistics, time
import numpy as np
from sigs_ctypes import bind

HERE = os.path.dirname(os.path.abspath(__file__))
os.add_dll_directory(r"C:\Program Files\HailoRT\bin")
_, F, _ = bind(os.path.join(HERE, "hailort_shim.dll"), os.path.join(HERE, "hailort_shim.sigs"))
F["hs_last_error"].restype = ctypes.c_char_p
F["hs_open_ex"].restype = ctypes.c_void_p

print("%-7s %10s %9s %8s %11s %11s %10s"
      % ("size", "in MB", "out MB", "depth", "ms/frame", "frames/s", "MB/s"))
print("-" * 74)
for tag in ("512p", "720p", "1080p", "4K"):
    hef = "big_%s.hef" % tag
    h = F["hs_open_ex"](hef.encode(), 0)
    if not h:
        print("%-7s open failed: %s" % (tag, F["hs_last_error"]().decode())); continue
    h = ctypes.c_void_p(h)
    n_in, n_out = F["hs_input_size"](h), F["hs_output_size"](h)
    for depth in (1, 4):
        xin = np.ascontiguousarray(
            np.random.default_rng(1).integers(0, 256, (depth, n_in), dtype=np.uint8))
        y = np.zeros(depth * n_out, dtype=np.uint8)

        def run():
            st = F["hs_infer_batch"](h, xin.ctypes.data_as(ctypes.c_void_p),
                                     y.ctypes.data_as(ctypes.c_void_p), depth, 60000)
            if st != 0:
                raise RuntimeError("status=%d %s" % (st, F["hs_last_error"]().decode()))
        try:
            run()
            ts = []
            for _ in range(8):
                t = time.perf_counter(); run(); ts.append((time.perf_counter() - t) * 1e3)
            wall = statistics.median(ts)
            per = wall / depth
            mbps = (n_in + n_out) / 1e6 / (per / 1000.0)
            print("%-7s %10.2f %9.2f %8d %11.3f %11.1f %10.1f"
                  % (tag, n_in / 1e6, n_out / 1e6, depth, per, 1000 / per, mbps))
        except Exception as e:
            print("%-7s %10.2f %9.2f %8d  FAILS: %s" % (tag, n_in/1e6, n_out/1e6, depth, e))
    F["hs_close"](h)
