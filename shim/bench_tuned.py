"""Which software knob actually moves the 4K number: DMA mapping, ultra power, or neither.

Four arms over the same HEF and the same buffers. DMA mapping pins caller memory so HailoRT
stops copying each frame through a bounce buffer -- invisible at 16 KB, potentially most of
the cost at 33 MB. Ultra performance is a firmware power state. Both are set before
configure, so each arm opens the device fresh.
"""
import ctypes, os, statistics, sys, time
import numpy as np
from sigs_ctypes import bind

HERE = os.path.dirname(os.path.abspath(__file__))
os.add_dll_directory(r"C:\Program Files\HailoRT\bin")
_, F, _ = bind(os.path.join(HERE, "hailort_shim.dll"), os.path.join(HERE, "hailort_shim.sigs"))
F["hs_last_error"].restype = ctypes.c_char_p
for fn in ("hs_open_ex", "hs_open_tuned"):
    F[fn].restype = ctypes.c_void_p


def arm(hef, dma, ultra, depth, reps=8):
    h = F["hs_open_tuned"](hef.encode(), 0, 1 if ultra else 0, 0)
    if not h:
        return None, "open failed: %s" % F["hs_last_error"]().decode()
    h = ctypes.c_void_p(h)
    n_in, n_out = F["hs_input_size"](h), F["hs_output_size"](h)
    xin = np.ascontiguousarray(
        np.random.default_rng(1).integers(0, 256, (depth, n_in), dtype=np.uint8))
    y = np.zeros(depth * n_out, dtype=np.uint8)
    pin, pout = xin.ctypes.data_as(ctypes.c_void_p), y.ctypes.data_as(ctypes.c_void_p)

    mapped = False
    if dma:
        a = F["hs_dma_map"](h, pin, xin.nbytes, 1)
        b = F["hs_dma_map"](h, pout, y.nbytes, 0)
        mapped = (a == 0 and b == 0)
        if not mapped:
            F["hs_close"](h)
            return None, "dma_map failed h2d=%d d2h=%d: %s" % (a, b, F["hs_last_error"]().decode())

    def run():
        st = F["hs_infer_batch"](h, pin, pout, depth, 60000)
        if st != 0:
            raise RuntimeError("status=%d %s" % (st, F["hs_last_error"]().decode()))

    try:
        run()
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); run(); ts.append((time.perf_counter() - t) * 1e3)
        per = statistics.median(ts) / depth
        mbps = (n_in + n_out) / 1e6 / (per / 1000.0)
        res = (per, mbps)
    except Exception as e:
        res, mbps = None, None
        return None, str(e)
    finally:
        if mapped:
            F["hs_dma_unmap"](h, pin, xin.nbytes, 1)
            F["hs_dma_unmap"](h, pout, y.nbytes, 0)
        F["hs_close"](h)
    return res, "ok"


def main(hef="big_4K.hef", depth=4):
    print("%s at depth %d" % (hef, depth))
    print("%-22s %11s %10s %10s" % ("arm", "ms/frame", "MB/s", "vs base"))
    print("-" * 58)
    base = None
    for tag, dma, ultra in (("baseline", 0, 0), ("ultra power", 0, 1),
                            ("dma mapped", 1, 0), ("dma + ultra", 1, 1)):
        res, msg = arm(hef, dma, ultra, depth)
        if res is None:
            print("%-22s %s" % (tag, msg)); continue
        per, mbps = res
        if base is None:
            base = per
        print("%-22s %11.3f %10.1f %9.2fx" % (tag, per, mbps, base / per))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "big_4K.hef",
         int(sys.argv[2]) if len(sys.argv) > 2 else 4)
