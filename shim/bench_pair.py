"""The same attention block on the desktop and on the accelerator, same weights, same input.

The desktop arm is the control: it says what the answer is in fp32, so the accelerator arm
is measured against a number rather than only against a stopwatch. Both arms get the input
the other one got -- a speed comparison between two graphs that disagree is not a comparison.

Latency for the UGen300 includes the USB round trip, because that is what it costs to use.
"""
import ctypes, json, os, statistics, subprocess, sys, time
import numpy as np
import torch, torch.nn as nn
from sigs_ctypes import bind

S, C, H = 64, 64, 4
D = C // H
HERE = os.path.dirname(os.path.abspath(__file__))
OSQ = r"C:\Users\ernes\scoop\shims\osqueryi"


class Batched(nn.Module):
    def __init__(s):
        super().__init__()
        s.n = nn.LayerNorm(C); s.qkv = nn.Linear(C, 3 * C); s.proj = nn.Linear(C, C)
        s.n2 = nn.LayerNorm(C); s.fc1 = nn.Linear(C, 4 * C); s.fc2 = nn.Linear(4 * C, C)

    def forward(s, x):
        q, k, v = s.qkv(s.n(x)).split(C, dim=-1)
        q = q.reshape(1, S, H, D).permute(0, 2, 1, 3)
        k = k.reshape(1, S, H, D).permute(0, 2, 3, 1)
        v = v.reshape(1, S, H, D).permute(0, 2, 1, 3)
        a = torch.softmax(torch.matmul(q, k) * D ** -0.5, dim=-1)
        o = torch.matmul(a, v).permute(0, 2, 1, 3).reshape(1, S, C)
        x = x + s.proj(o)
        return x + s.fc2(torch.nn.functional.gelu(s.fc1(s.n2(x))))


def osq(sql):
    try:
        r = subprocess.run([OSQ, "--json", sql], capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout or "[]")
    except Exception as e:
        return [{"error": str(e)}]


def timed(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts), min(ts), max(ts)


def main(hef, iters=50):
    print("== the box, per osquery ==")
    for r in osq("SELECT cpu_brand, cpu_logical_cores, physical_memory FROM system_info;"):
        print("   %s | %s cores | %.0f GiB"
              % (r["cpu_brand"].strip(), r["cpu_logical_cores"],
                 int(r["physical_memory"]) / 2**30))
    for r in osq("SELECT model, manufacturer, driver_version FROM video_info;"):
        print("   gpu: %s (%s) driver %s"
              % (r.get("model", "?").strip(), r.get("manufacturer", "?").strip(),
                 r.get("driver_version", "?")))

    torch.manual_seed(1171)
    net = Batched().eval()
    rng = np.random.default_rng(1171)
    x_np = rng.standard_normal((1, S, C)).astype(np.float32)
    x = torch.from_numpy(x_np)

    results, outs = {}, {}
    print("\n== desktop control ==")
    with torch.no_grad():
        outs["cpu"] = net(x).numpy().reshape(-1)
        results["cpu"] = timed(lambda: net(x), iters)
    print("   cpu    median %7.3f ms   min %7.3f   max %7.3f" % results["cpu"])

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        g = net.cuda(); xg = x.cuda()
        with torch.no_grad():
            outs["cuda"] = g(xg).cpu().numpy().reshape(-1)

            def run():
                g(xg); torch.cuda.synchronize()
            results["cuda"] = timed(run, iters)
        print("   cuda   median %7.3f ms   min %7.3f   max %7.3f   (%s)"
              % (results["cuda"] + (name,)))
        net = net.cpu()
    else:
        print("   cuda   unavailable in this interpreter")

    print("\n== accelerator ==")
    os.add_dll_directory(r"C:\Program Files\HailoRT\bin")
    _, F, _ = bind(os.path.join(HERE, "hailort_shim.dll"),
                   os.path.join(HERE, "hailort_shim.sigs"))
    F["hs_last_error"].restype = ctypes.c_char_p
    F["hs_open_ex"].restype = ctypes.c_void_p
    h = F["hs_open_ex"](hef.encode(), 1)
    if not h:
        print("   hs_open_ex failed: %s" % F["hs_last_error"]().decode())
        return 1
    h = ctypes.c_void_p(h)
    n_in, n_out = F["hs_input_size"](h), F["hs_output_size"](h)
    print("   float I/O frame sizes: in %d bytes (%d floats), out %d bytes"
          % (n_in, n_in // 4, n_out))

    xin = np.ascontiguousarray(x_np.reshape(-1))
    y = np.zeros(n_out // 4, dtype=np.float32)

    def infer():
        st = F["hs_infer"](h, xin.ctypes.data_as(ctypes.c_void_p), n_in,
                           y.ctypes.data_as(ctypes.c_void_p), n_out, 10000)
        if st != 0:
            raise RuntimeError("hs_infer status=%d: %s" % (st, F["hs_last_error"]().decode()))

    infer()
    outs["hailo"] = y.copy()
    results["hailo"] = timed(infer, iters)
    print("   ugen300 median %7.3f ms   min %7.3f   max %7.3f  (includes USB round trip)"
          % results["hailo"])
    F["hs_close"](h)

    print("\n== do the two arms agree ==")
    ref = outs["cpu"]
    for tag in ("cuda", "hailo"):
        if tag not in outs:
            continue
        a = outs[tag]
        if a.shape != ref.shape:
            print("   %-7s shape %s vs control %s" % (tag, a.shape, ref.shape)); continue
        err = np.abs(a - ref)
        denom = np.abs(ref).mean()
        print("   %-7s max|err| %.5f   mean|err| %.5f   relative %.2f%%   corr %.5f"
              % (tag, err.max(), err.mean(), 100 * err.mean() / denom,
                 float(np.corrcoef(a, ref)[0, 1])))

    print("\n== speed, control = cpu ==")
    for tag in ("cpu", "cuda", "hailo"):
        if tag in results:
            print("   %-7s %7.3f ms   %5.2fx the control"
                  % (tag, results[tag][0], results["cpu"][0] / results[tag][0]))
    json.dump({k: v[0] for k, v in results.items()}, open("bench_pair.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "rb_batched.hef"))
