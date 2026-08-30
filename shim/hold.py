import ctypes, os, sys, time
from sigs_ctypes import bind
os.add_dll_directory(r"C:\Program Files\HailoRT\bin")
H=os.path.dirname(os.path.abspath(__file__))
_, F, _ = bind(os.path.join(H,"hailort_shim.dll"), os.path.join(H,"hailort_shim.sigs"))
F["hs_last_error"].restype = ctypes.c_char_p
F["hs_open_ex"].restype = ctypes.c_void_p
h = F["hs_open_ex"](os.path.join(H,"big_512p.hef").encode(), 0)
print("%s: open %s" % (sys.argv[1], "OK" if h else "FAILED -- " + F["hs_last_error"]().decode()))
sys.stdout.flush()
if h:
    time.sleep(float(sys.argv[2]))
    F["hs_close"](ctypes.c_void_p(h))
