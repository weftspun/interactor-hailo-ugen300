"""Rung 3: put a chosen input through the UGen300 and read what comes back.

Every number measured on this device so far came from hailortcli, which generates its own
random inputs and reports only throughput. Nothing has yet gone: input I picked -> silicon
-> output I can compare. That is the prerequisite for scoring a candidate on the device,
which every idle-time or evolutionary loop needs.

Bound through hailort.sigs, the same declarative ABI convention the workspace uses for
iceoryx2 and openvr. Structs are laid out here because ctypes needs their stride; the
layouts are copied from hailort.h and asserted against the library's own behaviour rather
than trusted -- if a size is wrong the calls fail loudly instead of reading past the end.
"""
import ctypes
import sys

import numpy as np
from sigs_ctypes import bind

DLL = r"C:\Program Files\HailoRT\bin\libhailort.dll"
HAILO_MAX_NAME_SIZE = 128
HAILO_MAX_STREAM_NAME_SIZE = HAILO_MAX_NAME_SIZE
HAILO_MAX_NETWORK_GROUPS = 8
HAILO_FORMAT_TYPE_AUTO = 0
HAILO_SUCCESS = 0

lib, F, _ = bind(DLL, "hailort.sigs")


def check(status, what):
    if status != HAILO_SUCCESS:
        raise RuntimeError("%s failed, hailo_status=%d" % (what, status))


# --- struct layouts, from hailort.h ------------------------------------------------
class HailoFormat(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("order", ctypes.c_int), ("flags", ctypes.c_int)]


class VStreamParams(ctypes.Structure):
    _fields_ = [("user_buffer_format", HailoFormat),
                ("timeout_ms", ctypes.c_uint32),
                ("queue_size", ctypes.c_uint32),
                ("vstream_stats_flags", ctypes.c_int),
                ("pipeline_elements_stats_flags", ctypes.c_int)]


class VStreamParamsByName(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * HAILO_MAX_STREAM_NAME_SIZE),
                ("params", VStreamParams)]


# hailo_vdevice_params_t is only ever filled by the library and passed straight back, so it
# is an opaque buffer sized generously. init writes at most sizeof(struct); over-allocating
# is safe, under-allocating is not.
VDEVICE_PARAMS_BYTES = 512
CONFIGURE_PARAMS_BYTES = 8192


def main(hef_path, n_frames=1):
    vdev_params = (ctypes.c_ubyte * VDEVICE_PARAMS_BYTES)()
    check(lib.hailo_init_vdevice_params(ctypes.byref(vdev_params)), "init_vdevice_params")

    vdevice = ctypes.c_void_p()
    check(F["hailo_create_vdevice"](ctypes.cast(vdev_params, ctypes.c_void_p),
                                    ctypes.byref(vdevice)), "create_vdevice")
    print("vdevice created")

    hef = ctypes.c_void_p()
    check(F["hailo_create_hef_file"](ctypes.byref(hef), hef_path.encode()), "create_hef_file")
    print("hef loaded: %s" % hef_path)

    cfg = (ctypes.c_ubyte * CONFIGURE_PARAMS_BYTES)()
    check(F["hailo_init_configure_params_by_vdevice"](
        hef, vdevice, ctypes.cast(cfg, ctypes.c_void_p)), "init_configure_params")

    groups = (ctypes.c_void_p * HAILO_MAX_NETWORK_GROUPS)()
    n_groups = ctypes.c_size_t(HAILO_MAX_NETWORK_GROUPS)
    check(F["hailo_configure_vdevice"](vdevice, hef, ctypes.cast(cfg, ctypes.c_void_p),
                                       groups, ctypes.byref(n_groups)), "configure_vdevice")
    ng = groups[0]
    print("configured, %d network group(s)" % n_groups.value)

    in_p = (VStreamParamsByName * 16)()
    n_in = ctypes.c_size_t(16)
    check(F["hailo_make_input_vstream_params"](
        ng, False, HAILO_FORMAT_TYPE_AUTO, ctypes.cast(in_p, ctypes.c_void_p),
        ctypes.byref(n_in)), "make_input_vstream_params")

    out_p = (VStreamParamsByName * 16)()
    n_out = ctypes.c_size_t(16)
    check(F["hailo_make_output_vstream_params"](
        ng, False, HAILO_FORMAT_TYPE_AUTO, ctypes.cast(out_p, ctypes.c_void_p),
        ctypes.byref(n_out)), "make_output_vstream_params")
    print("vstreams: %d in, %d out" % (n_in.value, n_out.value))
    for i in range(n_in.value):
        print("   in  %s" % in_p[i].name.decode())
    for i in range(n_out.value):
        print("   out %s" % out_p[i].name.decode())

    ins = (ctypes.c_void_p * n_in.value)()
    check(F["hailo_create_input_vstreams"](ng, ctypes.cast(in_p, ctypes.c_void_p),
                                           n_in.value, ins), "create_input_vstreams")
    outs = (ctypes.c_void_p * n_out.value)()
    check(F["hailo_create_output_vstreams"](ng, ctypes.cast(out_p, ctypes.c_void_p),
                                            n_out.value, outs), "create_output_vstreams")

    in_size = ctypes.c_size_t()
    check(F["hailo_get_input_vstream_frame_size"](ins[0], ctypes.byref(in_size)), "in frame size")
    out_size = ctypes.c_size_t()
    check(F["hailo_get_output_vstream_frame_size"](outs[0], ctypes.byref(out_size)), "out frame size")
    print("frame sizes: in %d bytes, out %d bytes" % (in_size.value, out_size.value))

    # Two DIFFERENT chosen inputs. If the device is really computing, they differ; if the
    # harness is echoing something, they will not.
    results = []
    for seed in (1171, 2026):
        x = np.random.default_rng(seed).integers(0, 256, in_size.value, dtype=np.uint8)
        y = np.zeros(out_size.value, dtype=np.uint8)
        check(F["hailo_vstream_write_raw_buffer"](
            ins[0], x.ctypes.data_as(ctypes.c_void_p), in_size.value), "write")
        check(F["hailo_vstream_read_raw_buffer"](
            outs[0], y.ctypes.data_as(ctypes.c_void_p), out_size.value), "read")
        results.append(y.copy())
        print("seed %d -> output mean %.3f  sha %s"
              % (seed, y.mean(), __import__("hashlib").sha256(y.tobytes()).hexdigest()[:16]))

    same = np.array_equal(results[0], results[1])
    print("")
    print("two different inputs -> %s"
          % ("IDENTICAL OUTPUT, the device is not computing on my input"
             if same else "different outputs, the device computed on what I sent"))

    # Determinism: the same input twice must give the same answer.
    x = np.random.default_rng(1171).integers(0, 256, in_size.value, dtype=np.uint8)
    y2 = np.zeros(out_size.value, dtype=np.uint8)
    check(F["hailo_vstream_write_raw_buffer"](
        ins[0], x.ctypes.data_as(ctypes.c_void_p), in_size.value), "write")
    check(F["hailo_vstream_read_raw_buffer"](
        outs[0], y2.ctypes.data_as(ctypes.c_void_p), out_size.value), "read")
    print("repeat of seed 1171 -> %s"
          % ("bit-identical, deterministic" if np.array_equal(results[0], y2)
             else "DIFFERS from the first run"))

    F["hailo_release_output_vstreams"](outs, n_out.value)
    F["hailo_release_input_vstreams"](ins, n_in.value)
    F["hailo_release_hef"](hef)
    F["hailo_release_vdevice"](vdevice)
    print("released")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "efficientnet_lite0.hef")
