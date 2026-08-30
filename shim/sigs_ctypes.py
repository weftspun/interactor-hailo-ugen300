"""Bind a .sigs file to a shared library through ctypes.

The workspace already declares C ABIs in .sigs files -- iceoryx2.sigs, openvr_api.sigs,
gstreamer.sigs -- and feeds them to Chromium's generate_stubs.py, which emits a dlsym-backed
dispatch table for C and C++ callers. This is the same declaration consumed from Python:
one C declaration per line, '#' starts a comment, and the ABI lives in one file rather than
scattered through argtypes assignments where drift is invisible.

The parser is deliberately small. It handles the subset the .sigs files in this workspace
actually use -- pointers, the fixed-width integer types, bool, void -- and raises on
anything else rather than guessing, because a silently mis-bound argument corrupts memory
instead of failing.
"""
import ctypes
import re

#: Only what the .sigs files here use. An unknown type raises rather than defaulting,
#: because guessing an argument width is how ctypes segfaults instead of erroring.
TYPES = {
    "void": None,
    "bool": ctypes.c_bool,
    "char": ctypes.c_char,
    "int": ctypes.c_int,
    "unsigned": ctypes.c_uint,
    "size_t": ctypes.c_size_t,
    "intptr_t": ctypes.c_ssize_t,
    "uintptr_t": ctypes.c_size_t,
    "int8_t": ctypes.c_int8,
    "uint8_t": ctypes.c_uint8,
    "int16_t": ctypes.c_int16,
    "uint16_t": ctypes.c_uint16,
    "int32_t": ctypes.c_int32,
    "uint32_t": ctypes.c_uint32,
    "int64_t": ctypes.c_int64,
    "uint64_t": ctypes.c_uint64,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    # HailoRT's status enum. Named so a caller can compare against 0 for success.
    "hailo_status": ctypes.c_int,
}

# The function name is the last identifier before "(", so a return type ending in a
# star -- void *hs_open(...) -- keeps its star with the type rather than the name.
DECL = re.compile(r"^\s*(.+?)([A-Za-z_]\w*)\s*\((.*)\)\s*;\s*$")


def _ctype(text):
    """One C type to a ctypes type. Pointer depth is counted; const is discarded."""
    text = text.replace("const", " ").strip()
    stars = text.count("*")
    base = text.replace("*", " ").split()
    base = [w for w in base if w not in ("struct", "enum", "unsigned") or w == "unsigned"]
    key = " ".join(base).strip()
    if key not in TYPES:
        # unsigned int, unsigned char and friends
        key = key.replace("unsigned ", "u").replace("uchar", "uint8_t").replace("uint", "uint32_t")
    if key not in TYPES:
        raise KeyError("unmapped C type %r -- add it to TYPES rather than guessing" % text)
    t = TYPES[key]
    if t is None and stars == 0:
        return None                      # void return
    if t is None:
        t = ctypes.c_void_p              # void*
        stars -= 1
    for _ in range(stars):
        t = ctypes.POINTER(t) if t is not ctypes.c_char else ctypes.c_char_p
    return t


def parse(path):
    """Every declaration in a .sigs file, as (name, restype, argtypes)."""
    out = []
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = DECL.match(line)
        if not m:
            raise SyntaxError("not a C declaration: %r" % raw.rstrip())
        ret, name, args = m.groups()
        argtypes = []
        if args.strip() and args.strip() != "void":
            for a in args.split(","):
                a = a.strip()
                # Drop the parameter name, keeping any stars, which belong to the type.
                stripped = re.sub(r"[A-Za-z_]\w*\s*$", "", a).strip()
                a = stripped if stripped.replace("*", "").strip() else a
                argtypes.append(_ctype(a))
        out.append((name, _ctype(ret), argtypes))
    return out


def bind(lib_path, sigs_path, missing_ok=False):
    """Load the library and attach every .sigs declaration with argtypes set."""
    lib = ctypes.CDLL(lib_path)
    bound, missing = {}, []
    for name, restype, argtypes in parse(sigs_path):
        try:
            fn = getattr(lib, name)
        except AttributeError:
            missing.append(name)
            continue
        fn.restype = restype
        fn.argtypes = argtypes
        bound[name] = fn
    if missing and not missing_ok:
        raise AttributeError("not exported by %s: %s" % (lib_path, ", ".join(missing)))
    return lib, bound, missing


if __name__ == "__main__":
    import sys
    lib_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Program Files\HailoRT\bin\libhailort.dll"
    sigs = sys.argv[2] if len(sys.argv) > 2 else "hailort.sigs"
    decls = parse(sigs)
    print("%s: %d declarations" % (sigs, len(decls)))
    lib, bound, missing = bind(lib_path, sigs, missing_ok=True)
    print("bound   %d" % len(bound))
    print("missing %d %s" % (len(missing), missing or ""))
    for name, restype, argtypes in decls:
        mark = "ok " if name in bound else "MISS"
        print("  %s %-42s -> %s (%d args)"
              % (mark, name, getattr(restype, "__name__", restype), len(argtypes)))
