"""Does a HEF meet what llama.cpp's Hailo vision encoder requires of it?

mtmd throws on a HEF that fails any of these, after the device is already open. Checking
first turns that into an answer, and every rule here is read off hailo_encoder.cpp rather
than assumed.

The interesting failure is not a HEF that is rejected. It is a Qwen3-VL encoder that passes
every structural rule while carrying stock merger weights instead of EditScore's twelve --
same four outputs, same shapes, same config, wrong numbers. No check below sees that, which
is why check 7 reports the identity of the weights separately and cannot be satisfied by
structure alone.
"""
import ctypes, json, os, sys
from sigs_ctypes import bind

CONFIG = "hailo-config.json"
LAYERS = "encoder_output_layers_names_suffixes"
WANT = ["image_embeddings", "deepstack_layer_1", "deepstack_layer_2", "deepstack_layer_3"]

HERE = os.path.dirname(os.path.abspath(__file__))
os.add_dll_directory(r"C:\Program Files\HailoRT\bin")
_, F, _ = bind(os.path.join(HERE, "hailort_shim.dll"),
               os.path.join(HERE, "hailort_shim.sigs"))
F["hs_last_error"].restype = ctypes.c_char_p
F["hs_open"].restype = ctypes.c_void_p


def check(hef):
    results = []

    def rule(ok, name, detail):
        results.append((bool(ok), name, detail))

    h = F["hs_open"](hef.encode())
    if not h:
        rule(False, "opens", F["hs_last_error"]().decode(errors="replace"))
        return results
    h = ctypes.c_void_p(h)
    rule(True, "opens", "configured on the device")

    n = F["hs_num_outputs"](h)
    rule(n == 4, "exactly 4 outputs", "got %d, mtmd requires 4" % n)

    names, shapes = [], []
    for i in range(max(n, 0)):
        buf = ctypes.create_string_buffer(512)
        F["hs_output_name"](h, i, buf, 512)
        names.append(buf.value.decode(errors="replace"))
        hh, ww, ff = (ctypes.c_uint() for _ in range(3))
        F["hs_output_shape"](h, i, ctypes.byref(hh), ctypes.byref(ww), ctypes.byref(ff))
        shapes.append((hh.value, ww.value, ff.value))
    if shapes:
        rule(len(set(shapes)) == 1, "outputs share a shape",
             "; ".join("%s %s" % (a, b) for a, b in zip(names, shapes)))

    size = F["hs_resource"](h, CONFIG.encode(), None, 0)
    if size < 0:
        rule(False, "embeds " + CONFIG, "absent from the HEF")
        F["hs_close"](h)
        return results
    buf = ctypes.create_string_buffer(size + 1)
    F["hs_resource"](h, CONFIG.encode(), buf, size)
    rule(True, "embeds " + CONFIG, "%d bytes" % size)

    try:
        cfg = json.loads(buf.raw[:size].decode())
    except Exception as e:
        rule(False, "config parses", str(e))
        F["hs_close"](h)
        return results
    rule(True, "config parses", "keys: " + ", ".join(sorted(cfg)))

    for k in ("patch_size", "spatial_merge_size"):
        rule(k in cfg, "config has " + k, str(cfg.get(k, "missing")))

    layers = cfg.get(LAYERS, {})
    rule(all(k in layers for k in WANT), "names all 4 slots",
         "missing " + ", ".join(k for k in WANT if k not in layers) if
         not all(k in layers for k in WANT) else "image_embeddings + 3 deepstack")

    sfx = [layers[k] for k in WANT if k in layers]
    rule(len(set(sfx)) == len(sfx), "slot suffixes are distinct",
         "a repeat would collapse two layers onto one slot")
    have = {nm.split("/", 1)[-1] for nm in names}
    rule(set(sfx) <= have, "every suffix matches a stream",
         "unmatched: " + ", ".join(sorted(set(sfx) - have)) if set(sfx) - have else "all 4 resolve")

    F["hs_close"](h)
    return results


def main(hefs):
    worst = 0
    for hef in hefs:
        print("\n%s" % os.path.basename(hef))
        print("-" * 66)
        rs = check(hef)
        for ok, name, detail in rs:
            print("  %-4s %-28s %s" % ("PASS" if ok else "FAIL", name, detail))
        bad = sum(1 for ok, _, _ in rs if not ok)
        print("  => %s" % ("meets the mtmd contract" if not bad
                           else "%d of %d rules fail" % (bad, len(rs))))
        worst = max(worst, bad)
    return worst


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) is not None else 1)
