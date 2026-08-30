# interactor-hailo-ugen300

Apparatus for running models on the ASUS UGen300 (Hailo-10H) from Windows,
shelved 2026-08-29. The account of what was measured, retracted and left owed
is the logbook entry `logbook-qwen3vl-vit-translate-walls.md` in
`manuals-weftspun`; this repository holds what re-running it needs.

## Shelved state

Proven: torch -> ONNX -> DFC translate/optimize/compile -> HEF -> inference on
the device through the shim, including a self-attention block at 2.44% error
against fp32 once calibration matched the input distribution. Unresolved: the
full Qwen3-VL-4B vision tower does not translate; three explanations were
offered and retracted, and the owed next step is the bisection -- grow
`vit/rung_batch.py`'s block toward the tower one feature at a time (rotary,
depth, mergers) and report which addition first refuses.

Reopening condition, from the logbook: production demand for always-on judging
the desktop GPU cannot spare.

## Layout

- `shim/` -- `hailort_shim.cpp` + `.sigs`, the flat-C bridge over HailoRT's
  C++ InferModel (the Hailo-10H rejects the flat C API HailoRT exports), the
  ctypes binder, probes (`rung3_shim.py`, `mtmd_contract.py`, `hold.py`) and
  benches. Build against HailoRT 5.3.2's SDK; the DLL is not committed.
- `vit/` -- the Qwen3-VL vision tower export with the EditScore merger LoRA
  merged, the DFC translate scripts, and the minimal attention-block probes.
  The exported ONNX is 1.6 GB and reproducible from `export_vit.py`, so it is
  not committed.
- `mezo/` -- the QZO-on-silicon rung ladder: does a forced quantisation range
  reach the device while kernels stay bit-identical. Results `.txt` committed
  next to the scripts that printed them.
- `hef/` -- compiled artifacts small enough to keep, so restarting does not
  require the DFC before the device can be exercised. `rung3` also wants
  `efficientnet_lite0.hef`, which comes from the Hailo model zoo and is not
  redistributed here.

Environment: HailoRT 5.3.2 on Windows for the device half; DFC 5.3.0
(`hailo10h`) in WSL for the compile half.

## Binaries

The large and irreproducible-without-the-checkpoint artifacts -- the merged
tower ONNX exports, the QZO HAR/params archives, and a full mirror of the
session's working directories including logs and built DLL -- are on Hugging
Face: <https://huggingface.co/chibifire/hailo-ugen300-artifacts> (private).
It is deliberately not in the manifest; restarting takes this repository
first and fetches from there.
