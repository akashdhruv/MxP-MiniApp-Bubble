# RAPTOR FP-emulation workflow for src/advect.cpp

1. Build the reference binary: `make` (produces `./bubble`, FP64).
2. Build the RAPTOR-instrumented binary: `tools/build_raptor.sh` (produces `./bubble_raptor`).
3. Profile the WENO kernel: `tools/run_fp_sweep.sh --profile 5` → `out_profile/` + exponent histogram.
4. Run the sweep (reference + tf32): `tools/run_fp_sweep.sh` → `out_ref/` + `out_tf32/`.
5. Compare bubble deformation: `python3 tools/plot_compare.py --ref out_ref --cmp tf32=out_tf32` → `out_compare/bubble_deformation_compare.png`.

Notes:
- `BUBBLE_FP_MODE` (read by `src/advect.cpp` under `-DBUBBLE_RAPTOR`) selects `fp64` (default), `tf32`, or `profile`.
- tf32 emulation runs ~25x slower than fp64 (MPFR-backed); the full `params/bubble.ini` config (~20,478 steps) takes hours under tf32.
- `tools/run_fp_sweep.sh` forwards extra `key=value` args (e.g. `max_steps=50`) to both runs, for a quick smoke test before a full sweep.
