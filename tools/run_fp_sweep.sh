#!/usr/bin/env bash
# run_fp_sweep.sh -- reference (fp64) + tf32-emulated runs of the Deforming
# Bubble mini-app, for comparison with tools/plot_compare.py, plus an
# optional short RAPTOR profiling pass over src/advect.cpp's WENO kernel.
#
#   tools/run_fp_sweep.sh              # out_ref/ (fp64) + out_tf32/ (tf32)
#   tools/run_fp_sweep.sh --profile N  # also: out_profile/ + exponent histogram over N steps (default 5)
#
# NP (default 1) and PARAMS (default params/bubble.ini) are overridable env
# vars; any extra key=value arguments are forwarded to both runs.
set -euo pipefail
cd "$(dirname "$0")/.."

NP="${NP:-1}"
PARAMS="${PARAMS:-params/bubble.ini}"
PROFILE_STEPS=""

if [ "${1:-}" = "--profile" ]; then
    PROFILE_STEPS="${2:-5}"
    shift 2 || shift 1
fi
OVERRIDES=("$@")

[ -x ./bubble ] || make
[ -x ./bubble_raptor ] || tools/build_raptor.sh

run_one() {
    local label="$1" outdir="$2"
    shift 2
    echo "run_fp_sweep: $label -> $outdir/"
    mkdir -p "$outdir"
    "$@" 2>&1 | tee "$outdir/run.log"
}

run_one "reference (fp64)" out_ref \
    mpirun -np "$NP" ./bubble "$PARAMS" outdir=out_ref "${OVERRIDES[@]}"

run_one "tf32 emulation" out_tf32 \
    mpirun -np "$NP" -x BUBBLE_FP_MODE=tf32 ./bubble_raptor "$PARAMS" outdir=out_tf32 "${OVERRIDES[@]}"

if [ -n "$PROFILE_STEPS" ]; then
    run_one "profile ($PROFILE_STEPS steps)" out_profile \
        mpirun -np 1 -x BUBBLE_FP_MODE=profile -x BUBBLE_FP_LOG=out_profile/weno_flops_f64.bin \
        ./bubble_raptor "$PARAMS" outdir=out_profile max_steps="$PROFILE_STEPS"

    python3 dev/tools/RAPTOR/scripts/raptor_plot_float_histogram.py \
        out_profile/weno_flops_f64.bin --dtype float64 \
        --output out_profile/weno_exponent_hist.png
    echo "run_fp_sweep: wrote out_profile/weno_exponent_hist.png"
fi
