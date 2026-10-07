#!/usr/bin/env python3
"""plot_compare.py -- compare the bubble deformation of one or more
FP-precision-emulated runs against the FP64 reference run (U8/RAPTOR).

    python3 tools/plot_compare.py --ref out_ref --cmp tf32=out_tf32 [--step N]

Reads each run's last (or --step) dfun_<step>.pvti frame with plot_fields.py's
VTK reader, overlays the phi = 0 contours over the reference velocity field,
and reports max|phi_cmp - phi_ref| (Linf), RMS, and the area-drift delta
parsed from each run's run.log (written by tools/run_fp_sweep.sh).
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_fields import read_pvti  # noqa: E402

_FRAME_RE = re.compile(r"^dfun_(\d+)\.pvti$")
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]


def pick_frame(outdir, step=None):
    frames = []
    for name in sorted(os.listdir(outdir)):
        m = _FRAME_RE.match(name)
        if m:
            frames.append((int(m.group(1)), os.path.join(outdir, name)))
    if not frames:
        raise SystemExit("plot_compare: no dfun_*.pvti files in '%s'" % outdir)
    frames.sort()
    if step is None:
        return frames[-1]
    for s, path in frames:
        if s == step:
            return s, path
    raise SystemExit("plot_compare: no step %d in '%s'" % (step, outdir))


def area_drift_pct(outdir):
    log = os.path.join(outdir, "run.log")
    if not os.path.isfile(log):
        return None
    with open(log) as f:
        for line in f:
            if line.startswith("bubble: area"):
                m = re.search(r"drift\s+([-\d.eE+]+)%", line)
                if m:
                    return float(m.group(1))
    return None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="reference (fp64) output directory")
    ap.add_argument("--cmp", action="append", required=True, metavar="LABEL=DIR",
                     help="an emulated-precision output directory, repeatable")
    ap.add_argument("--step", type=int, default=None,
                     help="compare this step instead of each run's last frame")
    ap.add_argument("--out", default="out_compare",
                     help="directory to write the comparison PNG into (default out_compare)")
    args = ap.parse_args(argv[1:])

    _, ref_path = pick_frame(args.ref, args.step)
    ref_origin, ref_spacing, ref_time, ref_fields = read_pvti(ref_path)
    ref_dfun = ref_fields["dfun"]
    ny, nx = ref_dfun.shape
    x = ref_origin[0] + (np.arange(nx) + 0.5) * ref_spacing[0]
    y = ref_origin[1] + (np.arange(ny) + 0.5) * ref_spacing[1]
    X, Y = np.meshgrid(x, y)

    os.makedirs(args.out, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    if "u" in ref_fields and "v" in ref_fields:
        speed = np.sqrt(ref_fields["u"] ** 2 + ref_fields["v"] ** 2)
        im = ax.pcolormesh(X, Y, speed, cmap="Blues", shading="auto")
        fig.colorbar(im, ax=ax, label="|u| (reference)")
    ax.contour(X, Y, ref_dfun, levels=[0.0], colors="crimson", linewidths=2.0)
    handles = [plt.Line2D([0], [0], color="crimson", lw=2.0, label="reference (fp64)")]

    print("plot_compare: reference %s  t=%.6f" % (ref_path, ref_time))
    ref_drift = area_drift_pct(args.ref)

    for i, spec in enumerate(args.cmp):
        if "=" not in spec:
            raise SystemExit("plot_compare: --cmp expects LABEL=DIR, got '%s'" % spec)
        label, outdir = spec.split("=", 1)
        _, path = pick_frame(outdir, args.step)
        origin, spacing, time, fields = read_pvti(path)
        dfun = fields["dfun"]
        if dfun.shape != ref_dfun.shape:
            raise SystemExit("plot_compare: '%s' grid %s != reference grid %s" %
                              (outdir, dfun.shape, ref_dfun.shape))

        diff = dfun - ref_dfun
        linf = float(np.max(np.abs(diff)))
        rms = float(np.sqrt(np.mean(diff ** 2)))
        color = _COLORS[i % len(_COLORS)]
        ax.contour(X, Y, dfun, levels=[0.0], colors=color, linewidths=1.6, linestyles="dashed")
        handles.append(plt.Line2D([0], [0], color=color, lw=1.6, ls="dashed", label=label))

        print("plot_compare: %-8s t=%.6f  Linf|dphi|=%.6e  RMS|dphi|=%.6e" %
              (label, time, linf, rms))
        cmp_drift = area_drift_pct(outdir)
        if ref_drift is not None and cmp_drift is not None:
            print("plot_compare: %-8s area drift  ref=%.4f%%  %s=%.4f%%  delta=%.4f pp" %
                  (label, ref_drift, label, cmp_drift, cmp_drift - ref_drift))

    ax.set_aspect("equal")
    ax.set_xlim(ref_origin[0], x[-1] + 0.5 * ref_spacing[0])
    ax.set_ylim(ref_origin[1], y[-1] + 0.5 * ref_spacing[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("bubble deformation: phi = 0 contour, t = %.4f" % ref_time)
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.out, "bubble_deformation_compare.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print("plot_compare: wrote %s" % png)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
