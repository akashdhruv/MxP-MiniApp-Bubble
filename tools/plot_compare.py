#!/usr/bin/env python3
"""plot_compare.py -- compare the bubble deformation of one or more
FP-precision-emulated runs against the FP64 reference run (U8/RAPTOR).

    python3 tools/plot_compare.py --ref out_ref --cmp tf32=out_tf32 [--step N]
    python3 tools/plot_compare.py --ref out_ref --cmp tf32=out_tf32 --all-steps

Reads each run's last (or --step) dfun_<step>.pvti frame with plot_fields.py's
VTK reader, overlays the phi = 0 contours over the reference velocity field,
and reports max|phi_cmp - phi_ref| (Linf), RMS, and the area-drift delta
parsed from each run's run.log (written by tools/run_fp_sweep.sh).

With --all-steps the same comparison is made at every frame the reference and
all --cmp runs have in common: one `compare_<step>.png` each, the animated
`compare.gif`, and the error growth history `error_vs_time.png` + `.csv`.
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


def frame_map(outdir):
    """step -> dfun_<step>.pvti path, for every frame in outdir."""
    if not os.path.isdir(outdir):
        raise SystemExit("plot_compare: no such directory '%s'" % outdir)
    frames = {}
    for name in sorted(os.listdir(outdir)):
        m = _FRAME_RE.match(name)
        if m:
            frames[int(m.group(1))] = os.path.join(outdir, name)
    if not frames:
        raise SystemExit("plot_compare: no dfun_*.pvti files in '%s'" % outdir)
    return frames


def pick_frame(outdir, step=None):
    frames = frame_map(outdir)
    if step is None:
        last = max(frames)
        return last, frames[last]
    if step in frames:
        return step, frames[step]
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


def parse_cmp(specs):
    """[LABEL=DIR, ...] -> [(label, dir), ...]"""
    pairs = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit("plot_compare: --cmp expects LABEL=DIR, got '%s'" % spec)
        label, outdir = spec.split("=", 1)
        pairs.append((label, outdir))
    return pairs


def render_frame(ref_path, cmp_paths, png, step=None):
    """Draw the phi = 0 contour overlay for one frame.

    cmp_paths is [(label, pvti path)].  Returns (ref_time, stats) where stats
    is [(label, time, linf, rms)] in the same order.
    """
    ref_origin, ref_spacing, ref_time, ref_fields = read_pvti(ref_path)
    ref_dfun = ref_fields["dfun"]
    ny, nx = ref_dfun.shape
    x = ref_origin[0] + (np.arange(nx) + 0.5) * ref_spacing[0]
    y = ref_origin[1] + (np.arange(ny) + 0.5) * ref_spacing[1]
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    if "u" in ref_fields and "v" in ref_fields:
        speed = np.sqrt(ref_fields["u"] ** 2 + ref_fields["v"] ** 2)
        im = ax.pcolormesh(X, Y, speed, cmap="Blues", shading="auto")
        fig.colorbar(im, ax=ax, label="|u| (reference)")
    ax.contour(X, Y, ref_dfun, levels=[0.0], colors="crimson", linewidths=2.0)
    handles = [plt.Line2D([0], [0], color="crimson", lw=2.0, label="reference (fp64)")]

    stats = []
    for i, (label, path) in enumerate(cmp_paths):
        origin, spacing, time, fields = read_pvti(path)
        dfun = fields["dfun"]
        if dfun.shape != ref_dfun.shape:
            raise SystemExit("plot_compare: '%s' grid %s != reference grid %s" %
                             (path, dfun.shape, ref_dfun.shape))

        diff = dfun - ref_dfun
        linf = float(np.max(np.abs(diff)))
        rms = float(np.sqrt(np.mean(diff ** 2)))
        color = _COLORS[i % len(_COLORS)]
        ax.contour(X, Y, dfun, levels=[0.0], colors=color, linewidths=1.6, linestyles="dashed")
        handles.append(plt.Line2D([0], [0], color=color, lw=1.6, ls="dashed", label=label))
        stats.append((label, time, linf, rms))

    ax.set_aspect("equal")
    ax.set_xlim(ref_origin[0], x[-1] + 0.5 * ref_spacing[0])
    ax.set_ylim(ref_origin[1], y[-1] + 0.5 * ref_spacing[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = "bubble deformation: phi = 0 contour, t = %.4f" % ref_time
    if step is not None:
        title += "  (step %d)" % step
    ax.set_title(title)
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return ref_time, stats


def report_drift(ref_dir, cmp_pairs):
    ref_drift = area_drift_pct(ref_dir)
    for label, outdir in cmp_pairs:
        cmp_drift = area_drift_pct(outdir)
        if ref_drift is not None and cmp_drift is not None:
            print("plot_compare: %-8s area drift  ref=%.4f%%  %s=%.4f%%  delta=%.4f pp" %
                  (label, ref_drift, label, cmp_drift, cmp_drift - ref_drift))


def common_steps(ref_dir, cmp_pairs, stride):
    ref_frames = frame_map(ref_dir)
    steps = set(ref_frames)
    cmp_frames = {}
    for label, outdir in cmp_pairs:
        cmp_frames[label] = frame_map(outdir)
        steps &= set(cmp_frames[label])
    if not steps:
        raise SystemExit("plot_compare: no frame numbers are common to all runs "
                         "(mismatched plot_interval, or a run stopped early)")
    missing = len(ref_frames) - len(steps)
    if missing > 0:
        print("plot_compare: %d reference frame(s) have no counterpart in every "
              "--cmp run, skipping them" % missing)
    return sorted(steps)[::stride], ref_frames, cmp_frames


def write_error_history(outdir, series, steps_t):
    """series: label -> [(t, linf, rms)].  Writes the CSV and the PNG."""
    csv = os.path.join(outdir, "error_vs_time.csv")
    labels = sorted(series)
    with open(csv, "w") as f:
        f.write("step,time," + ",".join("linf_%s,rms_%s" % (l, l) for l in labels) + "\n")
        for i, (step, t) in enumerate(steps_t):
            cols = []
            for l in labels:
                cols.append("%.9e" % series[l][i][1])
                cols.append("%.9e" % series[l][i][2])
            f.write("%d,%.9e,%s\n" % (step, t, ",".join(cols)))
    print("plot_compare: wrote %s" % csv)

    fig, (ax_linf, ax_rms) = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True)
    for i, l in enumerate(labels):
        t = [row[0] for row in series[l]]
        color = _COLORS[i % len(_COLORS)]
        ax_linf.plot(t, [row[1] for row in series[l]], color=color, lw=1.6, label=l)
        ax_rms.plot(t, [row[2] for row in series[l]], color=color, lw=1.6, label=l)
    for ax, name in ((ax_linf, "Linf |dphi|"), (ax_rms, "RMS |dphi|")):
        ax.set_yscale("log")
        ax.set_ylabel(name)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    ax_rms.set_xlabel("t")
    ax_linf.set_title("deviation from the fp64 reference over time")
    fig.tight_layout()
    png = os.path.join(outdir, "error_vs_time.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print("plot_compare: wrote %s" % png)


def write_gif(outdir, pngs):
    if len(pngs) < 2:
        return
    gif = os.path.join(outdir, "compare.gif")
    try:
        from PIL import Image

        images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in pngs]
        images[0].save(gif, save_all=True, append_images=images[1:], duration=120, loop=0)
        print("plot_compare: wrote %s (%d frames)" % (gif, len(images)))
    except ImportError:
        sys.stderr.write("plot_compare.py: Pillow is not available, skipping the GIF "
                         "(PNG frames were written)\n")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference (fp64) output directory")
    ap.add_argument("--cmp", action="append", required=True, metavar="LABEL=DIR",
                    help="an emulated-precision output directory, repeatable")
    ap.add_argument("--step", type=int, default=None,
                    help="compare this step instead of each run's last frame")
    ap.add_argument("--all-steps", action="store_true",
                    help="compare at every frame common to all runs: one PNG each, "
                         "plus compare.gif and error_vs_time.png/.csv")
    ap.add_argument("--stride", type=int, default=1, metavar="N",
                    help="with --all-steps, use every Nth common frame (default 1)")
    ap.add_argument("--no-gif", action="store_true",
                    help="with --all-steps, skip the animated GIF")
    ap.add_argument("--out", default="out_compare",
                    help="directory to write the comparison PNG into (default out_compare)")
    args = ap.parse_args(argv[1:])

    if args.all_steps and args.step is not None:
        raise SystemExit("plot_compare: --step and --all-steps are mutually exclusive")
    if args.stride < 1:
        raise SystemExit("plot_compare: --stride must be >= 1")

    cmp_pairs = parse_cmp(args.cmp)
    os.makedirs(args.out, exist_ok=True)

    if not args.all_steps:
        _, ref_path = pick_frame(args.ref, args.step)
        cmp_paths = [(label, pick_frame(d, args.step)[1]) for label, d in cmp_pairs]
        png = os.path.join(args.out, "bubble_deformation_compare.png")
        ref_time, stats = render_frame(ref_path, cmp_paths, png)
        print("plot_compare: reference %s  t=%.6f" % (ref_path, ref_time))
        for label, time, linf, rms in stats:
            print("plot_compare: %-8s t=%.6f  Linf|dphi|=%.6e  RMS|dphi|=%.6e" %
                  (label, time, linf, rms))
        report_drift(args.ref, cmp_pairs)
        print("plot_compare: wrote %s" % png)
        return 0

    steps, ref_frames, cmp_frames = common_steps(args.ref, cmp_pairs, args.stride)
    print("plot_compare: %d common frame(s), steps %d..%d" %
          (len(steps), steps[0], steps[-1]))

    series = {label: [] for label, _ in cmp_pairs}
    steps_t = []
    pngs = []
    for step in steps:
        cmp_paths = [(label, cmp_frames[label][step]) for label, _ in cmp_pairs]
        png = os.path.join(args.out, "compare_%06d.png" % step)
        ref_time, stats = render_frame(ref_frames[step], cmp_paths, png, step=step)
        pngs.append(png)
        steps_t.append((step, ref_time))
        for label, time, linf, rms in stats:
            series[label].append((time, linf, rms))
        print("plot_compare: step %6d  t=%.6f  %s" %
              (step, ref_time,
               "  ".join("%s Linf=%.3e RMS=%.3e" % (l, li, r) for l, _, li, r in stats)))

    print("plot_compare: wrote %d frame(s) into %s/" % (len(pngs), args.out))
    if not args.no_gif:
        write_gif(args.out, pngs)
    write_error_history(args.out, series, steps_t)
    report_drift(args.ref, cmp_pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
