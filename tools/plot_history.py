#!/usr/bin/env python3
"""plot_history.py -- conservation + diagnostics plots for the Deforming
Bubble mini-app (U8).

    python3 tools/plot_history.py out/ [--prefix NAME]

Reads `out/history.csv` written by src/io.cpp, whose columns are fixed by the
Spec ("Output and post-processing"):

    step, time, dt, area, mass, phi_min, phi_max, max_divergence, sign_flips

and writes three PNGs into the same directory:

    history_conservation.png   area and mass versus time, plus their relative
                               drift against the t = 0 value (the quantity
                               tests/test_conservation bounds at 5%)
    history_divergence.png     max |div u| versus time against the 1e-12 bar of
                               invariant 6, with the per-step dt and the
                               cumulative redistance sign-flip count
    history_extrema.png        phi_min / phi_max versus time

Standard library + numpy + matplotlib only; no pandas, no `vtk` package.
"""

import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)


COLUMNS = [
    "step",
    "time",
    "dt",
    "area",
    "mass",
    "phi_min",
    "phi_max",
    "max_divergence",
    "sign_flips",
]

DIV_TOL = 1.0e-12  # invariant 6
AREA_BAR = 0.05  # Spec, Correctness bar: test_conservation


def read_history(path):
    """Return {column: np.ndarray} for history.csv, validating the header."""
    with open(path, "r") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("%s: file is empty" % path)
        header = [h.strip() for h in header]
        if header != COLUMNS:
            raise ValueError(
                "%s: unexpected header\n  got      %s\n  expected %s"
                % (path, ",".join(header), ",".join(COLUMNS))
            )
        rows = []
        for lineno, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(COLUMNS):
                raise ValueError(
                    "%s:%d: %d fields, expected %d"
                    % (path, lineno, len(row), len(COLUMNS))
                )
            rows.append([float(v) for v in row])
    if not rows:
        raise ValueError("%s: header only, no data rows" % path)
    data = np.array(rows, dtype=np.float64)
    return {name: data[:, k] for k, name in enumerate(COLUMNS)}


def plot_conservation(outdir, h, prefix):
    t = h["time"]
    area = h["area"]
    mass = h["mass"]
    a0 = area[0]
    m0 = mass[0]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)

    ax = axes[0]
    ax.plot(t, area, color="crimson", lw=1.6, label="area  $\\sum H(\\phi)\\,dx\\,dy$")
    ax.plot(t, mass, color="steelblue", lw=1.6, label="mass  $\\sum H(\\phi)\\,\\phi\\,dx\\,dy$")
    ax.axhline(a0, color="crimson", ls=":", lw=1.0)
    ax.axhline(m0, color="steelblue", ls=":", lw=1.0)
    ax.set_ylabel("area / mass")
    ax.set_title("bubble conservation   (t = 0: area %.9g, mass %.9g)" % (a0, m0))
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    if a0 != 0.0:
        drift_a = 100.0 * (area - a0) / a0
        ax.plot(t, drift_a, color="crimson", lw=1.6, label="area drift")
    if m0 != 0.0:
        drift_m = 100.0 * (mass - m0) / m0
        ax.plot(t, drift_m, color="steelblue", lw=1.6, label="mass drift")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.axhline(100.0 * AREA_BAR, color="0.2", ls="--", lw=1.0,
               label="Spec bar $\\pm$%.0f%%" % (100.0 * AREA_BAR))
    ax.axhline(-100.0 * AREA_BAR, color="0.2", ls="--", lw=1.0)
    ax.set_xlabel("t")
    ax.set_ylabel("relative drift [%]")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    png = os.path.join(outdir, prefix + "_conservation.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


def plot_divergence(outdir, h, prefix):
    t = h["time"]
    div = h["max_divergence"]
    dt = h["dt"]
    flips = h["sign_flips"]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)

    ax = axes[0]
    dmax = float(div.max()) if div.size else 0.0
    if dmax > 0.0:
        ax.semilogy(t, np.maximum(div, 1.0e-300), color="darkgreen", lw=1.6,
                    label="max $|\\nabla\\cdot u|$")
        ax.axhline(DIV_TOL, color="crimson", ls="--", lw=1.1,
                   label="invariant 6 tol $10^{-12}$")
    else:
        ax.plot(t, div, color="darkgreen", lw=1.6, label="max $|\\nabla\\cdot u|$ (identically 0)")
    ax.set_ylabel("max |div u|")
    ax.set_title("discrete divergence (max over the run: %.3e)" % dmax)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.plot(t, dt, color="darkorange", lw=1.6)
    ax.set_ylabel("dt")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, np.cumsum(flips), color="purple", lw=1.6)
    ax.set_xlabel("t")
    ax.set_ylabel("cumulative sign flips")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    png = os.path.join(outdir, prefix + "_divergence.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


def plot_extrema(outdir, h, prefix):
    t = h["time"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(t, h["phi_max"], color="crimson", lw=1.6, label="$\\phi_{max}$")
    ax.plot(t, h["phi_min"], color="steelblue", lw=1.6, label="$\\phi_{min}$")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("$\\phi$")
    ax.set_title("level-set extrema")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    png = os.path.join(outdir, prefix + "_extrema.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


def main(argv):
    args = argv[1:]
    outdir = None
    prefix = "history"
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--prefix":
            i += 1
            prefix = args[i]
        elif a.startswith("--prefix="):
            prefix = a.split("=", 1)[1]
        elif outdir is None:
            outdir = a
        else:
            sys.stderr.write("plot_history.py: unexpected argument '%s'\n" % a)
            return 2
        i += 1
    if outdir is None:
        outdir = "out"
    if not os.path.isdir(outdir):
        sys.stderr.write("plot_history.py: no such directory '%s'\n" % outdir)
        return 2

    path = os.path.join(outdir, "history.csv")
    if not os.path.isfile(path):
        sys.stderr.write("plot_history.py: no history.csv in '%s'\n" % outdir)
        return 1

    try:
        h = read_history(path)
    except ValueError as exc:
        sys.stderr.write("plot_history.py: %s\n" % exc)
        return 1

    n = h["step"].size
    a0, a1 = float(h["area"][0]), float(h["area"][-1])
    m0, m1 = float(h["mass"][0]), float(h["mass"][-1])
    drift = abs(a1 - a0) / a0 if a0 != 0.0 else 0.0
    print("plot_history: %s -> %d row(s), t in [%.6f, %.6f]"
          % (path, n, float(h["time"][0]), float(h["time"][-1])))
    print("plot_history: area %.9g -> %.9g   drift %.4f%%  (Spec bar %.0f%%)"
          % (a0, a1, 100.0 * drift, 100.0 * AREA_BAR))
    print("plot_history: mass %.9g -> %.9g" % (m0, m1))
    print("plot_history: max |div u| over the run = %.6e  (invariant 6 tol %.1e)"
          % (float(h["max_divergence"].max()), DIV_TOL))
    print("plot_history: sign flips, total = %d" % int(h["sign_flips"].sum()))

    pngs = [
        plot_conservation(outdir, h, prefix),
        plot_divergence(outdir, h, prefix),
        plot_extrema(outdir, h, prefix),
    ]
    for p in pngs:
        print("plot_history: wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
