#!/usr/bin/env python3
"""plot_fields.py -- contour / field visualisation for the Deforming Bubble
mini-app (U8).

    python3 tools/plot_fields.py out/ [--stride N] [--no-gif]

Reads the ASCII VTK ImageData set written by src/io.cpp: one `dfun_<step>.pvti`
master per frame, each listing the per-rank `dfun_<step>_<rank>.vti` pieces with
their extents in global index space.  The pieces are reassembled into a single
global array -- this is the check that the `.pvti` set really is one field --
and for every frame the `phi = 0` contour is drawn over the velocity field.
Writes `fields_<step>.png` per frame plus the animated GIF `fields.gif`.

Standard library + numpy + matplotlib only; no `vtk` package (Spec, "Output and
post-processing").
"""

import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)


def _ints(text):
    return [int(v) for v in text.split()]


def _floats(text):
    return [float(v) for v in text.split()]


def read_vti(path):
    """Return (extent, {name: 2D array shaped (ny, nx)}) for one piece."""
    root = ET.parse(path).getroot()
    image = root.find("ImageData")
    piece = image.find("Piece")
    ext = _ints(piece.get("Extent"))
    nx = ext[1] - ext[0]
    ny = ext[3] - ext[2]
    arrays = {}
    celldata = piece.find("CellData")
    if celldata is not None:
        for da in celldata.findall("DataArray"):
            values = np.array(_floats(da.text), dtype=np.float64)
            if values.size != nx * ny:
                raise ValueError(
                    "%s: array '%s' has %d values, extent implies %d"
                    % (path, da.get("Name"), values.size, nx * ny)
                )
            arrays[da.get("Name")] = values.reshape((ny, nx))
    return ext, arrays


def read_pvti(path):
    """Reassemble one frame.  Returns (origin, spacing, time, {name: array})."""
    root = ET.parse(path).getroot()
    pimage = root.find("PImageData")
    whole = _ints(pimage.get("WholeExtent"))
    origin = _floats(pimage.get("Origin"))
    spacing = _floats(pimage.get("Spacing"))
    nx = whole[1] - whole[0]
    ny = whole[3] - whole[2]

    time = 0.0
    fielddata = pimage.find("FieldData")
    if fielddata is not None:
        for da in fielddata.findall("DataArray"):
            if da.get("Name") == "TIME":
                time = float(da.text)

    names = [pda.get("Name") for pda in pimage.find("PCellData").findall("PDataArray")]
    fields = {n: np.full((ny, nx), np.nan) for n in names}

    folder = os.path.dirname(path)
    for piece in pimage.findall("Piece"):
        src = os.path.join(folder, piece.get("Source"))
        ext, arrays = read_vti(src)
        i0, i1, j0, j1 = ext[0], ext[1], ext[2], ext[3]
        for n in names:
            fields[n][j0:j1, i0:i1] = arrays[n]

    for n in names:
        if np.isnan(fields[n]).any():
            missing = int(np.isnan(fields[n]).sum())
            raise ValueError(
                "%s: %d of %d cells of '%s' were not covered by any piece"
                % (path, missing, nx * ny, n)
            )
    return origin, spacing, time, fields


def frame_png(outdir, step, origin, spacing, time, fields, dpi=110):
    dfun = fields["dfun"]
    ny, nx = dfun.shape
    x = origin[0] + (np.arange(nx) + 0.5) * spacing[0]
    y = origin[1] + (np.arange(ny) + 0.5) * spacing[1]
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    if "u" in fields and "v" in fields:
        speed = np.sqrt(fields["u"] ** 2 + fields["v"] ** 2)
        im = ax.pcolormesh(X, Y, speed, cmap="Blues", shading="auto")
        fig.colorbar(im, ax=ax, label="|u|")
        s = max(1, nx // 24)
        ax.quiver(
            X[::s, ::s], Y[::s, ::s], fields["u"][::s, ::s], fields["v"][::s, ::s],
            color="0.35", scale=20.0, width=0.0025,
        )
    ax.contour(X, Y, dfun, levels=[0.0], colors="crimson", linewidths=1.8)
    ax.set_aspect("equal")
    ax.set_xlim(origin[0], x[-1] + 0.5 * spacing[0])
    ax.set_ylim(origin[1], y[-1] + 0.5 * spacing[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("step %d   t = %.4f   phi = 0 contour" % (step, time))
    fig.tight_layout()
    png = os.path.join(outdir, "fields_%06d.png" % step)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def main(argv):
    args = [a for a in argv[1:]]
    stride = 1
    make_gif = True
    outdir = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--stride":
            i += 1
            stride = int(args[i])
        elif a.startswith("--stride="):
            stride = int(a.split("=", 1)[1])
        elif a == "--no-gif":
            make_gif = False
        elif outdir is None:
            outdir = a
        else:
            sys.stderr.write("plot_fields.py: unexpected argument '%s'\n" % a)
            return 2
        i += 1
    if outdir is None:
        outdir = "out"
    if not os.path.isdir(outdir):
        sys.stderr.write("plot_fields.py: no such directory '%s'\n" % outdir)
        return 2

    pattern = re.compile(r"^dfun_(\d+)\.pvti$")
    frames = []
    for name in sorted(os.listdir(outdir)):
        m = pattern.match(name)
        if m:
            frames.append((int(m.group(1)), os.path.join(outdir, name)))
    frames.sort()
    frames = frames[::stride]
    if not frames:
        sys.stderr.write("plot_fields.py: no dfun_*.pvti files in '%s'\n" % outdir)
        return 1

    pngs = []
    for step, path in frames:
        origin, spacing, time, fields = read_pvti(path)
        dfun = fields["dfun"]
        print(
            "plot_fields: %s -> %d x %d cells reassembled, t = %.6f, "
            "phi in [%.6f, %.6f]" % (os.path.basename(path), dfun.shape[1], dfun.shape[0],
                                     time, float(dfun.min()), float(dfun.max()))
        )
        pngs.append(frame_png(outdir, step, origin, spacing, time, fields))

    print("plot_fields: wrote %d PNG frame(s) into %s/" % (len(pngs), outdir))

    if make_gif and len(pngs) > 0:
        gif = os.path.join(outdir, "fields.gif")
        try:
            from PIL import Image

            images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in pngs]
            images[0].save(
                gif, save_all=True, append_images=images[1:], duration=120, loop=0
            )
            print("plot_fields: wrote %s (%d frames)" % (gif, len(images)))
        except ImportError:
            sys.stderr.write(
                "plot_fields.py: Pillow is not available, skipping the GIF "
                "(PNG frames were written)\n"
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
