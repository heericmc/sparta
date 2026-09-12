#!/usr/bin/env python3
"""Render B5 helium fields and optional two-run comparisons.

Usage: python tools/plot_fields_b5.py RUN_DIR [OTHER_RUN] --outdir FIGURES
Uses the dump's bounds, aligns cells by ID, and averages the last --frac of
nonempty frames. --frac 0 selects only the final frame. The inlet cap is
drawn separately. Only zero-density cells are omitted; dilute gas is shown.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from field_io import FieldFormatError, read_complete_frames, recorded_dt, resolved_dt

NCOL = 11  # id xc yc xlo ylo xhi yhi nrho u v temp


def read_frames(path, timestep=None, until_step=None, until_ms=None, dt=None):
    """Return (steps, box, cells) with cells[k] = (Ncell, 11) sorted by id.

    Only complete, nonempty frames are eligible. Bounds include their endpoint.
    """
    result = read_complete_frames(path)
    known_dt = resolved_dt(os.path.dirname(path), dt) if dt is not None else recorded_dt(os.path.dirname(path))
    if until_ms is not None:
        known_dt = resolved_dt(os.path.dirname(path), dt)
        limit = until_ms * 1e-3
        selected = [f for f in result.frames
                    if f.timestep * known_dt <= limit + abs(limit) * 1e-12]
    elif timestep is not None:
        selected = [next((f for f in result.frames if f.timestep == timestep), None)]
    else:
        selected = [f for f in result.frames if until_step is None or f.timestep <= until_step]
    if not selected or selected[0] is None:
        raise FieldFormatError("no complete frame matches the requested selection")
    if any(frame.data.shape[1] != NCOL for frame in selected):
        raise FieldFormatError(f"{path}: B5 plotting needs {NCOL}-column grid frames")
    keep = [(frame.timestep, frame.data[np.argsort(frame.data[:, 0])])
            for frame in selected if np.any(frame.data[:, 7:])]
    if not keep:
        raise FieldFormatError(f"{path}: selection contains no populated complete frame")
    box = list(selected[-1].bounds[:2])
    return [s for s, _ in keep], box, [fr for _, fr in keep]


def average_tail(path, frac=0.5, timestep=None, until_step=None, until_ms=None, dt=None):
    """Mean field over the last `frac` of the non-empty frames."""
    steps, box, frames = read_frames(path, timestep, until_step, until_ms, dt)
    k = max(1, int(round(len(frames) * frac)))
    sel = frames[-k:]
    ids = sel[0][:, 0]
    for fr in sel[1:]:
        if not np.array_equal(fr[:, 0], ids):
            raise SystemExit("grid changed between frames in %s" % path)
    geom = sel[0][:, 1:7]
    fields = np.mean([fr[:, 7:] for fr in sel], axis=0)
    d = dict(steps=steps[-k:], box=box, ids=ids, xc=geom[:, 0], yc=geom[:, 1],
             xlo=geom[:, 2], ylo=geom[:, 3], xhi=geom[:, 4], yhi=geom[:, 5],
              nrho=fields[:, 0], u=fields[:, 1], v=fields[:, 2], t=fields[:, 3],
             nframe=len(sel), ntot=len(frames), dt=(resolved_dt(os.path.dirname(path), dt)
                                                    if dt is not None else recorded_dt(os.path.dirname(path))))
    # Per-frame body density, so "converged" vs "still filling" is a number
    # rather than an impression.  Frame grids are identical, so reuse the mask.
    m = ((d["xc"] >= 0.010) & (d["xc"] <= 0.055) & (d["yc"] <= 0.017))
    w = d["yc"][m] * (d["xhi"][m] - d["xlo"][m]) * (d["yhi"][m] - d["ylo"][m])
    d["trend"] = [(s, float(np.sum(fr[m, 7] * w) / np.sum(w)))
                  for s, fr in zip(steps, frames)]
    return d


def surf_by_type(path):
    """{type: [((x1,y1),(x2,y2)), ...]} from a SPARTA .surf with typed Lines."""
    pts, out, sect = {}, {}, None
    for raw in open(path):
        tok = raw.split("#")[0].split()
        if not tok:
            continue
        if tok[0] in ("Points", "Lines"):
            sect = tok[0]
            continue
        if len(tok) == 2 and tok[1] in ("points", "lines"):
            continue
        if sect == "Points":
            pts[int(tok[0])] = (float(tok[1]), float(tok[2]))
        elif sect == "Lines":
            v = [int(x) for x in tok]
            typ = v[1] if len(v) == 4 else 1
            out.setdefault(typ, []).append((pts[v[-2]], pts[v[-1]]))
    return out


def draw_geom(ax, surf):
    for typ, segs in sorted(surf.items()):
        col, lw = ("0.25", 1.5) if typ == 1 else ("tab:red", 2.6)
        for sgn in (1, -1):
            for (x1, y1), (x2, y2) in segs:
                ax.plot([x1, x2], [sgn * y1, sgn * y2], color=col, lw=lw,
                        zorder=6, solid_capstyle="butt")


def verts_and_mirror(d):
    good = d["nrho"] > 0
    v = []
    for sgn in (1, -1):
        for a, b, c, e in zip(d["xlo"][good], d["ylo"][good],
                              d["xhi"][good], d["yhi"][good]):
            v.append([(a, sgn * b), (c, sgn * b), (c, sgn * e), (a, sgn * e)])
    return good, v


def panel(ax, fig, verts, vals, label, cmap, surf, box, center=False, clim=None):
    pc = PolyCollection(verts, array=vals, cmap=cmap, edgecolors="none")
    if center:
        m = np.nanmax(np.abs(vals))
        pc.set_clim(-m, m)
    elif clim:
        pc.set_clim(*clim)
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, label=label, pad=0.01)
    draw_geom(ax, surf)
    ax.set_xlim(box[0])
    ax.set_ylim(-box[1][1], box[1][1])
    ax.set_ylabel("r (m)")
    ax.set_aspect("equal")


def fields_figure(d, surf, title, out):
    good, verts = verts_and_mirror(d)
    mir = lambda a: np.concatenate([a[good], a[good]])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True,
                             constrained_layout=True)
    panel(axes[0], fig, verts, mir(np.log10(np.where(d["nrho"] > 0, d["nrho"], np.nan))),
          r"$\log_{10}\,n$  (m$^{-3}$)", "viridis", surf, d["box"])
    panel(axes[1], fig, verts, mir(d["u"]),
          r"axial velocity $u$  (m s$^{-1}$)", "coolwarm", surf, d["box"],
          center=True)
    tvals = np.where(d["t"] > 0, d["t"], np.nan)
    panel(axes[2], fig, verts, mir(tvals),
          r"thermal temperature $T$  (K)", "inferno", surf, d["box"],
          clim=(0.0, min(np.nanmax(tvals), 8.0)))
    axes[2].set_xlabel("x (m)")
    axes[0].set_title(title)
    axes[0].legend(handles=[Line2D([], [], color="0.25", lw=1.5, label="cell wall"),
                            Line2D([], [], color="tab:red", lw=2.6,
                                   label="emitting cap (surf type 2)")],
                   loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote %s" % out)


def axis_profile(d):
    """On-axis (ylo == 0) cells, sorted by x."""
    m = (d["ylo"] == 0.0) & (d["nrho"] > 0)
    o = np.argsort(d["xc"][m])
    return (d["xc"][m][o], d["nrho"][m][o], d["u"][m][o], d["t"][m][o])


def radial_profile(d, x0):
    m = (d["xlo"] <= x0) & (d["xhi"] > x0) & (d["nrho"] > 0)
    o = np.argsort(d["yc"][m])
    return d["yc"][m][o], d["nrho"][m][o]


def region_mean(d, xlo, xhi, rmax, field="nrho"):
    """Volume-weighted mean over an (x, r) box; axisymmetric weight r*dx*dr."""
    m = ((d["xc"] >= xlo) & (d["xc"] <= xhi) & (d["yc"] <= rmax)
         & (d["nrho"] > 0))
    w = d["yc"][m] * (d["xhi"][m] - d["xlo"][m]) * (d["yhi"][m] - d["ylo"][m])
    return float(np.sum(d[field][m] * w) / np.sum(w)), int(m.sum())


def on_axis_at(d, x0, field):
    x, n, u, t = axis_profile(d)
    vals = dict(nrho=n, u=u, t=t)[field]
    return float(vals[np.argmin(np.abs(x - x0))]), float(x[np.argmin(np.abs(x - x0))])


def compare_figure(a, b, la, lb, surf, out, xrad=0.030):
    if not np.array_equal(a["ids"], b["ids"]):
        raise SystemExit("run grids differ; ratio map not defined")
    good = (a["nrho"] > 0) & (b["nrho"] > 0)
    verts = []
    for sgn in (1, -1):
        for x1, y1, x2, y2 in zip(a["xlo"][good], a["ylo"][good],
                                  a["xhi"][good], a["yhi"][good]):
            verts.append([(x1, sgn * y1), (x2, sgn * y1), (x2, sgn * y2),
                          (x1, sgn * y2)])
    ratio = a["nrho"][good] / b["nrho"][good]
    lr = np.log10(np.concatenate([ratio, ratio]))

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 1.0, 1.0])
    axm = fig.add_subplot(gs[0, :])
    pc = PolyCollection(verts, array=lr, cmap="RdBu_r", edgecolors="none")
    m = np.nanpercentile(np.abs(lr), 99.5)
    pc.set_clim(-m, m)
    axm.add_collection(pc)
    fig.colorbar(pc, ax=axm, label=r"$\log_{10}$ density ratio", pad=0.01)
    draw_geom(axm, surf)
    axm.set_xlim(a["box"][0])
    axm.set_ylim(-a["box"][1][1], a["box"][1][1])
    axm.set_aspect("equal")
    axm.set_xlabel("x (m)")
    axm.set_ylabel("r (m)")
    axm.set_title("density ratio %s / %s (time-averaged tail frames)" % (la, lb))

    xa, na, ua, ta = axis_profile(a)
    xb, nb, ub, tb = axis_profile(b)
    ra, nra = radial_profile(a, xrad)
    rb, nrb = radial_profile(b, xrad)
    sub = [(fig.add_subplot(gs[1, :2]), [(xa, na, la), (xb, nb, lb)],
            r"on-axis $n$ (m$^{-3}$)", True),
           (fig.add_subplot(gs[1, 2:]), [(xa, ua, la), (xb, ub, lb)],
            r"on-axis $u$ (m s$^{-1}$)", False),
           (fig.add_subplot(gs[2, :2]), [(xa, ta, la), (xb, tb, lb)],
            r"on-axis $T$ (K)", False),
           (fig.add_subplot(gs[2, 2:]), [(ra, nra, la), (rb, nrb, lb)],
            r"$n$ (m$^{-3}$) at $x=%g$ mm" % (xrad * 1e3), True)]
    for ax, series, ylab, logy in sub:
        for xs, ys, lab in series:
            ax.plot(xs, ys, lw=1.3, label=lab)
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in (sub[0][0], sub[1][0], sub[2][0]):
        ax.set_xlabel("x (m)")
        ax.axvspan(0.0635, 0.0650, color="0.85", zorder=0)
    sub[3][0].set_xlabel("r (m)")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote %s" % out)


def report(d, tag):
    body, nb = region_mean(d, 0.010, 0.055, 0.017)
    throat, nt = region_mean(d, 0.0635, 0.0650, 0.0025)
    plume, xp = on_axis_at(d, 0.075, "nrho")
    uap, xu = on_axis_at(d, 0.0650, "u")
    ax_x, _, ax_u, _ = axis_profile(d)
    i = int(np.argmax(ax_u))
    umax, xm = float(ax_u[i]), float(ax_x[i])
    print("[%s] frames %d/%d, steps %d..%d" %
          (tag, d["nframe"], d["ntot"], d["steps"][0], d["steps"][-1]))
    print("  body n (x 10-55 mm, r<17 mm, %d cells) = %.4e m^-3" % (nb, body))
    print("  throat n (channel 63.5-65.0 mm, r<2.5 mm, %d cells) = %.4e m^-3"
          % (nt, throat))
    print("  plume n on axis at x=%.4f m = %.4e m^-3" % (xp, plume))
    print("  u on axis at aperture exit x=%.4f m = %.2f m/s" % (xu, uap))
    print("  max on-axis u = %.2f m/s at x=%.4f m" % (umax, xm))
    print("  body n per frame: %s"
          % ", ".join("%d:%.3e" % (s, v) for s, v in d["trend"]))
    return dict(body=body, throat=throat, plume=plume, plume_x=xp, uap=uap,
                uap_x=xu, umax=umax, umax_x=xm, nframe=d["nframe"],
                ntot=d["ntot"], s0=d["steps"][0], s1=d["steps"][-1])


def frame_label(d):
    s0, s1 = d["steps"][0], d["steps"][-1]
    if d["dt"] is None:
        return f"frames ending at steps {s0}-{s1}"
    return "frames ending at steps %d-%d (%.3g-%.3g ms)" % (
        s0, s1, s0 * d["dt"] * 1e3, s1 * d["dt"] * 1e3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("other", nargs="?")
    p.add_argument("--outdir", default=".")
    p.add_argument("--frac", type=float, default=0.5)
    p.add_argument("--xrad", type=float, default=0.030)
    p.add_argument("--surf", default="cell_b5.surf",
                   help="surf filename (relative to each run dir) to draw as geometry")
    p.add_argument("--label", default="B5 mflow cell",
                   help="prefix used in the figure title and output filenames")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--timestep", type=int)
    selection.add_argument("--until-step", type=int)
    selection.add_argument("--until-ms", type=float)
    p.add_argument("--dt", type=float,
                   help="seconds per SPARTA step when manifest.json has no DT")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    runs = [a.ref] + ([a.other] if a.other else [])
    ds, tags = [], []
    for r in runs:
        try:
            d = average_tail(os.path.join(r, "field.grid"), a.frac, a.timestep,
                             a.until_step, a.until_ms, a.dt)
        except FieldFormatError as exc:
            raise SystemExit(str(exc)) from exc
        tag = os.path.basename(os.path.normpath(r))
        surf = surf_by_type(os.path.join(r, a.surf))
        ds.append(d)
        tags.append(tag)
        fields_figure(d, surf, "%s, %s -- mean of %s" %
                      (a.label, tag, frame_label(d)),
                      os.path.join(a.outdir, "%s-2d-fields-%s.png" % (a.label.split()[0].lower(), tag)))
        report(d, tag)
    if len(ds) == 2:
        compare_figure(ds[0], ds[1], tags[0], tags[1],
                       surf_by_type(os.path.join(runs[0], a.surf)),
                       os.path.join(a.outdir, "%s-2d-compare.png" % a.label.split()[0].lower()), a.xrad)


if __name__ == "__main__":
    main()
