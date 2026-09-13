#!/usr/bin/env python3
"""Generate the Takahashi single-stage cell and four transparent flux stations.

Dimensions from Takahashi, Shlivko, Woolls & Hutzler, "Simulation of cryogenic
buffer gas beams", Phys. Rev. Research 3, 023018 (2021), Sec. III.A:
12.7 mm body ID, 53 mm body length, 4 mm inlet-tube ID, 20 mm inlet-tube
length, 5 mm aperture diameter, 0.5 mm plate thickness. The inlet cap is a
reflecting 4 K surface with prescribed mass injection, same as the B5 case.

Unlike the B5 case (which truncates its inlet to a short "simulated feed
stub"), this case models the full 20 mm inlet tube length exactly as given
in the paper, since the paper does not describe a truncation.

The .surf polygon itself only closes at RTOP_GEOM (~7 mm, just enough to
cleanly cap the actual hardware): the plate/cell wall does NOT get any
bigger. What gets bigger is the *simulated domain* around it -- out to
RTOP_DOMAIN and XHI far beyond the hardware -- so the wide-angle jet
backflow/backscatter off the plate's downstream face has real vacuum space
to expand into, rather than being clipped by an artificially close boundary.

Mesh: a coarse outer base plus 5 nested refinement levels (2 new bridging
levels added on top of the original 3, to step down from the new coarse
outer base to the same validated fine mesh used near the cell/aperture).
Each level spans the full radial extent so there are no internal radial
refinement boundaries to manage -- only axial ones. Each level-N band sits
inside its level-(N-1) parent by at least four level-(N-1) cells on every
open (axial) side, per the same nesting rule used in gen_b5.py, checked by
tools/check_grid_2d.py.

Run from the repository root: python cases/takahashi-single/gen_takahashi.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- paper geometry (Sec. III.A) -- UNCHANGED from the original case ----
RCELL = 0.00635              # 12.7 mm ID / 2
CELL_LENGTH = 0.053          # 53 mm body length (tube/cell junction -> plate face)
RB = 0.002                   # 4 mm inlet tube ID / 2
TUBE_LENGTH = 0.020           # 20 mm inlet tube length (full length, not truncated)
RAP = 0.0025                  # 5 mm aperture diameter / 2 (same as B5)
PLATE_THICK = 0.0005          # 0.5 mm plate thickness

XLO = -TUBE_LENGTH            # -0.020
XPLATE = CELL_LENGTH          # 0.053, plate cell-side face
XEXIT = XPLATE + PLATE_THICK  # 0.0535, channel exit

RTOP_GEOM = 55 * 1.3e-4        # 0.00715 m -- the .surf polygon's own closing radius (unchanged)

# ---- mesh: coarse outer base -> 2 bridging levels -> the original 3 fine levels ----
# Fine levels (unchanged from the original case, validated against check_grid_2d.py):
DX4, DR4 = 2.0e-4, 1.3e-4            # (old "L1") 200 x 130 um
DX5, DR5 = DX4 / 2.0, DR4 / 2.0       # (old "L2") 100 x 65 um
DX6, DR6 = DX4 / 4.0, DR4 / 4.0       # (old "L3") 50 x 32.5 um
# New bridging levels, each 2x the next-finer:
DX3, DR3 = DX4 * 2.0, DR4 * 2.0       # 400 x 260 um
DX2, DR2 = DX4 * 4.0, DR4 * 4.0       # 800 x 520 um
# New coarse outer base (SPARTA level 1, implicit -- the whole domain):
DX1, DR1 = DX4 * 8.0, DR4 * 8.0       # 1600 x 1040 um

NY = 63                        # RTOP_DOMAIN = 63 * DR1 ~= 6.55 cm (requested ~6.5 cm)
RTOP_DOMAIN = NY * DR1
NX = 109                       # XHI - XLO ~= 17.4 cm -> ~10.1 cm of plume past XEXIT (requested ~10 cm)
XHI = XLO + NX * DX1

# L2 band (level 2, 800x520um): wide outer bridge, well inset from the domain edges
L2_XLO, L2_XHI, L2_YHI = -0.013, 0.147, 0.060
# L3 band (level 3, 400x260um): narrower bridge
L3_XLO, L3_XHI, L3_YHI = -0.009, 0.10, 0.03
# L4 band (level 4, 200x130um = old "L1"): near cell + a stretch of near-aperture plume.
# Its y-extent must be taller than L5/L6's (RTOP_GEOM) by >=4 L4-cells, or the
# mesh jumps straight from L3 to L5/L6 at y=RTOP_GEOM, skipping L4 (4:1, gate fail).
L4_XLO, L4_XHI, L4_YHI = -0.0045, 0.075, RTOP_GEOM + 0.001
# L5 band (level 5, 100x65um = old "L2"): tube/cell junction through the whole cell body (unchanged)
L5_XLO, L5_XHI, L5_YHI = -0.003, 0.057, RTOP_GEOM
# L6 band (level 6, 50x32.5um = old "L3"): the plate/throat region (unchanged)
L6_XLO, L6_XHI, L6_YHI = 0.051, 0.0555, RTOP_GEOM

# ---- station / cap coordinates (unchanged -- depend only on XLO and the fine cell sizes) ----
XC = XLO + 1.5 * DX4 / 16        # cap: just inside xlo, non-face at L4
XTUBE = -0.0013 - 1.5 * DX5 / 16  # tube station, within the L5 band (nudged off-face)
XAP = 0.05325 - 1.5 * DX6 / 16    # aperture station, within the L6 band (mid-plate, nudged off-face)
R_TUBE_ST = RB + 0.0006          # through the tube wall into "solid"
R_AP_ST = RAP + 0.0013           # through the channel wall into the plate
XOUT, ROUT = XLO - 0.0003, -0.0003


def nonface(name, coord, origin, base, finest_level):
    for lev in range(1, finest_level + 1):
        h = base / 2 ** (lev - 1)
        idx = (coord - origin) / h
        assert abs(idx - round(idx)) > 1e-6, "%s on a cell face at L%d (index %.4f)" % (name, lev, idx)
    print("%-8s %.7g  non-face down to L%d (%.3g um)" % (name, coord, finest_level, base / 2 ** (finest_level - 1) * 1e6))


def check_nest(name, child_lo, child_hi, parent_lo, parent_hi, parent_cell):
    margin = 4 * parent_cell
    assert child_lo - parent_lo >= margin - 1e-12, "%s left margin too small (%.4g < %.4g)" % (name, child_lo - parent_lo, margin)
    assert parent_hi - child_hi >= margin - 1e-12, "%s right margin too small (%.4g < %.4g)" % (name, parent_hi - child_hi, margin)
    print("%-8s nested OK (x margins %.3g mm / %.3g mm, need >= %.3g mm)"
          % (name, (child_lo - parent_lo) * 1e3, (parent_hi - child_hi) * 1e3, margin * 1e3))


def check_nest_y(name, child_yhi, parent_yhi, parent_cell):
    if abs(child_yhi - parent_yhi) < 1e-12:
        print("%-8s y-extent matches parent (no radial step)" % name)
        return
    margin = 4 * parent_cell
    gap = parent_yhi - child_yhi
    assert gap >= margin - 1e-12, "%s y margin too small (%.4g < %.4g)" % (name, gap, margin)
    print("%-8s nested OK (y margin %.3g mm, need >= %.3g mm)" % (name, gap * 1e3, margin * 1e3))


def station(x, rtop, tag, fwd):
    p = [(x, 0.0), (x, rtop)] if fwd else [(x, rtop), (x, 0.0)]
    hdr = "# GENERATED by gen_takahashi.py. %s station at x = %.7g, %s movers (p1->p2 %s)." % (
        tag, x, "+x" if fwd else "-x", "+r" if fwd else "-r")
    return "\n".join([hdr, "", "2 points", "1 lines", "", "Points", "",
                      "1 %.9g %.9g" % p[0], "2 %.9g %.9g" % p[1], "", "Lines", "", "1 1 2", ""])


def loop():
    pts = [(XEXIT, RTOP_GEOM), (XEXIT, RAP), (XPLATE, RAP), (XPLATE, RCELL), (0.0, RCELL),
           (0.0, RB), (XC, RB), (XC, ROUT), (XOUT, ROUT), (XOUT, RTOP_GEOM)]
    types = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
    n = len(pts)
    out = ["# GENERATED by gen_takahashi.py -- Takahashi et al. (2021) single-stage cell, 2D axisymmetric.",
           "# 12.7 x 53 mm body, 5 mm aperture through 0.5 mm plate (%.6g-%.6g)," % (XPLATE, XEXIT),
           "# 4 mm ID x 20 mm inlet tube, CAP (type 2) at x = %.7g, normal +x." % XC,
           "# Polygon closes at RTOP_GEOM=%.6g m (the real hardware); the simulated" % RTOP_GEOM,
           "# domain (create_box) extends far beyond this -- see in.he_takahashi_mflow.",
           "# type 1 = every wall; read with `type group cellwalls clip`, then `group cap surf type 2`.",
           "", "%d points" % n, "%d lines" % n, "", "Points", ""]
    out += ["%d %.9g %.9g" % (i + 1, x, y) for i, (x, y) in enumerate(pts)]
    out += ["", "Lines", ""]
    out += ["%d %d %d %d" % (i + 1, types[i], i + 1, (i + 1) % n + 1) for i in range(n)]
    out.append("")
    return "\n".join(out)


def main():
    nonface("cap x", XC, XLO, DX4, 1)
    nonface("bore r", RB, 0.0, DR4, 2)
    nonface("tube st", XTUBE, XLO, DX4, 2)
    nonface("ap st", XAP, XLO, DX4, 3)
    nonface("r_cell", RCELL, 0.0, DR4, 3)
    nonface("r_ap", RAP, 0.0, DR4, 3)
    check_nest("L2", L2_XLO, L2_XHI, XLO, XHI, DX1)
    check_nest("L3", L3_XLO, L3_XHI, L2_XLO, L2_XHI, DX2)
    check_nest("L4", L4_XLO, L4_XHI, L3_XLO, L3_XHI, DX3)
    check_nest("L5", L5_XLO, L5_XHI, L4_XLO, L4_XHI, DX4)
    check_nest("L6", L6_XLO, L6_XHI, L5_XLO, L5_XHI, DX5)
    check_nest_y("L2", L2_YHI, RTOP_DOMAIN, DR1)
    check_nest_y("L3", L3_YHI, L2_YHI, DR2)
    check_nest_y("L4", L4_YHI, L3_YHI, DR3)
    check_nest_y("L5", L5_YHI, L4_YHI, DR4)
    check_nest_y("L6", L6_YHI, L5_YHI, DR5)
    for name, text in {
        "cell_takahashi.surf": loop(),
        "tube_fwd.surf": station(XTUBE, R_TUBE_ST, "tube", True),
        "tube_rev.surf": station(XTUBE, R_TUBE_ST, "tube", False),
        "ap_fwd.surf": station(XAP, R_AP_ST, "aperture", True),
        "ap_rev.surf": station(XAP, R_AP_ST, "aperture", False),
    }.items():
        with open(os.path.join(HERE, name), "w", newline="\n") as f:
            f.write(text)
        print("wrote", name)
    print("domain: x [%.6g, %.6g] (%d cells @ %.4g um), r [0, %.6g] (%d cells @ %.4g um)"
          % (XLO, XHI, NX, DX1 * 1e6, RTOP_DOMAIN, NY, DR1 * 1e6))
    print("plume length past aperture: %.4g m; hardware polygon closes at r=%.6g m" % (XHI - XEXIT, RTOP_GEOM))
    sccm = 20.0  # matches the paper's Fig. 1 example flow rate
    ndot = sccm * 4.478e17
    mdot = ndot * 6.6464791e-27
    fill = 4 * ndot / (1.07 * 145.5 * 3.141592653589793 * RAP ** 2)
    print("%.0f SCCM: Mdot = %.6e kg/s, aperture-law fill n = %.4e m^-3" % (sccm, mdot, fill))


if __name__ == "__main__":
    main()
