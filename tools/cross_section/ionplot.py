"""Compare the tracer's previous BaF+-Ne cross sections with the Ba+-Ne-anchored Q1(E)."""
import json, math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("ion_curve4.json"))
E = np.array(d["E"]); Qc = np.array(d["central(ra=1.5)"]) * 1e-20
Qhi = np.array(d["stiff(ra=1.8,x2)"]) * 1e-20; Qlo = np.array(d["soft(ra=1.2,x0.5)"]) * 1e-20
lq = lambda x, Q: np.exp(np.interp(np.log(x), np.log(E), np.log(Q)))

MU = 156.325 * 20.1797 / (156.325 + 20.1797)          # 17.87 u
EK_PER_V2 = MU * 1.66054e-27 / (2 * 1.380649e-23)     # E_cm[K] = this * v^2
T_CELL = 23.0
vref = math.sqrt(8 * 1.380649e-23 * T_CELL / (20.1797 * 1.66054e-27 * math.pi))
E_REF = EK_PER_V2 * vref**2
LAB = (156.325 + 20.1797) / 20.1797                   # E_lab(BaF+) = LAB * E_cm (Ne at rest)
K2EV = 8.617333e-5

def edep(Ecm, lo=2.0e-18, hi=1.5e-19):                # tracer sigma_of_speed, with E_cm = 1/2 mu v_coll^2
    return np.maximum(lo * np.sqrt(E_REF / np.maximum(Ecm, E_REF)), hi)

PREV = {"previous edep (default)": edep, "previous const 2.0e-18": lambda x: np.full(np.shape(x), 2.0e-18),
        "previous const 1.5e-19": lambda x: np.full(np.shape(x), 1.5e-19),
        "older const 5.0e-19": lambda x: np.full(np.shape(x), 5.0e-19)}

# ---- thermal average at the cell temperature ----
x = np.linspace(1e-4, 30, 6000)
sbar = lambda f, T: 0.5 * np.trapz(x**2 * np.exp(-x) * f(np.maximum(x * T, E[0])), x)
print(f"T={T_CELL} K: v_ref={vref:.1f} m/s  E_ref={E_REF:.1f} K  edep floor reached at E_cm={E_REF*(2e-18/1.5e-19)**2:.0f} K "
      f"(= {E_REF*(2e-18/1.5e-19)**2*LAB*K2EV:.2f} eV lab)")
print(f"thermal sigma_bar at {T_CELL} K: recommended {sbar(lambda e: lq(e, Qc), T_CELL):.2e} m^2 "
      f"(band {sbar(lambda e: lq(e, Qlo), T_CELL):.2e}-{sbar(lambda e: lq(e, Qhi), T_CELL):.2e})")

# ---- ratio table ----
E0 = 500 / K2EV / LAB                                  # 500 eV lab -> E_cm [K]
print(f"500 eV lab = E_cm {E0:.3g} K;  Q1 there = {lq(E0, Qc):.2e} (band {lq(E0, Qlo):.2e}-{lq(E0, Qhi):.2e})")
print(f"\n{'E_cm[K]':>9} {'E_lab[eV]':>9} {'rec Q1':>9} " + " ".join(f"{k[9:]:>16s}" for k in PREV))
for e in [30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, E0]:
    q = lq(e, Qc)
    print(f"{e:9.3g} {e*LAB*K2EV:9.3g} {q:9.2e} " + " ".join(f"{float(f(np.array([e]))[0])/q:15.2f}x" for f in PREV.values()))

# ---- slowing-down length: dE_lab/dx = -n Q1 f E_lab, f = 2mM/(m+M)^2  ->  L = (1/(n f)) int dlnE / Q1 ----
Eth = 1.5 * T_CELL                                     # stop at ~ thermal mean energy
g = np.geomspace(Eth, E0, 4000)
I = lambda f: np.trapz(1 / f(g), np.log(g))
Irec, Ilo, Ihi = I(lambda e: lq(e, Qc)), I(lambda e: lq(e, Qhi)), I(lambda e: lq(e, Qlo))
print(f"\nslowing-down length 500 eV -> thermal, relative to recommended (band {Ilo/Irec:.2f}-{Ihi/Irec:.2f}):")
for k, f in PREV.items():
    print(f"  {k:28s} L/L_rec = {I(f)/Irec:5.2f}")
edges = [Eth, 1e2, 1e3, 1e4, 1e5, E0]
print("  share of recommended path length by E_cm range: " + ", ".join(
    f"{a:.0f}-{b:.0g} K: {100*np.trapz(1/lq(gg, Qc), np.log(gg))/Irec:.0f}%" for a, b in zip(edges[:-1], edges[1:])
    for gg in [np.geomspace(a, b, 500)]))

# ---- plot ----
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=150)
bg, ink, ink2, orange, blue = "#fcfcfb", "#0b0b0b", "#52514e", "#eb6834", "#2a78d6"
fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
Eg = np.geomspace(3, 1e6, 600)
ax.fill_between(Eg, lq(Eg, Qlo), lq(Eg, Qhi), color=orange, alpha=0.2, lw=0)
ax.plot(Eg, lq(Eg, Qc), color=orange, lw=2.2, label="BaF⁺–Ne recommended Q⁽¹⁾ (Ba⁺–Ne ab initio + ZBL wall); band = wall uncertainty")
C4 = 1.37 * 315775.02 * 0.529177**4
ax.plot(Eg, 1.105 * 2 * np.pi * np.sqrt(C4 / Eg) * 1e-20, color=ink2, lw=1, alpha=0.6,
        label="pure polarization (1.105 × Langevin), for reference")
ax.plot(Eg, edep(Eg), color=blue, lw=2.2, label=f"previous edep model (σ_low 2e-18 → 1/v → floor 1.5e-19, T={T_CELL:.0f} K)")
for (k, f), ls in zip(list(PREV.items())[1:], ["--", ":", "-."]):
    v = float(f(np.array([1.0]))[0])
    ax.axhline(v, color=ink2, ls=ls, lw=1.1)
    ax.text(9e5, v * 1.06, k, ha="right", va="bottom", color=ink2, fontsize=8.2)
ax.axvline(E0, color=ink2, lw=0.8, alpha=0.5); ax.text(E0 * 0.9, 3.3e-20, "500 eV\ninjection", ha="right", color=ink2, fontsize=8)
ax.axvline(1.5 * T_CELL, color=ink2, lw=0.8, alpha=0.5); ax.text(1.5 * T_CELL * 1.1, 3.3e-20, "thermal\n(23 K)", color=ink2, fontsize=8)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(3, 1e6); ax.set_ylim(2.5e-20, 1e-17)
ax.set_xlabel("centre-of-mass collision energy E / k_B  (K)")
ax.set_ylabel("momentum-transfer cross section  (m²)")
ax.set_title("BaF⁺–Ne cross section: previous tracer values vs ab-initio-anchored estimate", fontsize=10.5, loc="left")
sec = ax.secondary_xaxis("top", functions=(lambda e: e * LAB * K2EV, lambda ev: ev / LAB / K2EV))
sec.set_xlabel("BaF⁺ lab kinetic energy, Ne at rest (eV)", color=ink2, fontsize=9)
ax.grid(True, which="major", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(fontsize=7.6, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("bafplus_ne_cross_section_vs_previous.png", facecolor=bg)
print("saved")
