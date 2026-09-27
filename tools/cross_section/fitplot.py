import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# He: LJ fit to YbF-He theory (Skoff 2011).  Ne variants = He potential scaled He->Ne:
#   NeBap: Ba+-RG ab initio ratios (Buchachenko & Viehland 2018): eps x3.35, rm x0.864  <- recommended
#   NeBa : Ba-RG ab initio ratios: eps x4.5, rm x0.875
#   Ne/NeLo/NeHi: rare-gas-dimer style scaling eps x2.5 (1.9-3.0), rm ~unchanged
TAGS = ["He", "NeBap", "NeBa", "Ne", "NeLo", "NeHi", "BaHe", "BaNe"]
C = {t: json.load(open(f"curve_{t}.json")) for t in TAGS}

def sbar(T, f):
    x = np.linspace(1e-4, 30, 4000)
    return 0.5 * np.trapz(x**2 * np.exp(-x) * f(x * T), x)

fits = {}
for t, c in C.items():
    E = np.array(c["E"]); s = np.array(c["sD"]); m = E >= 3; E, s = E[m], s[m]
    best = None
    for b in np.linspace(0.3, 1.2, 91):
        for d in np.linspace(0.0, 0.3, 61):
            M = np.vstack([E**-b, E**-d]).T
            coef, *_ = np.linalg.lstsq(M / s[:, None], np.ones_like(s), rcond=None)
            if coef.min() < 0: continue
            err = np.sqrt(np.mean(((M @ coef) / s - 1) ** 2))
            if best is None or err < best[0]: best = (err, coef[0], b, coef[1], d)
    fits[t] = best
    f = lambda E, bb=best: bb[1] * E**-bb[2] + bb[3] * E**-bb[4]
    chk = " ".join(f"T{T}:{sbar(T, f):.0f}/{c['sbar'][c['Tbar'].index(T)]:.0f}" for T in [4, 18, 293, 1000])
    print(f"{t:6s} sigma_D(E)[1e-20 m^2] = {best[1]:.1f}*E^-{best[2]:.2f} + {best[3]:.1f}*E^-{best[4]:.3f}  rms {100*best[0]:.1f}%  (fit/exact sbar {chk})")
json.dump(fits, open("fits.json", "w"), indent=1)
F = lambda t, E: (fits[t][1] * E**-fits[t][2] + fits[t][3] * E**-fits[t][4]) * 1e-20

# ---- previous simulation values ----
PREV = {"const2e-18 (σ_high, BaF–Ne)": 2.0e-18, "const1.5e-19 (σ_low floor)": 1.5e-19,
        "repo default BaF–He (2.7e-18)": 2.7e-18}

Eg = np.logspace(np.log10(3), np.log10(5000), 300)
print("\nE_cm[K] v_rel[m/s]  BaF-Ne rec  band            BaF-He   | prev/rec: 2e-18  1.5e-19")
band = np.array([F(t, Eg) for t in ["NeBap", "NeBa", "Ne", "NeLo", "NeHi"]])
for E in [5, 10, 18, 30, 60, 100, 300, 1000, 3000]:
    b = np.array([F(t, E) for t in ["NeBap", "NeBa", "Ne", "NeLo", "NeHi"]])
    r = F("NeBap", E)
    print(f"{E:6g} {np.sqrt(E/1.0746e-3):8.0f}   {r:.2e}  {b.min():.1e}-{b.max():.1e}  {F('He', E):.2e}  |  {2e-18/r:5.2f}  {1.5e-19/r:5.2f}")

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
blue, orange, ink, ink2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
ax.fill_between(Eg, band.min(0), band.max(0), color=orange, alpha=0.18, lw=0)
ax.plot(Eg, F("He", Eg), color=blue, lw=2, label="BaF–He model (fit to YbF–He theory)")
ax.plot(Eg, F("NeBap", Eg), color=orange, lw=2, label="BaF–Ne recommended (Ba⁺-scaled); band = all He→Ne scalings")
ax.errorbar([4], [1.4e-18], yerr=[[0.7e-18], [0.7e-18]], fmt="s", ms=6, color=ink, capsize=3, label="BaF–He measured, ~4 K (Bu 2017; Albrecht 2020)")
ax.plot([4], [2.7e-18], "s", ms=6, color=ink)
ax.errorbar([293, 20], [41e-20, 203e-20], yerr=[[4e-20, 75e-20], [4e-20, 75e-20]], fmt="D", ms=5, mfc="#fcfcfb", color=ink, capsize=3,
            label="YbF–He measured σ̄_D (Skoff 2011), at E = kT")
# previous values: neutral gray reference lines, labelled directly
styles = ["--", ":", "-."]
for (name, v), ls in zip(PREV.items(), styles):
    ax.axhline(v, color=ink2, ls=ls, lw=1.2)
    ax.text(4800, v * 1.05, f"previous: {name}", ha="right", va="bottom", color=ink2, fontsize=8.3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1.2e-19, 4.5e-18); ax.set_xlim(3, 5000)
ax.set_xlabel("centre-of-mass collision energy E / k_B  (K)")
ax.set_ylabel("diffusion (momentum-transfer) cross section σ_D  (m²)")
ax.set_title("BaF–He / BaF–Ne diffusion cross section vs collision energy, with previous sim values", fontsize=10.5, loc="left")
sec = ax.secondary_xaxis("top", functions=(lambda E: np.sqrt(np.maximum(E, 1e-9) / 1.0746e-3), lambda v: 1.0746e-3 * v**2))
sec.set_xlabel("BaF–Ne relative speed (m/s)", color=ink2, fontsize=9)
ax.grid(True, which="major", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(fontsize=7.6, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.07))
fig.tight_layout()
fig.savefig("baf_ne_he_cross_section.png", facecolor=fig.get_facecolor())
print("saved")
