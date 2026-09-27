"""Classical momentum-transfer (diffusion) cross section Q1(E) for BaF+ - Ne,
using the Ba+-Ne ab initio CCSD(T)/CBS potential of Buchachenko & Viehland,
JCP 148, 154304 (2018), Table IV, as the proxy:
    sigma0 = 3.254 A (V=0), Re = 4.267 A, De = 75.0 cm^-1, C4 = 1.37 a.u.
Model: Tang-Toennies form V = A exp(-b r) - f4(br) C4/r^4 - f6(br) C6/r^6,
with A, b, C6 fitted to (sigma0, Re, De).  High-energy check: ZBL universal
screened-Coulomb repulsion for Ba-Ne (+ same C4 tail).
Units: A, K.  1 A^2 = 1e-20 m^2.
"""
import numpy as np, json, math

EH_K = 315775.02; A0 = 0.529177
C4 = 1.37 * EH_K * A0**4            # K A^4
SIG0, RE, DE = 3.254, 4.267, 75.0 * 1.438777

def tt(n, x):
    s = np.zeros_like(x); term = np.ones_like(x)
    for k in range(n + 1):
        if k: term = term * x / k
        s = s + term
    return 1 - np.exp(-x) * s

def V_tt(r, A, b, C6):
    return A * np.exp(-b * r) - tt(4, b * r) * C4 / r**4 - tt(6, b * r) * C6 / r**6

def V_zbl(r, Z1=56, Z2=10):
    a = 0.8854 * A0 / (Z1**0.23 + Z2**0.23); x = r / a
    phi = 0.1818*np.exp(-3.2*x) + 0.5099*np.exp(-0.9423*x) + 0.2802*np.exp(-0.4029*x) + 0.02817*np.exp(-0.2016*x)
    return Z1 * Z2 * 14.3996 / r * phi * 11604.5 - C4 / r**4 * tt(4, 2.0 * r)   # eV*A -> K

def fit_tt():
    r = np.linspace(2.5, 8, 5501)
    best = None
    for b in np.linspace(1.5, 5.0, 351):
        for C6 in np.linspace(0, 3e6, 61):
            att = tt(4, b*SIG0)*C4/SIG0**4 + tt(6, b*SIG0)*C6/SIG0**6
            A = att / np.exp(-b*SIG0)
            v = V_tt(r, A, b, C6); i = np.argmin(v)
            err = ((r[i]-RE)/0.01)**2 + ((-v[i]-DE)/1.0)**2
            if best is None or err < best[0]: best = (err, A, b, C6, r[i], -v[i])
    return best

def Q1(E, V, rgrid, nb=500, ns=160):
    """classical momentum-transfer cross section at CM energy E (K)."""
    bL = (4 * C4 / E) ** 0.25                       # Langevin capture radius
    Vr = V(rgrid)
    # hard-ish radius where V = E
    rE = rgrid[np.where(Vr > E)[0].max()] if np.any(Vr > E) else rgrid[0]
    bmax = 4 * max(bL, rE, 1.0)
    bs = np.linspace(1e-4, bmax, nb)
    s, w = np.polynomial.legendre.leggauss(ns); s = 0.5*(s+1); w = 0.5*w
    chi = np.empty(nb)
    for i, b in enumerate(bs):
        F = 1 - Vr / E - b*b / rgrid**2
        neg = np.where(F <= 0)[0]
        if len(neg) == 0: j = 0
        else: j = neg.max()
        if j + 1 >= len(rgrid): chi[i] = 0; continue
        # linear refine root between rgrid[j], rgrid[j+1]
        r1, r2, F1, F2 = rgrid[j], rgrid[j+1], F[j], F[j+1]
        rc = r1 - F1*(r2-r1)/(F2-F1) if F2 != F1 else r2
        u = 1 - s*s; r = rc / u
        Fr = 1 - V(r)/E - b*b/r**2
        Fr = np.maximum(Fr, 1e-14)
        integ = 2*s/np.sqrt(Fr)                   # dr/(r^2 sqrt F) = (du/rc)/sqrt F, du = 2 s ds
        chi[i] = np.pi - 2*b/rc*np.sum(w*integ)
    return 2*np.pi*np.trapz((1-np.cos(chi))*bs, bs)

if __name__ == "__main__":
    best = fit_tt()
    _, A, b, C6, re, de = best
    print(f"TT fit: A={A:.4g} K, b={b:.3f}/A, C6={C6:.3g} K A^6 ({C6/EH_K/A0**6:.0f} a.u.) -> Re={re:.3f}, De={de:.1f} K (target {RE}, {DE:.1f})")
    rg = np.concatenate([np.linspace(0.05, 2, 4000), np.linspace(2, 60, 20000)[1:], np.geomspace(60, 3000, 3000)[1:]])
    Vt = lambda r: V_tt(r, A, b, C6)
    for r in [1.0, 1.5, 2.0, 2.5, 3.0]:
        print(f"  V(r={r}) TT={Vt(np.array([r]))[0]:.3g} K  ZBL={V_zbl(np.array([r]))[0]:.3g} K")
    # validation: pure polarization + small hard core -> Q1/sigma_L -> 1.105
    Vp = lambda r: 1e9*(1.0/r)**12 - C4/r**4
    for E in [1.0, 10.0]:
        sL = 2*np.pi*math.sqrt(C4/E)
        print(f"  check pure r^-4: E={E} K  Q1/sigma_L = {Q1(E, Vp, rg)/sL:.3f} (expect ~1.105)")
    Es = np.geomspace(1, 1e6, 61)
    qt = [Q1(E, Vt, rg) for E in Es]
    qz = [Q1(E, V_zbl, rg) for E in Es]
    json.dump({"E": Es.tolist(), "Q_tt": qt, "Q_zbl": qz, "A": A, "b": b, "C6": C6}, open("ion_curve.json", "w"))
    for E, a, z in zip(Es[::5], qt[::5], qz[::5]):
        print(f"E={E:9.3g} K  Q1_TT={a:8.2f} A^2  Q1_ZBL={z:8.2f} A^2  Langevin*1.105={1.105*2*np.pi*math.sqrt(C4/E):8.2f}")
