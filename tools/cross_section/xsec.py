"""Quantum (Numerov partial-wave) diffusion / momentum-transfer cross sections
for a heavy molecule (BaF) colliding with He or Ne, using an LJ(12-6) model
potential V = eps[(rm/r)^12 - 2(rm/r)^6].

Units: length Angstrom, energy Kelvin, mass amu.  1 A^2 = 1e-20 m^2.
sigma_D(E) = 4pi/k^2 sum_l (l+1) sin^2(d_l - d_{l+1})
sigma_bar(T) = 1/2 int x^2 e^-x sigma_D(x kT) dx   (Skoff 2011 eq. 3)
"""
import numpy as np, json, sys

HB2 = 24.2537  # hbar^2/(2 amu kB A^2)  [K A^2]

def mu(m1, m2): return m1 * m2 / (m1 + m2)

def V_lj(r, eps, rm):
    s = (rm / r) ** 6
    return eps * (s * s - 2 * s)

def ricc(lmax, x):
    """Riccati-Bessel jhat_l(x)=x j_l(x), nhat_l(x)=x y_l(x), l=0..lmax."""
    y = np.empty(lmax + 2); y[0] = -np.cos(x); y[1] = -np.cos(x) / x - np.sin(x)
    for l in range(1, lmax + 1):
        y[l + 1] = (2 * l + 1) / x * y[l] - y[l - 1]
    # Miller downward for jhat
    start = lmax + 2 + int(max(20, 2 * np.sqrt(x) + x if x > lmax else 20 + np.sqrt(lmax * 10)))
    start = max(start, int(x) + 40)
    j = np.zeros(start + 2); j[start + 1] = 0.0; j[start] = 1e-300
    for l in range(start, 0, -1):
        j[l - 1] = (2 * l + 1) / x * j[l] - j[l + 1]
        if abs(j[l - 1]) > 1e250:
            j[l - 1:] *= 1e-250
    j *= np.sin(x) / j[0]
    return j[:lmax + 2], y[:lmax + 2]

def phase_shifts(E, m, eps, rm):
    A = m / HB2
    k = np.sqrt(A * E)
    lmax = int(k * rm * 2.5) + 40
    L = np.arange(lmax + 2)
    rmin = 0.55 * rm
    rmax = max(12 * rm, rm + 40.0 / k)
    h = min(0.004 * rm, 0.05 / k, 0.01)
    r = np.arange(rmin, rmax + h, h)
    W = A * V_lj(r, eps, rm) - k * k            # r-dependent part
    cent = L * (L + 1.0)
    h12 = h * h / 12.0
    u_prev = np.zeros(lmax + 2)
    u_cur = np.full(lmax + 2, 1e-30)
    f_prev = cent / r[0] ** 2 + W[0]
    f_cur = cent / r[1] ** 2 + W[1]
    i1 = len(r) - 30; i2 = len(r) - 1
    u1 = None
    for i in range(1, len(r) - 1):
        f_next = cent / r[i + 1] ** 2 + W[i + 1]
        u_next = ((2 + 10 * h12 * f_cur) * u_cur - (1 - h12 * f_prev) * u_prev) / (1 - h12 * f_next)
        # renormalize to avoid overflow
        s = np.abs(u_next)
        if s.max() > 1e100:
            s = np.where(s > 1e100, s, 1.0)
            u_next /= s; u_cur /= s
            if u1 is not None: u1 /= s
        u_prev, u_cur = u_cur, u_next
        f_prev, f_cur = f_cur, f_next
        if i + 1 == i1: u1 = u_next.copy()
    u2 = u_cur
    j1, n1 = ricc(lmax + 1, k * r[i1]); j2, n2 = ricc(lmax + 1, k * r[i2])
    j1, n1, j2, n2 = j1[:lmax + 2], n1[:lmax + 2], j2[:lmax + 2], n2[:lmax + 2]
    R = u1 / u2
    d = np.arctan2(R * j2 - j1, R * n2 - n1)   # tan d = (R j2 - j1)/(R n2 - n1)
    return k, d

def sigma_D(E, m, eps, rm):
    k, d = phase_shifts(E, m, eps, rm)
    l = np.arange(len(d) - 1)
    return 4 * np.pi / k**2 * np.sum((l + 1) * np.sin(d[:-1] - d[1:]) ** 2)

def sigma_bar(T, Eg, sg):
    x = np.linspace(1e-4, 30, 3000)
    s = np.interp(np.log(x * T), np.log(Eg), sg)
    return 0.5 * np.trapz(x**2 * np.exp(-x) * s, x)

# ---- classical LJ Omega(1,1)* (Neufeld 1972) for fast fitting ----
def omega11(Ts):
    return 1.06036 / Ts**0.15610 + 0.19300 / np.exp(0.47635 * Ts) + 1.03587 / np.exp(1.52996 * Ts) + 1.76474 / np.exp(3.89411 * Ts)

def sbar_classical(T, eps, rm):
    s = rm / 2 ** (1 / 6)
    return np.pi * s * s * omega11(T / eps)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fit":
        # YbF-He theory (Skoff 2011 Table I), A^2
        T = np.array([20., 80., 293.]); S = np.array([79.6, 55.8, 40.9])
        best = None
        for eps in np.linspace(2, 80, 391):
            for rm in np.linspace(3.0, 7.0, 401):
                p = sbar_classical(T, eps, rm)
                err = np.sum(np.log(p / S) ** 2)
                if best is None or err < best[0]: best = (err, eps, rm, p)
        print("best LJ fit eps=%.2f K rm=%.3f A rms_log=%.3f pred=%s" % (best[1], best[2], np.sqrt(best[0] / 3), best[3]))
    elif mode == "curve":
        m1, m2, eps, rm = map(float, sys.argv[2:6]); tag = sys.argv[6]
        m = mu(m1, m2)
        Eg = np.logspace(np.log10(0.2), np.log10(20000), 90)
        sg = np.array([sigma_D(E, m, eps, rm) for E in Eg])
        Ts = [1, 2, 4, 6, 10, 18, 20, 40, 80, 150, 293, 500, 1000, 2000, 3000]
        sb = {T: sigma_bar(T, Eg, sg) for T in Ts}
        json.dump({"tag": tag, "mu": m, "eps": eps, "rm": rm, "E": Eg.tolist(), "sD": sg.tolist(),
                   "Tbar": Ts, "sbar": [sb[T] for T in Ts]}, open(f"curve_{tag}.json", "w"))
        print(tag, "mu=%.3f" % m, " ".join("T%g:%.1f" % (T, sb[T]) for T in Ts))
