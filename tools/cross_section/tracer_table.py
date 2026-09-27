"""Tracer-effective cross section sigma_t(E_cm) for ParticleTracing3D.jl.

The tracer draws collisions at rate n*sigma_t*v_coll (v_coll = sqrt(8kT/(pi m_gas) + V^2)),
picks a flux-weighted Maxwellian partner and scatters isotropically in the CM frame, so its
mean drag on an ion moving at V through gas at T is  n*sigma_t*v_coll*mu*<g g_par>/<g>.
The true drag for momentum-transfer cross section Q1(g) is  n*mu*<Q1(g) g g_par>.
Equating the two at every V gives
    sigma_t(V) = <Q1(g) g g_par> <g> / (v_coll <g g_par>)
(averages over the gas Maxwellian).  sigma_t -> Q1 for V >> thermal, and -> the thermal
Chapman-Enskog value (so the diffusion coefficient is right) as V -> 0.  Tabulated vs the
E_cm = mu v_coll^2/2 the tracer computes, at T = 23 K (exact at V=0 for any T, T-independent at
high V; approximate in between for other local temperatures).
"""
import json, numpy as np
kB = 8314.46; m, M = 20.1797, 156.325; mu = m*M/(m+M); T = 23.0
d = json.load(open("ion_curve4.json")); E = np.array(d["E"])
cols = {"central": "central(ra=1.5)", "low": "soft(ra=1.2,x0.5)", "high": "stiff(ra=1.8,x2)"}
def Q1fun(key):
    lE, lQ = np.log(E), np.log(np.array(d[key])*1e-20)
    s0 = (lQ[1]-lQ[0])/(lE[1]-lE[0])                 # power-law extrapolation below 1 K
    def f(Ek):
        x = np.log(np.maximum(Ek, 1e-12))
        return np.exp(np.where(x < lE[0], lQ[0] + s0*(x-lE[0]), np.interp(x, lE, lQ)))
    return f
rng = np.random.default_rng(1)
vg = rng.normal(0, np.sqrt(kB*T/m), size=(2_000_000, 3))
vref = np.sqrt(8*kB*T/(np.pi*m))
Vs = np.concatenate([[0.5], np.geomspace(2, 40000, 90)])
out = {k: [] for k in cols}; Ec = []
for V in Vs:
    g_vec = np.array([V, 0, 0]) - vg; g = np.linalg.norm(g_vec, axis=1); gpar = g_vec[:, 0]
    vc = np.sqrt(vref**2 + V**2); Ec.append(mu*vc**2/(2*kB))
    for k, key in cols.items():
        Q = Q1fun(key)(mu*g**2/(2*kB))
        out[k].append(np.mean(Q*g*gpar)*np.mean(g)/(vc*np.mean(g*gpar)))
Ec = np.array(Ec)
ref = Q1fun(cols["central"])
print(f"v_ref={vref:.1f} m/s, E_cm(V=0)={Ec[0]:.2f} K")
print("  V[m/s]   E_cm[K]   sigma_t     Q1(E_cm)   ratio")
for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    print(f"{Vs[i]:8.1f} {Ec[i]:9.3g} {out['central'][i]:.3e}  {float(ref(Ec[i])):.3e}  {out['central'][i]/float(ref(Ec[i])):.3f}")
# thermal-average cross-check: Chapman-Enskog sigma_bar(23 K) * gbar / vref
x = np.linspace(1e-4, 30, 6000); sb = 0.5*np.trapz(x**2*np.exp(-x)*ref(x*T), x)
gbar = np.sqrt(8*kB*T/(np.pi*mu))
print(f"check V->0: sigma_bar(23K)*gbar/vref = {sb*gbar/vref:.3e}  vs sigma_t(V=0.5) = {out['central'][0]:.3e}")
json.dump({"Ec": Ec.tolist(), **out}, open("tracer_sigma_t.json", "w"))
