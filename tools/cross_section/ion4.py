"""Final BaF+-Ne proxy (Ba+-Ne) Q1(E).
r >= 2.9 A : ab-initio-matched model (sigma0, Re, De, C4 of Buchachenko & Viehland 2018 CBS)
r <= r_a   : ZBL (Ba-Ne) universal repulsion (scaled by s for uncertainty)
between    : exponential (log-linear) interpolation of V.
"""
import numpy as np, json, math
from ion import C4, SIG0, RE, DE, tt, V_zbl, Q1, EH_K, A0
AU6 = EH_K*A0**6
b, beta, C6 = 1.80, 3.80, 225*AU6
W = lambda r: tt(4, beta*r)*C4/r**4 + tt(6, beta*r)*C6/r**6
A = W(np.array([SIG0]))[0]*math.exp(b*SIG0)
Vm = lambda r: A*np.exp(-np.minimum(b*r, 700)) - W(r)
RJ = 2.9
def make(ra, s=1.0):
    v1 = Vm(np.array([RJ]))[0]; v0 = s*V_zbl(np.array([ra]))[0]
    k = math.log(v0/v1)/(RJ-ra)
    def V(r):
        mid = v1*np.exp(k*(RJ - r))
        return np.where(r >= RJ, Vm(r), np.where(r <= ra, s*V_zbl(r), mid))
    return V, k
rg = np.concatenate([np.linspace(0.05, 2, 4000), np.linspace(2, 60, 20000)[1:], np.geomspace(60, 3000, 3000)[1:]])
E = np.geomspace(1, 1e6, 49); out = {"E": E.tolist()}
var = {"central(ra=1.5)": (1.5, 1.0), "stiff(ra=1.8,x2)": (1.8, 2.0), "soft(ra=1.2,x0.5)": (1.2, 0.5)}
print(f"V_model(2.9)={Vm(np.array([RJ]))[0]:.1f} K")
for name, (ra, s) in var.items():
    V, k = make(ra, s)
    print(f"{name}: effective wall slope {k:.2f}/A; V(2.5)={V(np.array([2.5]))[0]:.0f} K V(2.0)={V(np.array([2.0]))[0]:.0f} K")
    out[name] = [Q1(e, V, rg, nb=700) for e in E]
json.dump(out, open("ion_curve4.json", "w"))
print("E[K]      E_lab[eV]  " + "  ".join(f"{k:>18s}" for k in var))
for i in range(0, 49, 3):
    print(f"{E[i]:9.3g} {E[i]/0.11434*8.617e-5:9.3g}  " + "  ".join(f"{out[k][i]:18.2f}" for k in var))
