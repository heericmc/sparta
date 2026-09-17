import sys
sys.path.insert(0, 'tools')
from plot_fields_b5 import average_tail
import numpy as np

s = sys.argv[1]
path = f'results/he/takahashi-fig2-sccm{s}/field.grid'
d = average_tail(path, 0.25)
x = d['xc']; y = d['yc']
mask = (x > 0.0530) & (x < 0.0550) & (y < 0.0008)
idx = np.where(mask)[0]
nrho = d['nrho'][idx]; u = d['u'][idx]; t = d['t'][idx]
w = nrho
wsum = w.sum()
u_w = (w*u).sum()/wsum
t_w = (w*t).sum()/wsum
nrho_avg = nrho.mean()
print(f'{s} {len(idx)} {nrho_avg:.6e} {u_w:.6f} {t_w:.6f}')
