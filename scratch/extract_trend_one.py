import sys
sys.path.insert(0, 'tools')
from plot_fields_b5 import average_tail

s = sys.argv[1]
path = f'results/he/takahashi-fig2-sccm{s}/field.grid'
d = average_tail(path, 0.25)  # frac only affects u/v/t averaging; trend covers all frames
with open(f'/root/trend_sccm{s}.csv', 'w') as f:
    f.write('step,bulk_nrho_m-3\n')
    for step, val in d['trend']:
        f.write(f'{step},{val}\n')
print(f'sccm{s}: wrote {len(d["trend"])} trend points')
