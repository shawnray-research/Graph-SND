import sys, os, glob, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anatomy import *
rows = []
for f in sorted(glob.glob(sys.argv[1])):
    z = np.load(f, allow_pickle=True)
    D = z["D"].astype(np.float64); n = D.shape[0]
    an = anatomy(D)
    rows.append(dict(file=os.path.basename(f), task=str(z["task"]), kind=str(z["kind"]), n=n,
                     seed=int(z["seed"]), iter=int(z["iter"]), reward=float(z["reward"]), mu=an["mu"],
                     cv=np.sqrt(an["s2_tot"]) / an["mu"], share_distinct=an["s2_main"] / an["s2_tot"],
                     share_rel=an["s2_int"] / an["s2_tot"], kappa=an["kappa"], s2int_mu2=an["s2_int"] / an["mu"] ** 2,
                     eff=efficiency_law(an)))
out = sys.argv[2] if len(sys.argv) > 2 else None
if out:
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['file']:45s} n={r['n']:3d} it={r['iter']:3d} mu={r['mu']:.4f} cv={r['cv']:.3f} distinct={r['share_distinct']:.3f} rel={r['share_rel']:.3f} kappa={r['kappa']:.2f} s2int/mu2={r['s2int_mu2']:.2e} eff={r['eff']:.2f}")
