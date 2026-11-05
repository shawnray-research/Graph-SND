"""E4b: 1024-model zoo (4 recipes x 256 seeds): exact laws and designs at n=1024."""
import sys, os, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anatomy import *
z = np.load(sys.argv[1], allow_pickle=True); D = z["D"].astype(np.float64)
an = anatomy(D); n = an["n"]; mu = an["mu"]; N = an["N"]
rng = np.random.default_rng(0); des = Designs(D, rng)
rows = []
REPS = 2000
rowsum = D.sum(1)
def landmark_fast(m, reps):
    k = max(1, int(round(m / (n - 1)))); out = np.empty(reps)
    for t in range(reps):
        L = rng.choice(n, size=k, replace=False)
        sub = D[np.ix_(L, L)][np.triu_indices(k, 1)].sum()
        out[t] = (rowsum[L].sum() - sub) / (k * (n - 1) - k * (k - 1) // 2)
    return out, k * (n - 1) - k * (k - 1) // 2
for k in [1, 2, 4, 8, 16]:
    m = n * k // 2
    res = {"round_robin": des.round_robin(k, REPS), "uniform": des.uniform(m, REPS),
           "bernoulli_ht": des.bernoulli_ht(m, REPS), "bernoulli_ratio": des.bernoulli_ratio(m, REPS),
           "agent_subsample": des.agent_subsample(m, REPS), "landmark": landmark_fast(m, REPS)}
    vals = np.array([des.expander_fixed(k, seed=t)[0] for t in range(100)])
    pred = {"round_robin": var_balanced(an, k), "uniform": var_uniform(an, m),
            "bernoulli_ht": var_bernoulli_ht(D, m / N)}
    for name, (est, mm) in res.items():
        rows.append(dict(design=name, k=k, m=mm, rel_rmse=float(np.sqrt(np.mean((est - mu) ** 2)) / mu),
                         var_meas=float(np.var(est, ddof=1)), var_pred=float(pred.get(name, np.nan))))
    rows.append(dict(design="expander_fixed", k=k, m=m, rel_rmse=float(np.sqrt(np.mean((vals - mu) ** 2)) / mu),
                     var_meas=float(np.var(vals, ddof=1)), var_pred=float(var_balanced(an, k))))
    lv, _ = des.local_knn(k); rf, _ = des.ring_fixed(k)
    rows.append(dict(design="local_knn", k=k, m=m, rel_rmse=abs(lv - mu) / mu, var_meas=0, var_pred=np.nan))
    rows.append(dict(design="ring_fixed", k=k, m=m, rel_rmse=abs(rf - mu) / mu, var_meas=0, var_pred=np.nan))
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"n={n} N={int(N)} mu={mu:.4f} kappa={an['kappa']:.3f} eff={efficiency_law(an):.3f}")
for r in rows:
    vp = r["var_pred"]; ratio = r["var_meas"] / vp if vp == vp and vp > 0 else float("nan")
    print(f"k={r['k']:2d} m={r['m']:6d} {r['design']:16s} relRMSE={100*r['rel_rmse']:.3f}%  meas/pred={ratio:.3f}")
