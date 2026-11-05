"""E1: exact-law validation.  E2: matched-budget design comparison + stress tests."""
import sys, json, glob, os
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anatomy import *

REPS_E1 = 3000
REPS_E2 = 2000


def e1_one(job):
    mid, source, family, D, seed = job
    rng = np.random.default_rng(1000 + seed)
    an = anatomy(D); n = an["n"]
    des = Designs(D, rng)
    rows = []
    for k in [1, 2, 4, 8]:
        if k >= n - 1:
            continue
        m = n * k // 2
        vb = var_balanced(an, k)
        for name, fn, vpred in [
            ("round_robin", lambda: des.round_robin(k, REPS_E1)[0], vb),
            ("shuffled_circulant", lambda: des.shuffled_circulant(k, REPS_E1)[0], vb),
            ("uniform", lambda: des.uniform(m, REPS_E1)[0], var_uniform(an, m)),
            ("bernoulli_ht", lambda: des.bernoulli_ht(m, REPS_E1)[0], var_bernoulli_ht(D, m / an["N"])),
        ]:
            est = fn()
            rows.append(dict(matrix=mid, source=source, family=family, n=n, k=k, m=m, design=name,
                             var_pred=vpred, var_meas=float(np.var(est, ddof=1)),
                             bias=float(est.mean() - an["mu"]), mu=an["mu"], kappa=an["kappa"],
                             s2_tot=an["s2_tot"], s2_main=an["s2_main"], s2_int=an["s2_int"],
                             eff_pred=efficiency_law(an), reps=REPS_E1))
    return rows


def e2_one(job):
    mid, source, family, D, seed = job
    rng = np.random.default_rng(2000 + seed)
    an = anatomy(D); n = an["n"]; mu = an["mu"]
    des = Designs(D, rng)
    rows = []
    for k in [1, 2, 4, 8, 16]:
        if k >= n - 1:
            continue
        m = n * k // 2
        rand = {
            "round_robin": lambda: des.round_robin(k, REPS_E2),
            "uniform": lambda: des.uniform(m, REPS_E2),
            "bernoulli_ht": lambda: des.bernoulli_ht(m, REPS_E2),
            "bernoulli_ratio": lambda: des.bernoulli_ratio(m, REPS_E2),
            "agent_subsample": lambda: des.agent_subsample(m, REPS_E2),
            "landmark": lambda: des.landmark(m, REPS_E2),
        }
        for name, fn in rand.items():
            est, mm = fn()
            est = est[np.isfinite(est)]
            rows.append(dict(matrix=mid, source=source, family=family, n=n, k=k, m_target=m, m=mm,
                             design=name, rel_rmse=float(np.sqrt(np.mean((est - mu) ** 2)) / mu),
                             rel_bias=float((est.mean() - mu) / mu), fixed=0, kappa=an["kappa"]))
        fixed = {
            "local_knn": lambda: des.local_knn(k),
            "ring_fixed": lambda: des.ring_fixed(k),
        }
        for name, fn in fixed.items():
            val, mm = fn()
            rows.append(dict(matrix=mid, source=source, family=family, n=n, k=k, m_target=m, m=mm,
                             design=name, rel_rmse=float(abs(val - mu) / mu),
                             rel_bias=float((val - mu) / mu), fixed=1, kappa=an["kappa"]))
        # fixed random expander: deterministic per draw; report RMS over 100 independent draws
        vals = np.array([des.expander_fixed(k, seed=1000 * seed + t)[0] for t in range(100)])
        rows.append(dict(matrix=mid, source=source, family=family, n=n, k=k, m_target=m, m=m,
                         design="expander_fixed", rel_rmse=float(np.sqrt(np.mean((vals - mu) ** 2)) / mu),
                         rel_bias=float((vals.mean() - mu) / mu), fixed=1, kappa=an["kappa"]))
    return rows


def load_real(patterns):
    jobs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            z = np.load(f, allow_pickle=True)
            D = np.asarray(z["D"], dtype=np.float64)
            n = D.shape[0]
            if n % 2 or n < 6:
                continue
            fam = f"{str(z['task'])}_{str(z['kind']) if 'kind' in z else 'trained'}"
            jobs.append((os.path.basename(f)[:-4], "real", fam, D, int(z["seed"]) if "seed" in z else 0))
    return jobs


if __name__ == "__main__":
    import csv
    which = sys.argv[1]          # e1 | e2
    outcsv = sys.argv[2]
    real_patterns = sys.argv[3:]
    jobs = []
    if which == "e1" and os.environ.get("SKIP_SYNTH") != "1":
        for fam in FAMILIES:
            for n in [50, 100, 200]:
                for s in range(2):
                    D = synthetic(fam, n, np.random.default_rng(100 * s + n))
                    jobs.append((f"{fam}_n{n}_s{s}", "synthetic", fam, D, s))
    elif which == "e2" and os.environ.get("SKIP_SYNTH") != "1":
        for fam in FAMILIES:
            for s in range(3):
                D = synthetic(fam, 200, np.random.default_rng(7 + 31 * s))
                jobs.append((f"{fam}_n200_s{s}", "synthetic", fam, D, s))
    jobs += load_real(real_patterns)
    fn = e1_one if which == "e1" else e2_one
    with Pool(int(os.environ.get("NPROC", 6))) as pool:
        res = pool.map(fn, jobs, chunksize=1)
    rows = [r for rr in res for r in rr]
    with open(outcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows from {len(jobs)} matrices -> {outcsv}")
