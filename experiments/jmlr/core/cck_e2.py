"""Adaptive pivot baseline of Chechik, Cohen & Kaplan (2015) ("CCK") at the equal budget m = 2n
of Table 2 (tab:e2), next to the Balanced (round-robin, k=4) and Uniform-pairs rows.

Run from the repository root: python experiments/jmlr/core/cck_e2.py results/jmlr/cck/cck_results.json
(The VMAS rows in results/jmlr/cck/cck_results.json used the long-run tables saved by 00:00 UTC on
2026-11-05, iterations 50-200.)

CCK draw (budget m):
  1. pivot p ~ Uniform{0..n-1}
  2. evaluate the n-1 pivot pairs d_{pj} exactly; S_piv = sum_j d_{pj}
  3. proxy w_i = d_{pi} + S_piv/(n-1) for i != p
  4. m' = m - (n-1) pairs i.i.d. with replacement among the non-pivot pairs {i,j} (i,j != p) with
     pi_ij = (w_i + w_j) / ((n-2) * sum_{k != p} w_k)   (sums to one over the non-pivot pairs)
     sampled exactly as: endpoint i ~ w/W, partner j ~ Uniform(non-pivot \\ {i})
  5. Hansen-Hurwitz: S_hat = mean_s d_{e_s}/pi_{e_s};  mu_hat = (S_piv + S_hat)/N   (unbiased)
Exact variance (no Monte Carlo error): Var = mean_p (sum_e d_e^2/pi_e - S_np^2) / (m' N^2).
Aggregation over the three synthetic tables per family = sqrt(mean of per-table relative MSE),
which is what fig_designs.py / Table 2 use (checked against e2_synth_v2.csv).
"""
import sys, os, glob, re, json
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anatomy import synthetic, FAMILIES, Designs, anatomy, var_balanced, var_uniform  # noqa: E402

REPS = 2000
K = 4


def cck_draws(D, m, reps, rng, pivot=None):
    """pivot=None: random pivot per draw (main row). pivot=int: fixed pivot (oracle variants)."""
    n = D.shape[0]
    N = n * (n - 1) / 2
    mp = m - (n - 1)
    assert mp > 0
    out = np.empty(reps)
    for t in range(reps):
        p = int(rng.integers(n)) if pivot is None else int(pivot)
        row = D[p]
        S_piv = row.sum() - row[p]
        w = row + S_piv / (n - 1)
        w[p] = 0.0
        W = w.sum()
        i = rng.choice(n, size=mp, p=w / W)
        r = rng.integers(0, n - 2, size=mp)
        a = np.minimum(i, p)
        b = np.maximum(i, p)
        j = r + (r >= a)
        j = j + (j >= b)
        pi = (w[i] + w[j]) / ((n - 2) * W)
        out[t] = (S_piv + np.mean(D[i, j] / pi)) / N
    return out


def cck_exact_var_per_pivot(D, m):
    """Exact Var(mu_hat | pivot p) for each p (the estimator is conditionally unbiased)."""
    n = D.shape[0]
    N = n * (n - 1) / 2
    mp = m - (n - 1)
    iu0, iu1 = np.triu_indices(n, 1)
    V = np.empty(n)
    for p in range(n):
        row = D[p]
        S_piv = row.sum() - row[p]
        w = row + S_piv / (n - 1)
        w[p] = 0.0
        W = w.sum()
        mask = (iu0 != p) & (iu1 != p)
        ii, jj = iu0[mask], iu1[mask]
        pi = (w[ii] + w[jj]) / ((n - 2) * W)
        assert abs(pi.sum() - 1) < 1e-9
        d = D[ii, jj]
        V[p] = (np.sum(d * d / pi) - d.sum() ** 2) / (mp * N ** 2)
    return V


def run_one(job):
    name, group, D, seed = job
    D = np.asarray(D, dtype=np.float64)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    n = D.shape[0]
    m = 2 * n
    an = anatomy(D)
    mu = an["mu"]
    ss = np.random.SeedSequence([20151, seed])
    r1, r2, r3 = [np.random.default_rng(s) for s in ss.spawn(3)]
    des = Designs(D, r1)
    rr, m_rr = des.round_robin(K, REPS)
    un, m_un = des.uniform(m, REPS)
    assert m_rr == m and m_un == m
    ck = cck_draws(D, m, REPS, r2)
    Vp = cck_exact_var_per_pivot(D, m)
    medoid = int(np.argmin(D.sum(1)))
    ck_med = cck_draws(D, m, REPS, r3, pivot=medoid)
    rel = lambda e: float(np.sqrt(np.mean((e - mu) ** 2)) / mu)
    return dict(
        name=name, group=group, n=n, m=m, m_prime=m - (n - 1), mu=mu, kappa=an["kappa"],
        rr=rel(rr), un=rel(un), cck=rel(ck), cck_medoid_mc=rel(ck_med),
        cck_bias=float((ck.mean() - mu) / mu),
        rr_exact=float(np.sqrt(var_balanced(an, K)) / mu),
        un_exact=float(np.sqrt(var_uniform(an, m)) / mu),  # without-replacement law
        cck_exact=float(np.sqrt(Vp.mean()) / mu),
        cck_medoid_exact=float(np.sqrt(Vp[medoid]) / mu),
        cck_bestpivot_exact=float(np.sqrt(Vp.min()) / mu),
        cck_worstpivot_exact=float(np.sqrt(Vp.max()) / mu),
    )


def jobs_synth():
    J = []
    for fi, fam in enumerate(FAMILIES):
        for s in range(3):
            D = synthetic(fam, 200, np.random.default_rng(7 + 31 * s))
            J.append((f"{fam}_n200_s{s}", "synth:" + fam, D, 100 * fi + s))
    return J


def jobs_real():
    J = []
    seed = 10000
    for f in sorted(glob.glob("results/jmlr/matrices/zoo/zoo_fmnist*.npz")):
        z = np.load(f, allow_pickle=True)
        J.append((os.path.basename(f)[:-4], "zoo", np.asarray(z["D"], float), seed)); seed += 1
    best = {}
    for f in glob.glob("results/jmlr/matrices/vmas_long/*_trained_it*.npz"):
        mo = re.match(r"(.*)_it(\d+)\.npz$", os.path.basename(f))
        pre, it = mo.group(1), int(mo.group(2))
        if pre not in best or it > best[pre][0]:
            best[pre] = (it, f)
    for pre in sorted(best):
        it, f = best[pre]
        z = np.load(f, allow_pickle=True)
        J.append((f"{pre}_it{it}", "vmas_long", np.asarray(z["D"], float), seed)); seed += 1
    for f in sorted(glob.glob("results/jmlr/samples/judged*.npz")):
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            D = np.asarray(z[k], float)
            if D.ndim != 2 or D.shape[0] != D.shape[1] or not np.all(np.isfinite(D)):
                print("skip", f, k, D.shape, flush=True)
                continue
            J.append((os.path.basename(f)[:-4] + ":" + k, "judge", D, seed)); seed += 1
    return J


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/jmlr/cck/cck_results.json"
    J = jobs_synth() + jobs_real()
    with Pool(6) as pool:
        res = pool.map(run_one, J, chunksize=1)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out, len(res))
