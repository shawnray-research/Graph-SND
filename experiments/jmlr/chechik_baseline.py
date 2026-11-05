"""Baseline: adaptive pivot scheme of Chechik, Cohen & Kaplan (2015) vs balanced and uniform designs.

Per draw (budget m comparisons, m >= n):
  (1) pivot u ~ Uniform{0..n-1}; evaluate d(u,j) for all j != u   (n-1 comparisons, counted in m)
  (2) proxies q_i = d(u,i) + dbar_u (i != u), q_u = dbar_u, dbar_u = mean_j d(u,j)
  (3) m' = m - (n-1) further pairs, i.i.d. with P{i,j} proportional to q_i + q_j
  (4) Hansen-Hurwitz estimate of the total over the pairs NOT containing u, plus the
      exactly known pivot total T_u = sum_j d(u,j); mu_hat = (T_u + T_rest_hat) / N.

Estimator variants (all exactly unbiased for mu, conditionally on u and hence unconditionally):
  adaptive_pivot      [primary] the law P{i,j} ~ q_i+q_j conditioned on u not in {i,j} (equivalently: draw
                      i ~ q_i/sum(q), j uniform among the other n-1, and reject the already-known pivot
                      pairs, which cost no comparison); i ~ q_i/(sum q - q_u) over i != u, j uniform among
                      the n-2 others; p_e = (q_i+q_j)/((n-2)(sum q - q_u)); T_rest_hat = (1/m') sum_e d_e/p_e.
  adaptive_pivot_zf   unrestricted sampler (p_e = (q_i+q_j)/((n-1) sum q)), HH restricted to non-pivot pairs
                      with draws that land on a pivot pair zero-filled (still counted in m'):
                      T_rest_hat = (1/m') sum_e 1[u not in e] d_e / p_e.
  adaptive_pivot_hh   unrestricted sampler, plain HH over all N pairs, ignores T_u:
                      mu_hat = (1/(m' N)) sum_e d_e / p_e.

Harness check: round_robin / shuffled_circulant (balanced, k-regular, m = nk/2) and uniform pairs without
replacement, via anatomy.Designs (same code as core/e1e2.py), on the same tables
(synthetic(fam, 200, default_rng(7 + 31 s)), s = 0,1,2) as Table 2 (results/jmlr/core/e2_synth_v2.csv).

Usage: python3 chechik_baseline.py out.csv [zoo_fmnist.npz] [judged_claude.npz]
"""
import sys, os, csv
import numpy as np
from multiprocessing import Pool

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "core"))
sys.path.insert(0, _here)
from anatomy import anatomy, Designs, synthetic, FAMILIES

REPS = int(os.environ.get("REPS", 2000))
KS = [2, 4, 8, 16]          # m = n k / 2  ->  m in {n, 2n, 4n, 8n}


def sample_full(q, mp, rng):
    """i.i.d. pairs with P{i,j} = (q_i+q_j)/((n-1) sum q): i ~ q/sum(q), j uniform among the other n-1."""
    n = len(q)
    cq = np.cumsum(q)
    i = np.minimum(np.searchsorted(cq, rng.random(mp) * cq[-1], side="right"), n - 1)
    j = rng.integers(n - 1, size=mp)
    j = j + (j >= i)
    return i, j, (q[i] + q[j]) / ((n - 1) * cq[-1])


def sample_restricted(q, u, mp, rng):
    """Same law conditioned on u not in {i,j} (= rejection of already-known pivot pairs):
    P{i,j} = (q_i+q_j)/((n-2)(sum q - q_u)); i ~ q/(sum q - q_u) over i != u, j uniform over n-2 others."""
    n = len(q)
    q2 = q.copy(); q2[u] = 0.0
    cq2 = np.cumsum(q2)
    Q2 = cq2[-1]
    i = np.minimum(np.searchsorted(cq2, rng.random(mp) * Q2, side="right"), n - 1)
    a = np.minimum(i, u); b = np.maximum(i, u)
    r = rng.integers(n - 2, size=mp)
    j = r + (r >= a)
    j = j + (j >= b)
    return i, j, (q2[i] + q2[j]) / ((n - 2) * Q2)


def pivot_proxies(D, u):
    n = D.shape[0]
    du = D[u].copy(); du[u] = 0.0
    Tu = du.sum()
    dbar = Tu / (n - 1)
    q = du + dbar
    q[u] = dbar
    return q, Tu


def adaptive_pivot(D, m, reps, rng):
    n = D.shape[0]
    N = n * (n - 1) / 2.0
    mp = m - (n - 1)
    assert mp >= 1, "adaptive pivot needs m >= n"
    outR = np.empty(reps); outZ = np.empty(reps); outA = np.empty(reps)
    for t in range(reps):
        u = int(rng.integers(n))
        q, Tu = pivot_proxies(D, u)
        # primary: restricted sampler + HH over non-pivot pairs + exact pivot total
        i, j, p = sample_restricted(q, u, mp, rng)
        outR[t] = (Tu + (D[i, j] / p).mean()) / N
        # variants on draws from the unrestricted sampler
        i, j, p = sample_full(q, mp, rng)
        w = D[i, j] / p
        outA[t] = w.mean() / N
        outZ[t] = (Tu + np.where((i != u) & (j != u), w, 0.0).mean()) / N
    return {"adaptive_pivot": outR, "adaptive_pivot_zf": outZ, "adaptive_pivot_hh": outA}


def check_sampler(n=7, draws=2_000_000, seed=1):
    """Empirical pair frequencies vs the closed-form p_e used in the estimator (chi-square)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3)) * np.exp(rng.standard_normal(n))[:, None]
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    u = 2
    q, Tu = pivot_proxies(D, u)
    for name, f in [("full", lambda: sample_full(q, draws, rng)), ("restricted", lambda: sample_restricted(q, u, draws, rng))]:
        i, j, p = f()
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        assert (i != j).all()
        if name == "restricted":
            assert ((i != u) & (j != u)).all()
        key = lo * n + hi
        cnt = np.bincount(key, minlength=n * n).astype(float)
        pe = np.zeros(n * n); pe[key] = p
        sup = pe > 0
        chi2 = float((((cnt[sup] - draws * pe[sup]) ** 2) / (draws * pe[sup])).sum())
        dof = int(sup.sum()) - 1
        print(f"sampler {name}: support={int(sup.sum())} sum_p={pe[sup].sum():.12f} chi2={chi2:.1f} dof={dof}", flush=True)


def run_one(job):
    mid, source, family, D, seed = job
    D = np.asarray(D, dtype=np.float64)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    rng = np.random.default_rng(5000 + seed)
    an = anatomy(D); n = an["n"]; mu = an["mu"]
    des = Designs(D, rng)
    rows = []
    for k in KS:
        if k >= n - 1:
            continue
        m = n * k // 2
        ests = {}
        ests["round_robin"] = des.round_robin(k, REPS)[0]
        ests["shuffled_circulant"] = des.shuffled_circulant(k, REPS)[0]
        ests["uniform"] = des.uniform(m, REPS)[0]
        if m >= n:
            ests.update(adaptive_pivot(D, m, REPS, rng))
        for name, est in ests.items():
            e = est / mu - 1.0
            sd = float(np.std(e, ddof=1))
            rows.append(dict(matrix=mid, source=source, family=family, n=n, k=k, m=m, design=name,
                             reps=REPS, rel_rmse=float(np.sqrt(np.mean(e ** 2))),
                             rel_bias=float(e.mean()), rel_sd=sd,
                             z_bias=float(e.mean() / (sd / np.sqrt(REPS))) if sd > 0 else 0.0,
                             mu=float(mu), kappa=float(an["kappa"])))
    return rows


if __name__ == "__main__":
    check_sampler()
    outcsv = sys.argv[1]
    jobs = []
    for fam in FAMILIES:
        for s in range(3):
            D = synthetic(fam, 200, np.random.default_rng(7 + 31 * s))
            jobs.append((f"{fam}_n200_s{s}", "synthetic", fam, D, s))
    for f in sys.argv[2:]:
        z = np.load(f, allow_pickle=True)
        if "D" in z.files:
            jobs.append((os.path.basename(f)[:-4], "real", "zoo128", z["D"], 0))
        else:
            for key in z.files:
                jobs.append((key, "real", "judge_" + key, z[key], 0))
    for j in jobs:
        D = np.asarray(j[3], dtype=np.float64)
        print(j[0], D.shape, "asym=%.2e" % np.abs(D - D.T).max(), "diag=%.2e" % np.abs(np.diag(D)).max(),
              "nan=%d" % int(np.isnan(D).sum()), "min_offdiag=%.3g" % D[np.triu_indices(D.shape[0], 1)].min(),
              flush=True)
    with Pool(int(os.environ.get("NPROC", 8))) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    rows = [r for rr in res for r in rr]
    with open(outcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows from {len(jobs)} matrices -> {outcsv}")
