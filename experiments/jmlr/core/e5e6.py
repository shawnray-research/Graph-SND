"""E5: design-based confidence intervals from round-robin rounds (coverage).
E6: recovering per-member distinctiveness a_i from O(nk) comparisons."""
import sys, os, glob, csv
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anatomy import *

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 7: 2.365, 15: 2.131, 31: 2.040}
REPS = 2000


def tq(df):
    if df in T975: return T975[df]
    return 1.96 + 2.4 / df  # adequate for df >= 8


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]


def one(job):
    mid, source, family, D = job
    rng = np.random.default_rng(abs(hash(mid)) % (2 ** 32))
    an = anatomy(D); n = an["n"]; mu = an["mu"]; a = an["a"]
    rounds = round_robin_rounds(n)
    iu = np.triu_indices(n, 1); d = D[iu]; N = len(d)
    rows = []
    for k in [2, 4, 8, 16]:
        if k >= n - 1: continue
        m = n * k // 2
        cov_rr = cov_u = 0; w_rr = []; w_u = []
        sp_rr = []; sp_u = []
        for t in range(REPS):
            perm = rng.permutation(n)
            sel = rng.choice(n - 1, size=k, replace=False)
            P = perm[rounds[sel]]                      # (k, n/2, 2)
            vals = D[P[..., 0], P[..., 1]]             # (k, n/2)
            ybar = vals.mean(1); T = ybar.mean()
            v = ybar.var(ddof=1) / k * (1 - k / (n - 1))
            h = tq(k - 1) * np.sqrt(v)
            cov_rr += (abs(T - mu) <= h); w_rr.append(2 * h / mu)
            # uniform pairs CI (SRSWOR of pairs)
            idx = rng.choice(N, size=m, replace=False)
            du = d[idx]; Tu = du.mean()
            vu = du.var(ddof=1) / m * (1 - m / N)
            hu = 1.96 * np.sqrt(vu)
            cov_u += (abs(Tu - mu) <= hu); w_u.append(2 * hu / mu)
            if t < 300:
                # distinctiveness from RR: each member has exactly k comparisons
                E = P.reshape(-1, 2); dv = D[E[:, 0], E[:, 1]]
                ends = np.r_[E[:, 0], E[:, 1]]
                s = np.bincount(ends, weights=np.r_[dv, dv], minlength=n)
                c = np.bincount(ends, minlength=n)
                ahat = (n - 1) / (n - 2) * (s / np.maximum(c, 1) - T)
                sp_rr.append(spearman(ahat, a))
                Eu = np.c_[iu[0][idx], iu[1][idx]]
                endu = np.r_[Eu[:, 0], Eu[:, 1]]
                su = np.bincount(endu, weights=np.r_[du, du], minlength=n)
                cu = np.bincount(endu, minlength=n)
                ahu = np.where(cu > 0, (n - 1) / (n - 2) * (su / np.maximum(cu, 1) - Tu), 0.0)
                sp_u.append(spearman(ahu, a))
        rows.append(dict(matrix=mid, source=source, family=family, n=n, k=k, m=m, kappa=an["kappa"],
                         cover_rr=cov_rr / REPS, cover_u=cov_u / REPS,
                         width_rr=float(np.median(w_rr)), width_u=float(np.median(w_u)),
                         spearman_rr=float(np.nanmean(sp_rr)), spearman_u=float(np.nanmean(sp_u))))
    return rows


if __name__ == "__main__":
    out = sys.argv[1]; pats = sys.argv[2:]
    jobs = []
    for fam in FAMILIES:
        D = synthetic(fam, 200, np.random.default_rng(7))
        jobs.append((f"{fam}_n200_s0", "synthetic", fam, D))
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            z = np.load(f, allow_pickle=True); D = z["D"].astype(np.float64)
            if D.shape[0] % 2 or D.shape[0] < 16: continue
            jobs.append((os.path.basename(f)[:-4], "real", f"{z['task']}_{z['kind']}", D))
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        res = pool.map(one, jobs, chunksize=1)
    rows = [r for rr in res for r in rr]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for r in rows:
        if r["k"] == 4:
            print(f"{r['matrix'][:40]:40s} k=4 cover RR {r['cover_rr']:.3f} U {r['cover_u']:.3f} | width RR {100*r['width_rr']:.2f}% U {100*r['width_u']:.2f}% | spearman RR {r['spearman_rr']:.3f} U {r['spearman_u']:.3f}")
