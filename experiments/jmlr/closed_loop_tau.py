"""Closed-loop diversity control with a soft-updated estimate (DiCo's running average).

Same drifting populations as closed_loop.py (Section 5.8, simulated control), but the
controller rescales with c_t = mu_des / s_t, where s_t = (1 - tau) s_{t-1} + tau * est_t is
DiCo's soft update (tau = 0.01 by default in DiCo; closed_loop.py is tau = 1). The realized
diversity relative to the target is mu_t / s_t, so the tracking error mixes the averaged
estimation noise of the design with the lag of the running average behind the drifting mu_t.
All designs and all tau values share one population trajectory per seed.

Usage: python closed_loop_tau.py out.json [n_pops] [n_workers] [team sizes, e.g. 50 or 50,500]
"""
import json
import math
import sys

import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "closed_loop_tau.json"
NPOP = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OBS, ACT, B, T, BURN = 8, 2, 32, 300, 100
TAUS = [1.0, 0.3, 0.1, 0.03, 0.01]
DESIGNS = ["full", "uniform", "balanced"]


def round_robin_pairs(n, k, perm):  # identical to closed_loop.py
    I, J = [], []
    for r in range(k):
        I.append(perm[r % (n - 1)])
        J.append(perm[n - 1])
        for t in range(1, n // 2):
            I.append(perm[(r + t) % (n - 1)])
            J.append(perm[(r - t) % (n - 1)])
    return np.array(I), np.array(J)


def run_pop(args):
    n, k, setting, sig_s, pop = args
    N = n * (n - 1) // 2
    m = n * k // 2
    iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(1000 * n + 100 * (sig_s > 0) + pop)
    W = rng.normal(size=(n, ACT, OBS)) / math.sqrt(OBS)
    bvec = rng.normal(size=(n, ACT)) * 0.1
    scale = rng.lognormal(0.0, sig_s, size=n) if sig_s > 0 else np.ones(n)
    obs = rng.normal(size=(B, OBS))
    s = {(d, tau): None for d in DESIGNS for tau in TAUS}
    err = {(d, tau): [] for d in DESIGNS for tau in TAUS}
    for t in range(T):
        W += 0.02 * rng.normal(size=W.shape)
        H = scale[:, None, None] * np.tanh(np.einsum("nao,bo->nba", W, obs) + bvec[:, None, :])
        D = np.zeros((n, n))
        for i in range(n):
            D[i] = np.sqrt(((H[i][None] - H) ** 2).sum(-1)).mean(-1)
        dvec = D[iu]
        mu = dvec.mean()
        I, J = round_robin_pairs(n, k, rng.permutation(n))
        est = {"full": mu, "uniform": dvec[rng.choice(N, m, replace=False)].mean(), "balanced": D[I, J].mean()}
        for d in DESIGNS:
            for tau in TAUS:
                prev = s[(d, tau)]
                s[(d, tau)] = est[d] if prev is None else (1 - tau) * prev + tau * est[d]
                if t >= BURN:
                    err[(d, tau)].append(mu / s[(d, tau)] - 1.0)
    return (n, setting, pop), {f"{d}|{tau}": float(np.sqrt(np.mean(np.square(err[(d, tau)])))) for d in DESIGNS for tau in TAUS}


if __name__ == "__main__":
    from multiprocessing import Pool
    SIZES = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [50, 500]
    NK = [(n, k) for n, k in [(50, 5), (500, 8)] if n in SIZES]
    tasks = [(n, k, setting, sig_s, pop) for n, k in NK
             for setting, sig_s in [("homogeneous", 0.0), ("heterogeneous", 0.7)] for pop in range(NPOP)]
    tasks.sort(key=lambda a: -a[0])  # long (n = 500) tasks first
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        out_rows = dict(pool.map(run_pop, tasks))
    results = []
    for n, k in NK:
        for setting in ["homogeneous", "heterogeneous"]:
            pops = [out_rows[(n, setting, p)] for p in range(NPOP)]
            for d in DESIGNS:
                for tau in TAUS:
                    v = np.array([r[f"{d}|{tau}"] for r in pops])
                    results.append(dict(n=n, k=k, m=n * k // 2, setting=setting, design=d, tau=tau,
                                        rms_mean=float(v.mean()), rms_se=float(v.std(ddof=1) / math.sqrt(len(v))),
                                        rms_per_pop=v.tolist()))
            print(f"n={n} {setting}: " + " | ".join(
                f"tau={tau}: " + " ".join(f"{d[:4]} {100*np.mean([r[f'{d}|{tau}'] for r in pops]):.3f}" for d in DESIGNS)
                for tau in TAUS), flush=True)
    json.dump(results, open(out, "w"), indent=0)
