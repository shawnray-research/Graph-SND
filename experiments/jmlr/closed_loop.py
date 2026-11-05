"""Closed-loop diversity control with sparse estimates (validates prop:dico).

A DiCo-style controller rescales agent-specific action components by
c_t = mu_des / mu_hat_t at every step while the components drift (as during
training). Behavioral distance = mean over a batch of observations of the
Euclidean distance between agent-specific action means (W2 between Gaussians
with shared covariance), which is homogeneous of degree one, so the realized
diversity is exactly mu_des * mu_t / mu_hat_t.
"""
import json
import math
import sys

import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "closed_loop.json"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
rng = np.random.default_rng(seed)
OBS, ACT, B, T = 8, 2, 32, 300


def round_robin_pairs(n, k, perm):
    # circle method: n even, rounds r = 0..n-2, each a perfect matching
    I, J = [], []
    for r in range(k):
        I.append(perm[r % (n - 1)])
        J.append(perm[n - 1])
        for t in range(1, n // 2):
            I.append(perm[(r + t) % (n - 1)])
            J.append(perm[(r - t) % (n - 1)])
    return np.array(I), np.array(J)


def anatomy_int(D):
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    mu = D[iu].mean()
    a = (D.sum(1) - (n - 1) * mu) / (n - 2)
    R = D - mu - a[:, None] - a[None, :]
    np.fill_diagonal(R, 0)
    s_tot = ((D[iu] - mu) ** 2).mean()
    s_int = (R[iu] ** 2).mean()
    return mu, s_tot, s_int


results = []
for n, k in [(50, 5), (500, 8)]:
    N = n * (n - 1) // 2
    m = n * k // 2
    iu = np.triu_indices(n, 1)
    for setting, sig_s in [("homogeneous", 0.0), ("heterogeneous", 0.7)]:
        W = rng.normal(size=(n, ACT, OBS)) / math.sqrt(OBS)
        bvec = rng.normal(size=(n, ACT)) * 0.1
        scale = rng.lognormal(0.0, sig_s, size=n) if sig_s > 0 else np.ones(n)
        obs = rng.normal(size=(B, OBS))
        track = {d: [] for d in ["full", "bernoulli_ht", "bernoulli_ratio", "uniform", "balanced"]}
        pred = {"uniform": [], "balanced": [], "bernoulli_ht": [], "bernoulli_ratio": []}
        for t in range(T):
            W += 0.02 * rng.normal(size=W.shape)
            H = scale[:, None, None] * np.tanh(np.einsum("nao,bo->nba", W, obs) + bvec[:, None, :])
            # pairwise mean Euclidean distance over the observation batch
            D = np.zeros((n, n))
            for i in range(n):
                D[i] = np.sqrt(((H[i][None] - H) ** 2).sum(-1)).mean(-1)
            mu, s_tot, s_int = anatomy_int(D)
            dvec = D[iu]
            ests = {"full": mu}
            keep = rng.random(N) < m / N
            ests["bernoulli_ht"] = dvec[keep].sum() / m  # HT: divide by p*N = m
            ests["bernoulli_ratio"] = dvec[keep].mean() if keep.any() else mu
            sel = rng.choice(N, m, replace=False)
            ests["uniform"] = dvec[sel].mean()
            I, J = round_robin_pairs(n, k, rng.permutation(n))
            ests["balanced"] = D[I, J].mean()
            for d, est in ests.items():
                realized = mu / est  # realized diversity / target (homogeneous degree one)
                track[d].append(realized - 1.0)
            pred["uniform"].append(math.sqrt(s_tot / m * (N - m) / (N - 1)) / mu)
            pred["balanced"].append(math.sqrt(s_int / m * (n - 1 - k) / (n - 3)) / mu)
            pred["bernoulli_ht"].append(math.sqrt((1 - m / N) / m * (mu ** 2 + s_tot)) / mu)
            pred["bernoulli_ratio"].append(math.sqrt(s_tot / m * (N - m) / (N - 1)) / mu)
        row = {"n": n, "k": k, "m": m, "setting": setting,
               "kappa_last": (s_tot - s_int) / s_int}
        for d in track:
            arr = np.array(track[d])
            row["rms_" + d] = float(np.sqrt((arr ** 2).mean()))
            row["mean_" + d] = float(arr.mean())
            row["trace_" + d] = [float(v) for v in arr[:150]]
        for d in pred:
            row["pred_" + d] = float(np.sqrt(np.mean(np.array(pred[d]) ** 2)))
        results.append(row)
        print({k_: (v if not isinstance(v, list) else "...") for k_, v in row.items()}, flush=True)
json.dump(results, open(out, "w"))
