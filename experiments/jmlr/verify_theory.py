"""Numerical verification of the anatomy / balanced-design theory.

Checks (each prints PASS/FAIL with the max discrepancy):
  1. anatomy: orthogonality, energy identity, zero row sums of R
  2. exact error identity for arbitrary weighted graphs
  3. randomized k-regular design: unbiasedness + exact variance formula
  4. uniform pairs without replacement: exact variance formula
  5. efficiency law Var_U / Var_B = (1+kappa) n(n-3)/((n-2)(n+1))
  6. metric bound s^2 <= n/(n-2) mu^2 on adversarial metrics
  7. fixed d-regular spectral bound vs realised error
"""
import itertools
import math
import sys

import networkx as nx
import numpy as np

rng = np.random.default_rng(0)


def anatomy(D):
    n = D.shape[0]
    N = n * (n - 1) / 2
    iu = np.triu_indices(n, 1)
    mu = D[iu].mean()
    m = D.sum(1) / (n - 1)
    a = (n - 1) / (n - 2) * (m - mu)
    R = D - mu - a[:, None] - a[None, :]
    np.fill_diagonal(R, 0.0)
    return mu, a, R


def energies(D):
    n = D.shape[0]
    N = n * (n - 1) / 2
    mu, a, R = anatomy(D)
    iu = np.triu_indices(n, 1)
    tot = ((D[iu] - mu) ** 2).sum()
    main = (n - 2) * (a ** 2).sum()
    inter = (R[iu] ** 2).sum()
    return mu, tot / N, main / N, inter / N


def random_metric(n, kind):
    if kind == "gauss":
        X = rng.normal(size=(n, 16))
    elif kind == "scale":
        X = rng.normal(size=(n, 16)) * rng.lognormal(0, 0.8, size=(n, 1))
    elif kind == "roles":
        X = rng.normal(size=(n, 16)) * 0.2
        X[: n // 2, 0] += 3.0
    elif kind == "outlier":
        X = rng.normal(size=(n, 16)) * 0.05
        X[0] += 10.0
    elif kind == "ring":
        t = rng.uniform(0, 2 * np.pi, n)
        X = np.c_[np.cos(t), np.sin(t)]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    return D


def check(name, ok, info):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
    return ok


allok = True
# 1. anatomy
for kind in ["gauss", "scale", "roles", "outlier", "ring"]:
    D = random_metric(40, kind)
    n = D.shape[0]
    mu, a, R = anatomy(D)
    iu = np.triu_indices(n, 1)
    M = a[:, None] + a[None, :]
    C = np.full_like(D, mu)
    o1 = abs((C[iu] * M[iu]).sum()) + abs((C[iu] * R[iu]).sum()) + abs((M[iu] * R[iu]).sum())
    rs = np.abs(R.sum(1)).max()
    mu2, tot, main, inter = energies(D)
    allok &= check(f"anatomy[{kind}]", o1 < 1e-8 and rs < 1e-10 and abs(tot - main - inter) < 1e-10 * max(tot, 1),
                   f"orth={o1:.1e} rowsum={rs:.1e} energy_gap={abs(tot-main-inter):.1e} shares M={main/tot:.3f} R={inter/tot:.3f}")

# 2. exact error identity on arbitrary weighted graphs
D = random_metric(30, "scale")
n = 30
mu, a, R = anatomy(D)
worst = 0
for trial in range(200):
    p = rng.uniform(0.05, 0.5)
    E = [(i, j) for i, j in itertools.combinations(range(n), 2) if rng.random() < p]
    if not E:
        continue
    w = rng.uniform(0.1, 2.0, len(E))
    W = w.sum()
    est = sum(wk * D[i, j] for wk, (i, j) in zip(w, E)) / W
    pi = np.zeros(n)
    for wk, (i, j) in zip(w, E):
        pi[i] += wk
        pi[j] += wk
    pi /= 2 * W
    pred = mu + ((2 * pi - 2.0 / n) * a).sum() + sum(wk * R[i, j] for wk, (i, j) in zip(w, E)) / W
    worst = max(worst, abs(est - pred))
allok &= check("error identity (weighted, 200 graphs)", worst < 1e-10, f"max |est-pred|={worst:.1e}")


# 3./4./5. randomized k-regular vs uniform pairs, exact variances
def var_balanced_formula(D, k):
    n = D.shape[0]
    mu, tot, main, inter = energies(D)
    m = n * k / 2
    return inter / m * (n - 1 - k) / (n - 3)


def var_uniform_formula(D, m):
    n = D.shape[0]
    N = n * (n - 1) / 2
    mu, tot, main, inter = energies(D)
    return tot / m * (N - m) / (N - 1)


def circulant_edges(n, k):
    E = []
    for i in range(n):
        for s in range(1, k // 2 + 1):
            E.append((i, (i + s) % n))
    return E


for kind in ["gauss", "scale", "roles", "outlier"]:
    n, k = 24, 4
    D = random_metric(n, kind)
    mu, tot, main, inter = energies(D)
    Ebase = circulant_edges(n, k)
    m = len(Ebase)
    ests_b, ests_u = [], []
    iu = np.triu_indices(n, 1)
    allpairs = np.c_[iu]
    for t in range(20000):
        perm = rng.permutation(n)
        ests_b.append(np.mean([D[perm[i], perm[j]] for i, j in Ebase]))
        sel = rng.choice(len(iu[0]), m, replace=False)
        ests_u.append(D[iu[0][sel], iu[1][sel]].mean())
    ests_b, ests_u = np.array(ests_b), np.array(ests_u)
    vb, vu = ests_b.var(), ests_u.var()
    fb, fu = var_balanced_formula(D, k), var_uniform_formula(D, m)
    kappa = main / inter
    law = (1 + kappa) * n * (n - 3) / ((n - 2) * (n + 1))
    ok = abs(ests_b.mean() - mu) < 4 * math.sqrt(fb / 20000) + 1e-12 and abs(vb / fb - 1) < 0.05 and abs(vu / fu - 1) < 0.05
    allok &= check(f"balanced/uniform variance[{kind}]", ok,
                   f"bias_b={ests_b.mean()-mu:.2e} Vb emp/form={vb:.3e}/{fb:.3e} Vu emp/form={vu:.3e}/{fu:.3e} "
                   f"ratio emp={vu/vb:.2f} law={law:.2f}")

# 6. metric bound: s^2 <= n/(n-2) mu^2 ; try adversarial metrics
worst = 0
for t in range(300):
    n = int(rng.integers(5, 60))
    kind = rng.choice(["gauss", "scale", "roles", "outlier", "ring"])
    D = random_metric(n, kind)
    if rng.random() < 0.3:  # random graph shortest-path metric
        G = nx.gnp_random_graph(n, 0.2, seed=int(rng.integers(1e9)))
        if not nx.is_connected(G):
            continue
        for u, v in G.edges:
            G[u][v]["w"] = rng.lognormal(0, 1.5)
        L = dict(nx.all_pairs_dijkstra_path_length(G, weight="w"))
        D = np.array([[L[i][j] for j in range(n)] for i in range(n)])
    mu, tot, main, inter = energies(D)
    worst = max(worst, inter / (n / (n - 2) * mu ** 2))
allok &= check("metric bound s^2 <= n/(n-2) mu^2 (300 metrics)", worst <= 1 + 1e-9, f"max ratio={worst:.3f}")
# star metric (one outlier) is near-tight for main effects; equilateral-two-clusters for R
# 7. fixed d-regular spectral bound
for kind in ["gauss", "scale", "roles", "ring"]:
    n, d = 200, 8
    D = random_metric(n, kind)
    mu, a, R = anatomy(D)
    G = nx.random_regular_graph(d, n, seed=1)
    A = nx.to_numpy_array(G)
    ev = np.sort(np.linalg.eigvalsh(A))
    lam2, lamn = ev[-2], ev[0]
    est = np.mean([D[i, j] for i, j in G.edges])
    nuc = np.linalg.svd(R, compute_uv=False).sum()
    bound = (lam2 - lamn) / (2 * d) * nuc / (n * mu)
    old = (max(abs(lam2), abs(lamn)) + d / (n - 1)) / d * np.linalg.svd(D, compute_uv=False).sum() / (n * mu)
    allok &= check(f"spectral bound[{kind}]", abs(est / mu - 1) <= bound + 1e-12,
                   f"realised={abs(est/mu-1):.2e} new_bound={bound:.2e} old_bound={old:.2e}")

print("ALL PASS" if allok else "SOME FAILED")
sys.exit(0 if allok else 1)
