"""Tests for balanced comparison designs (exact laws checked numerically)."""

import itertools
import math

import numpy as np

from graphsnd.designs import (
    anatomy,
    balanced_design,
    balanced_variance,
    design_variance,
    estimate_mean_dissimilarity,
    median_of_designs,
    metric_relvar_bound,
    round_robin_estimate,
    separable_mean,
    sparse_distinctiveness,
)


def _metric(n, seed=0, dim=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)) * rng.lognormal(0, 1, size=(n, 1))
    return np.linalg.norm(X[:, None] - X[None], axis=-1)


def test_anatomy_orthogonal_and_canonical():
    D = _metric(9)
    an = anatomy(D)
    n = D.shape[0]
    assert abs(an.a.sum()) < 1e-9
    assert np.allclose(an.R.sum(axis=1), 0.0, atol=1e-9)
    assert math.isclose(an.sigma2_tot, an.sigma2_main + an.sigma2_int, rel_tol=1e-9)
    recon = an.mu + an.a[:, None] + an.a[None, :] + an.R
    np.fill_diagonal(recon, 0.0)
    assert np.allclose(recon, D)
    assert an.sigma2_int <= n / (n - 2) * an.mu ** 2  # universal metric bound


def test_designs_are_simple_and_regular():
    for n, k in [(10, 1), (10, 3), (11, 2), (12, 5), (12, 11)]:
        for kind in ("circulant", "round_robin"):
            if kind == "round_robin" and n % 2:
                continue
            E = balanced_design(n, k, rng=1, kind=kind)
            assert E.shape == (n * k // 2, 2)
            assert np.all(E[:, 0] != E[:, 1])
            assert len({tuple(sorted(e)) for e in E.tolist()}) == len(E)
            deg = np.bincount(E.ravel(), minlength=n)
            assert np.all(deg == k)


def test_exact_unbiasedness_and_variance_by_enumeration():
    n, k = 6, 2
    D = _metric(n, seed=3)
    an = anatomy(D)
    from graphsnd.designs import _circulant_edges

    H = _circulant_edges(n, k)
    vals = []
    for perm in itertools.permutations(range(n)):
        E = np.asarray(perm)[H]
        vals.append(D[E[:, 0], E[:, 1]].mean())
    vals = np.asarray(vals)
    assert math.isclose(vals.mean(), an.mu, rel_tol=1e-12)
    assert math.isclose(vals.var(), balanced_variance(an.sigma2_int, n, k), rel_tol=1e-9)


def test_additive_tables_are_estimated_exactly():
    n = 40
    b = np.zeros(n)
    b[7] = 50.0
    D = 1.0 + b[:, None] + b[None, :]
    np.fill_diagonal(D, 0.0)
    an = anatomy(D)
    est, _, _ = estimate_mean_dissimilarity(D, n, k=1, rng=0)
    assert math.isclose(est, an.mu, rel_tol=1e-12)


def test_metric_bound_and_median():
    n = 64
    D = _metric(n, seed=5)
    an = anatomy(D)
    rng = np.random.default_rng(0)
    ests = np.asarray([estimate_mean_dissimilarity(D, n, 2, rng)[0] for _ in range(4000)])
    relvar = ests.var() / an.mu ** 2
    assert relvar <= metric_relvar_bound(n, 2) * 1.1
    est = median_of_designs(lambda i, j: D[i, j], n, eps=0.1, delta=0.01, rng=1)
    assert abs(est / an.mu - 1) <= 0.1


def test_sparse_distinctiveness_unbiased():
    n, k = 12, 4
    D = _metric(n, seed=7)
    an = anatomy(D)
    rng = np.random.default_rng(2)
    acc = np.zeros(n)
    reps = 4000
    for _ in range(reps):
        _, E, v = estimate_mean_dissimilarity(D, n, k, rng)
        acc += sparse_distinctiveness(E, v, n)
    assert np.allclose(acc / reps, an.a, atol=0.15 * np.abs(an.a).max())


def test_separable_means_are_exact():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 6))
    iu = np.triu_indices(30, 1)
    D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    D1 = np.abs(X[:, None] - X[None]).sum(-1)
    assert math.isclose(separable_mean(X, "sqeuclidean"), D2[iu].mean(), rel_tol=1e-10)
    assert math.isclose(separable_mean(X, "l1"), D1[iu].mean(), rel_tol=1e-10)


def test_general_shape_variance_by_enumeration():
    n = 6
    D = _metric(n, seed=11)
    an = anatomy(D)
    for E in ([(0, j) for j in range(1, n)], [(i, i + 1) for i in range(n - 1)],
              [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (1, 2)]):
        E = np.asarray(E)
        vals = []
        for perm in itertools.permutations(range(n)):
            e = np.asarray(perm)[E]
            vals.append(D[e[:, 0], e[:, 1]].mean())
        vals = np.asarray(vals)
        assert math.isclose(vals.mean(), an.mu, rel_tol=1e-12)
        assert math.isclose(vals.var(), design_variance(E, n, an), rel_tol=1e-9)


def test_round_robin_variance_estimate_unbiased():
    n, k = 10, 3
    D = _metric(n, seed=13)
    rng = np.random.default_rng(5)
    out = np.asarray([round_robin_estimate(D, n, k, rng) for _ in range(20000)])
    T, V = out[:, 0], out[:, 1]
    assert abs(T.mean() / anatomy(D).mu - 1) < 0.01
    assert abs(V.mean() / T.var() - 1) < 0.05
