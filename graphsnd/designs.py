"""Balanced comparison designs for mean pairwise dissimilarity.

Estimating a mean pairwise dissimilarity (System Neural Diversity, Self-BLEU,
ensemble disagreement, nucleotide diversity, ...) needs only linearly many
comparisons when the compared pairs form a randomly relabeled regular graph:

* :func:`anatomy` splits a dissimilarity table into its level ``mu`` (the mean
  pairwise dissimilarity), per-agent distinctiveness ``a`` and relational
  residual ``R`` (Theorem "anatomy of diversity").
* :func:`balanced_design` draws a uniformly relabeled ``k``-regular design with
  ``m = n k / 2`` pairs. The design mean is exactly unbiased, distinctiveness
  cancels from its error, and its variance is
  ``sigma2_int / m * (n - 1 - k) / (n - 3)`` (:func:`balanced_variance`).
* :func:`estimate_mean_dissimilarity` evaluates a user-supplied dissimilarity
  only on the design pairs; :func:`median_of_designs` gives a
  ``(1 +- eps)`` guarantee for every metric with ``O((n + eps^-2) log(1/delta))``
  evaluations.
* :func:`separable_mean` computes the exact all-pairs mean in linear time for
  squared-Euclidean-type and L1-type dissimilarities, where no sampling is needed.

All functions are NumPy-only; torch tensors are accepted via ``np.asarray``.

Example::

    from graphsnd.designs import anatomy, estimate_mean_dissimilarity
    # d(i, j): any (expensive) dissimilarity between population members i and j
    est, pairs, vals = estimate_mean_dissimilarity(d, n, k=2, rng=0)  # n*k/2 calls
    an = anatomy(D)  # if a full table D is available: an.mu, an.a, an.R, an.kappa
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np

RngLike = Union[None, int, np.random.Generator]


def _rng(rng: RngLike) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


@dataclass(frozen=True)
class Anatomy:
    """Canonical decomposition ``d_ij = mu + a_i + a_j + r_ij`` (i != j)."""

    mu: float               # level: the mean pairwise dissimilarity (SND)
    a: np.ndarray           # distinctiveness, sums to zero
    R: np.ndarray           # relational residual, zero diagonal and zero row sums
    sigma2_tot: float       # pairwise variance of d
    sigma2_main: float      # variance carried by distinctiveness
    sigma2_int: float       # variance carried by relational structure

    @property
    def kappa(self) -> float:
        if self.sigma2_int == 0:
            return 0.0 if self.sigma2_main == 0 else math.inf
        return self.sigma2_main / self.sigma2_int

    @property
    def relational_share(self) -> float:
        return 0.0 if self.sigma2_tot == 0 else self.sigma2_int / self.sigma2_tot


def anatomy(D) -> Anatomy:
    """Anatomy of a symmetric dissimilarity table with zero diagonal (n >= 3)."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n) or n < 3:
        raise ValueError("D must be a square matrix with n >= 3")
    if not np.allclose(D, D.T):
        raise ValueError("D must be symmetric")
    D = D.copy()
    np.fill_diagonal(D, 0.0)
    N = n * (n - 1) / 2
    iu = np.triu_indices(n, 1)
    mu = float(D[iu].mean())
    S = D.sum(axis=1)
    a = (S - (n - 1) * mu) / (n - 2)
    R = D - mu - a[:, None] - a[None, :]
    np.fill_diagonal(R, 0.0)
    sigma2_tot = float(((D[iu] - mu) ** 2).sum() / N)
    sigma2_main = float((n - 2) * (a ** 2).sum() / N)
    sigma2_int = float((R[iu] ** 2).sum() / N)
    return Anatomy(mu, a, R, sigma2_tot, sigma2_main, sigma2_int)


def _circulant_edges(n: int, k: int) -> np.ndarray:
    """Simple k-regular circulant on Z_n (offsets 1..k//2, plus n/2 if k is odd)."""
    if not 1 <= k <= n - 1:
        raise ValueError("need 1 <= k <= n - 1")
    if (n * k) % 2:
        raise ValueError("a k-regular graph on n vertices needs n*k even")
    i = np.arange(n)
    blocks = [np.stack([i, (i + h) % n], axis=1) for h in range(1, k // 2 + 1)]
    if k % 2:
        half = np.arange(n // 2)
        blocks.append(np.stack([half, half + n // 2], axis=1))
    return np.concatenate(blocks, axis=0)


def _round_robin_edges(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """k distinct random rounds of the circle-method 1-factorization of K_n (n even)."""
    if n % 2:
        raise ValueError("round-robin designs need n even")
    rounds = rng.choice(n - 1, size=k, replace=False)
    out = []
    for r in rounds:
        pairs = [(n - 1, r)]
        for t in range(1, n // 2):
            pairs.append(((r + t) % (n - 1), (r - t) % (n - 1)))
        out.append(np.asarray(pairs))
    return np.concatenate(out, axis=0)


def balanced_design(n: int, k: int, rng: RngLike = None, kind: str = "circulant") -> np.ndarray:
    """Uniformly relabeled simple k-regular design; returns an (n*k/2, 2) int array.

    Every agent is compared exactly k times. The mean and variance of the design
    error depend only on (n, k), so ``kind`` ("circulant" or "round_robin") does
    not change them; both cost O(n k).
    """
    g = _rng(rng)
    if kind == "circulant":
        E = _circulant_edges(n, k)
    elif kind == "round_robin":
        E = _round_robin_edges(n, k, g)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    perm = g.permutation(n)
    return perm[E]


def balanced_variance(sigma2_int: float, n: int, k: int) -> float:
    """Exact variance of the balanced-design mean (Theorem 'randomized balanced designs')."""
    m = n * k / 2
    return sigma2_int / m * (n - 1 - k) / (n - 3)


def design_variance(edges: np.ndarray, n: int, an: "Anatomy") -> float:
    """Exact variance of a uniformly relabeled design of ANY shape.

    Depends on the design only through m = #pairs and Q = sum of squared degrees:
    Var = (nQ - 4m^2) / (2(n-2)m^2) * sigma2_main + B(m, Q) / m^2 * sigma2_int.
    Regular designs (Q = 4m^2/n) minimize it iff kappa >= 2(n-1)/(n(n-3)).
    """
    edges = np.asarray(edges)
    m = len(edges)
    deg = np.bincount(edges.ravel(), minlength=n)
    Q = float((deg.astype(float) ** 2).sum())
    B = m - (Q - 2 * m) / (n - 2) + 2 * (m * m + m - Q) / ((n - 2) * (n - 3))
    return (n * Q - 4 * m * m) / (2 * (n - 2) * m * m) * an.sigma2_main + B / (m * m) * an.sigma2_int


def metric_relvar_bound(n: int, k: int) -> float:
    """Universal bound v_{n,k} on Var(T)/mu^2 for every pseudometric table.

    Uses sigma2_int <= (n-2)/n * mu^2, which is attained by the balanced cut metric.
    """
    return 2.0 * (n - 2) * (n - 1 - k) / (n * n * (n - 3) * k)


def degree_for_accuracy(n: int, eps: float) -> int:
    """Smallest admissible degree with failure probability <= 1/4 at relative error eps (n >= 4)."""
    k = min(n - 1, max(1, math.ceil(8.0 / (n * eps * eps))))
    if (n * k) % 2:
        k = k + 1 if k + 1 <= n - 1 else k - 1
    return k


def _evaluate(dissim, edges: np.ndarray) -> np.ndarray:
    if callable(dissim):
        try:
            vals = np.asarray(dissim(edges[:, 0], edges[:, 1]), dtype=float)
            if vals.shape == (len(edges),):
                return vals
        except Exception:
            pass
        return np.asarray([dissim(int(i), int(j)) for i, j in edges], dtype=float)
    D = np.asarray(dissim, dtype=float)
    return D[edges[:, 0], edges[:, 1]]


def estimate_mean_dissimilarity(
    dissim: Union[Callable, np.ndarray],
    n: int,
    k: int = 2,
    rng: RngLike = None,
    kind: str = "circulant",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Unbiased balanced-design estimate of the mean pairwise dissimilarity.

    ``dissim`` is either an (n, n) table or a callable ``f(i, j)`` (scalar or
    vectorized over index arrays). Only the n*k/2 design pairs are evaluated.
    Returns ``(estimate, edges, values)``.
    """
    edges = balanced_design(n, k, rng, kind)
    vals = _evaluate(dissim, edges)
    return float(vals.mean()), edges, vals


def round_robin_estimate(
    dissim: Union[Callable, np.ndarray],
    n: int,
    k: int = 4,
    rng: RngLike = None,
) -> Tuple[float, float]:
    """Round-robin balanced estimate with an exactly unbiased variance estimate.

    Uses k >= 2 random rounds of a relabeled 1-factorization (n even). Returns
    ``(T, V_hat)`` with E[T] = mean pairwise dissimilarity and
    E[V_hat] = Var(T), where V_hat = (1 - k/(n-1)) * s^2 / k and s^2 is the
    sample variance of the k round means.
    """
    if n % 2 or not 2 <= k <= n - 1:
        raise ValueError("need n even and 2 <= k <= n - 1")
    g = _rng(rng)
    edges = g.permutation(n)[_round_robin_edges(n, k, g)]
    vals = _evaluate(dissim, edges).reshape(k, n // 2)
    means = vals.mean(axis=1)
    T = float(means.mean())
    V = float((1.0 - k / (n - 1)) * means.var(ddof=1) / k)
    return T, V


def median_of_designs(
    dissim: Union[Callable, np.ndarray],
    n: int,
    eps: float,
    delta: float,
    rng: RngLike = None,
) -> float:
    """(1 +- eps)-approximation with probability >= 1 - delta for every pseudometric (n >= 8)."""
    g = _rng(rng)
    k = degree_for_accuracy(n, eps)
    L = max(1, math.ceil(8.0 * math.log(1.0 / delta)))
    ests = [estimate_mean_dissimilarity(dissim, n, k, g)[0] for _ in range(L)]
    return float(np.median(ests))


def sparse_distinctiveness(edges: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Distinctiveness estimates a_hat_i from a balanced design's evaluated pairs."""
    edges = np.asarray(edges)
    values = np.asarray(values, dtype=float)
    T = values.mean()
    sums = np.bincount(edges[:, 0], weights=values, minlength=n) + np.bincount(edges[:, 1], weights=values, minlength=n)
    deg = np.bincount(edges[:, 0], minlength=n) + np.bincount(edges[:, 1], minlength=n)
    m_hat = sums / np.maximum(deg, 1)
    return (n - 1) * (m_hat - T) / (n - 2)


def separable_mean(X, kind: str = "sqeuclidean") -> float:
    """Exact all-pairs mean dissimilarity in O(n p) ('sqeuclidean') or O(p n log n) ('l1').

    ``X`` is an (n, p) feature matrix. For total variation between categorical
    distributions, pass the probability vectors with kind='l1' and halve the result.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    N = n * (n - 1) / 2
    if kind == "sqeuclidean":
        sq = (X ** 2).sum()
        tot = n * sq - (X.sum(axis=0) ** 2).sum()
        return float(tot / N)
    if kind == "l1":
        Xs = np.sort(X, axis=0)
        coef = 2 * np.arange(1, n + 1) - n - 1
        return float((coef[:, None] * Xs).sum() / N)
    raise ValueError(f"unknown kind {kind!r}")
