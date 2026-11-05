"""Anatomy of a pairwise dissimilarity table and comparison designs.

d_ij = mu + a_i + a_j + r_ij  (i != j),  sum_i a_i = 0,  sum_{j != i} r_ij = 0.
mu: level (= SND for behavioral distances); a: per-member distinctiveness;
r: relational structure. Orthogonal on the pair set:
sigma2_tot = sigma2_main + sigma2_int.
"""
import numpy as np


def anatomy(D):
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    N = n * (n - 1) / 2
    iu = np.triu_indices(n, 1)
    mu = D[iu].mean()
    m = D.sum(1) / (n - 1)
    a = (n - 1) * (m - mu) / (n - 2)
    R = D - mu - a[:, None] - a[None, :]
    np.fill_diagonal(R, 0.0)
    d = D[iu]
    s2_tot = np.mean((d - mu) ** 2)
    s2_main = np.mean((a[iu[0]] + a[iu[1]]) ** 2)
    s2_int = np.mean(R[iu] ** 2)
    return dict(n=n, N=N, mu=mu, a=a, R=R, s2_tot=s2_tot, s2_main=s2_main, s2_int=s2_int,
                kappa=s2_main / s2_int if s2_int > 0 else np.inf)


# ---------------------------------------------------------------- exact laws
def var_uniform(an, m):
    N = an["N"]
    return an["s2_tot"] / m * (N - m) / (N - 1)


def var_bernoulli_ht(D, p):
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    d = D[iu]
    N = len(d)
    return (1 - p) / p * np.sum(d ** 2) / N ** 2


def var_balanced(an, k):
    """Any simple k-regular design under uniform relabeling (round-robin is one)."""
    n = an["n"]
    m = n * k / 2
    return an["s2_int"] / m * (n - 1 - k) / (n - 3)


def efficiency_law(an):
    n = an["n"]
    return (1 + an["kappa"]) * n * (n - 3) / ((n - 2) * (n + 1))


# ---------------------------------------------------------------- designs
def round_robin_rounds(n):
    """Circle-method 1-factorization of K_n (n even): array (n-1, n/2, 2)."""
    assert n % 2 == 0
    rounds = np.zeros((n - 1, n // 2, 2), dtype=np.int64)
    for r in range(n - 1):
        rounds[r, 0] = (n - 1, r)
        for k in range(1, n // 2):
            rounds[r, k] = ((r + k) % (n - 1), (r - k) % (n - 1))
    return rounds


def circulant_offsets(n, k):
    offs = list(range(1, k // 2 + 1))
    antipodal = (k % 2 == 1)
    return offs, antipodal


def circulant_edges(n, k):
    offs, antipodal = circulant_offsets(n, k)
    E = []
    for o in offs:
        for i in range(n):
            E.append((i, (i + o) % n))
    if antipodal:
        for i in range(n // 2):
            E.append((i, i + n // 2))
    E = np.array(E, dtype=np.int64)
    return E


class Designs:
    """Monte-Carlo samplers returning estimates of mu = mean_{i<j} d_ij."""

    def __init__(self, D, rng):
        self.D = np.asarray(D, dtype=np.float64)
        self.n = self.D.shape[0]
        self.iu = np.triu_indices(self.n, 1)
        self.d = self.D[self.iu]
        self.N = len(self.d)
        self.rng = rng
        self.rounds = round_robin_rounds(self.n) if self.n % 2 == 0 else None

    # randomized, unbiased, k-regular (ours)
    def round_robin(self, k, reps):
        n, D, rng = self.n, self.D, self.rng
        out = np.empty(reps)
        for t in range(reps):
            perm = rng.permutation(n)
            sel = rng.choice(n - 1, size=k, replace=False)
            P = perm[self.rounds[sel].reshape(-1, 2)]
            out[t] = D[P[:, 0], P[:, 1]].mean()
        return out, n * k // 2

    def shuffled_circulant(self, k, reps):
        n, D, rng = self.n, self.D, self.rng
        E = circulant_edges(n, k)
        out = np.empty(reps)
        for t in range(reps):
            perm = rng.permutation(n)
            P = perm[E]
            out[t] = D[P[:, 0], P[:, 1]].mean()
        return out, len(E)

    def uniform(self, m, reps):
        out = np.empty(reps)
        for t in range(reps):
            idx = self.rng.choice(self.N, size=m, replace=False)
            out[t] = self.d[idx].mean()
        return out, m

    def bernoulli_ht(self, m, reps):
        p = m / self.N
        out = np.empty(reps)
        for t in range(reps):
            mask = self.rng.random(self.N) < p
            out[t] = self.d[mask].sum() / (p * self.N)
        return out, m

    def bernoulli_ratio(self, m, reps):
        p = m / self.N
        out = np.empty(reps)
        for t in range(reps):
            mask = self.rng.random(self.N) < p
            out[t] = self.d[mask].mean() if mask.any() else np.nan
        return out, m

    def agent_subsample(self, m, reps):
        s = int(round((1 + np.sqrt(1 + 8 * m)) / 2))
        s = max(2, min(self.n, s))
        out = np.empty(reps)
        for t in range(reps):
            S = self.rng.choice(self.n, size=s, replace=False)
            sub = self.D[np.ix_(S, S)]
            out[t] = sub[np.triu_indices(s, 1)].mean()
        return out, s * (s - 1) // 2

    def landmark(self, m, reps):
        n = self.n
        k = max(1, int(round(m / (n - 1))))
        out = np.empty(reps)
        mm = k * (n - 1) - k * (k - 1) // 2
        for t in range(reps):
            L = self.rng.choice(n, size=k, replace=False)
            mask = np.zeros((n, n), dtype=bool)
            mask[L, :] = True
            mask[:, L] = True
            np.fill_diagonal(mask, False)
            out[t] = self.D[np.triu(mask, 1)].mean()
        return out, mm

    # fixed (deterministic) designs: return a single value
    def local_knn(self, k):
        n, D = self.n, self.D
        Dn = D + np.diag(np.full(n, np.inf))
        nbr = np.argsort(Dn, axis=1)[:, :k]
        mask = np.zeros((n, n), dtype=bool)
        mask[np.repeat(np.arange(n), k), nbr.ravel()] = True
        mask = mask | mask.T
        np.fill_diagonal(mask, False)
        sel = np.triu(mask, 1)
        return D[sel].mean(), int(sel.sum())

    def ring_fixed(self, k):
        E = circulant_edges(self.n, k)
        return self.D[E[:, 0], E[:, 1]].mean(), len(E)

    def expander_fixed(self, k, seed=0):
        E = random_regular_edges(self.n, k, np.random.default_rng(seed))
        return self.D[E[:, 0], E[:, 1]].mean(), len(E)


# ---------------------------------------------------------------- synthetic families
def synthetic(family, n, rng):
    if family == "near_simplex":   # equidistant team + small idiosyncratic noise
        X = np.eye(n) + 0.05 * rng.standard_normal((n, n))
        return pdist_euclid(X)
    if family == "one_outlier":    # one specialist/faulty member
        X = np.eye(n) + 0.05 * rng.standard_normal((n, n))
        X[0] *= 6.0
        return pdist_euclid(X)
    if family == "hetero_scale":   # members differ in how idiosyncratic they are
        s = np.exp(0.6 * rng.standard_normal(n))
        X = s[:, None] * rng.standard_normal((n, 64))
        return pdist_euclid(X)
    if family == "two_roles":      # two equal roles, labels ordered by role
        c = np.repeat([0, 1], n // 2)
        C = np.array([[0.0] * 8, [3.0] + [0.0] * 7])
        X = C[c] + 0.3 * rng.standard_normal((n, 8))
        return pdist_euclid(X)
    if family == "unequal_roles":  # four roles of unequal size, ordered
        sizes = np.array([0.4, 0.3, 0.2, 0.1]) * n
        sizes = np.maximum(1, np.round(sizes).astype(int)); sizes[0] += n - sizes.sum()
        c = np.repeat(np.arange(4), sizes)
        C = 3.0 * rng.standard_normal((4, 8))
        X = C[c] + 0.3 * rng.standard_normal((n, 8))
        return pdist_euclid(X)
    if family == "ring":           # continuum of behaviours on a circle, ordered
        th = 2 * np.pi * np.arange(n) / n
        X = np.c_[np.cos(th), np.sin(th)] + 0.02 * rng.standard_normal((n, 2))
        return pdist_euclid(X)
    raise ValueError(family)


def pdist_euclid(X):
    G = X @ X.T
    sq = np.diag(G)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0.0)
    D = np.sqrt(D2)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D

FAMILIES = ["near_simplex", "one_outlier", "hetero_scale", "two_roles", "unequal_roles", "ring"]


def random_regular_edges(n, k, rng):
    """Random k-regular (multi)graph: union of k independent uniform perfect matchings.
    Regular with multiplicity; an expander w.h.p. for k >= 3."""
    assert n % 2 == 0
    E = np.concatenate([np.sort(rng.permutation(n).reshape(-1, 2), axis=1) for _ in range(k)])
    return E.astype(np.int64)
