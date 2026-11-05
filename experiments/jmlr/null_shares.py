"""Distinctiveness share of unstructured populations (Section 7.4, null reference).

Individuals are i.i.d. standard Gaussian points in R^p (p = 2, the action
dimension of the VMAS tasks), compared by Euclidean distance. Prints the 5th-95th
percentile range of the distinctiveness share over 2,000 draws at n = 16 and 32.
"""
import numpy as np

rng = np.random.default_rng(0)


def share(D):
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    mu = D[iu].mean()
    a = (D.sum(1) - (n - 1) * mu) / (n - 2)
    R = D - mu - a[:, None] - a[None, :]
    np.fill_diagonal(R, 0)
    tot = ((D[iu] - mu) ** 2).mean()
    inter = (R[iu] ** 2).mean()
    return (tot - inter) / tot


for p in (2, 8, 64):
    for n in (16, 32):
        s = []
        for _ in range(2000):
            X = rng.standard_normal((n, p))
            D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
            s.append(share(D))
        lo, hi = 100 * np.percentile(s, [5, 95])
        print(f"p={p:3d} n={n:3d}  distinctiveness share 5-95%: {lo:.0f}-{hi:.0f}%  median {100*np.median(s):.0f}%")
