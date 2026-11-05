import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anatomy import anatomy
def kmeans(X, K, restarts, rng):
    best, best_lab = np.inf, None
    for _ in range(restarts):
        C = [X[rng.integers(len(X))]]
        for _k in range(1, K):  # k-means++
            d2 = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
            C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(C)
        for _it in range(300):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            Cn = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(K)])
            if np.allclose(Cn, C): break
            C = Cn
        inertia = ((X - C[lab]) ** 2).sum()
        if inertia < best: best, best_lab = inertia, lab
    return best_lab
def ari(a, b):
    from math import comb
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    M = np.zeros((len(ua), len(ub)), dtype=np.int64); np.add.at(M, (ia, ib), 1)
    s = sum(comb(int(x), 2) for x in M.ravel()); sa = sum(comb(int(x), 2) for x in M.sum(1)); sb = sum(comb(int(x), 2) for x in M.sum(0))
    n = len(a); exp = sa * sb / comb(n, 2); mx = 0.5 * (sa + sb)
    return (s - exp) / (mx - exp)
for f in sys.argv[1:]:
    z = np.load(f, allow_pickle=True); an = anatomy(z["D"]); w, V = np.linalg.eigh(an["R"])
    idx = np.argsort(-np.abs(w))[:3]; X = V[:, idx]
    lab = kmeans(X, 4, 10, np.random.default_rng(0))
    print(os.path.basename(f), "n", len(lab), "ARI(k-means K=4 on top-3 eigvecs of R, recipe labels) = %.4f" % ari(lab, z["recipe"]))
