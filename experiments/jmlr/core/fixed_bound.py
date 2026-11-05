"""Theorem (fixed regular designs): |mu_G/mu - 1| <= (lam2 - lamn)/(2k) * ||R||_*/(n mu).
Evaluate bound vs realized error for fixed simple random k-regular graphs and the fixed ring,
on synthetic families with role/angle-ordered labels and on real tables."""
import sys, os, csv, glob
import numpy as np, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anatomy import *
rows = []
def evaluate(name, D, graphs_k=(4, 8, 16), ngraphs=20):
    an = anatomy(D); n = an["n"]; mu = an["mu"]
    nuc = np.abs(np.linalg.eigvalsh(an["R"])).sum() / (n * mu)
    for k in graphs_k:
        if k >= n - 1: continue
        for g in range(ngraphs + 1):
            if g < ngraphs:
                G = nx.random_regular_graph(k, n, seed=1000 * k + g); kind = "random_regular"
            else:
                G = nx.circulant_graph(n, list(range(1, k // 2 + 1))); kind = "ring"
            A = nx.to_numpy_array(G, nodelist=range(n))
            ev = np.sort(np.linalg.eigvalsh(A))[::-1]
            E = np.array(G.edges())
            err = abs(D[E[:, 0], E[:, 1]].mean() / mu - 1)
            bound = (ev[1] - ev[-1]) / (2 * k) * nuc
            rows.append(dict(table=name, n=n, k=k, graph=kind, seed=g, lam2=ev[1], lamn=ev[-1], nuc_rel=nuc,
                             err=err, bound=bound, ratio=err / bound))
for fam in FAMILIES:
    evaluate(fam, synthetic(fam, 200, np.random.default_rng(7)))
for f in sys.argv[2:]:
    for p in sorted(glob.glob(f)):
        z = np.load(p, allow_pickle=True); D = z["D"].astype(np.float64)
        evaluate(os.path.basename(p)[:-4], D, graphs_k=(4, 8), ngraphs=20)
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
import collections
agg = collections.defaultdict(list)
for r in rows: agg[(r["table"], r["k"], r["graph"])].append(r)
for key, rs in agg.items():
    e = np.array([r["err"] for r in rs]); b = np.array([r["bound"] for r in rs])
    print(f"{key[0][:34]:34s} k={key[1]:2d} {key[2]:15s} err max {100*e.max():7.3f}% mean {100*e.mean():7.3f}% | bound {100*b.mean():8.2f}% | max ratio {np.max(e/b):.3f} | (l2-ln)/2k {np.mean([(r['lam2']-r['lamn'])/(2*r['k']) for r in rs]):.3f} nuc {rs[0]['nuc_rel']:.3f}")
