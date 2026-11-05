"""Error-bar experiment (Sec. 5.7) on the three unmodified final VMAS tables that e5e6.csv omitted.
Run from the repository root: python experiments/jmlr/core/e5e6_missing_tables.py"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import e5e6
base = "results/jmlr/matrices/vmas_wave1/"
names = ["navigation_n16_s0_trained_it20", "navigation_n16_s2_trained_it34", "navigation_n32_s0_trained_it10"]
rows = []
for nm in names:
    z = np.load(base + nm + ".npz", allow_pickle=True); D = z["D"].astype(np.float64)
    rows += e5e6.one((nm, "real", f"{z['task']}_{z['kind']}", D))
for r in rows:
    print(r["matrix"], "k", r["k"], "cover_rr %.3f cover_u %.3f spearman_rr %.3f" % (r["cover_rr"], r["cover_u"], r["spearman_rr"]))
R = list(csv.DictReader(open("results/jmlr/core/e5e6.csv")))
vm = [r for r in R if r["source"] == "real" and "zoo" not in r["matrix"]]
for k in ("4", "16"):
    xs = [float(r["cover_rr"]) for r in vm if r["k"] == k]
    sp = [float(r["spearman_rr"]) for r in vm if r["k"] == k and r["spearman_rr"] not in ("", "nan")]
    print("existing VMAS k=%s: cover_rr %.3f-%.3f (n=%d), spearman mean %.3f" % (k, min(xs), max(xs), len(xs), np.nanmean(sp)))
