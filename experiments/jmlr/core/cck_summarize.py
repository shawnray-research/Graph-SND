import json, sys, collections, numpy as np
res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/jmlr/cck/cck_results.json"))
fams = ["near_simplex", "hetero_scale", "one_outlier", "two_roles", "unequal_roles", "ring"]
keys = ["rr", "un", "cck", "rr_exact", "un_exact", "cck_exact", "cck_medoid_mc", "cck_medoid_exact", "cck_bestpivot_exact", "cck_worstpivot_exact"]
g = collections.defaultdict(list)
for r in res:
    if r["group"].startswith("synth:"):
        g[r["group"][6:]].append(r)
pool = lambda rows, k: 100 * np.sqrt(np.mean([x[k] ** 2 for x in rows]))
print("%-22s" % "pooled RMSE %", " ".join("%13s" % f for f in fams))
for k in keys:
    print("%-22s" % k, " ".join("%13.4f" % pool(g[f], k) for f in fams))
print("kappa", " ".join("%13.5f" % np.mean([x["kappa"] for x in g[f]]) for f in fams))
print("cck rel bias (%)", " ".join("%13.4f" % (100*np.mean([x["cck_bias"] for x in g[f]])) for f in fams))
print("per-table cck MC/exact:", [(r["name"], round(100*r["cck"],3), round(100*r["cck_exact"],3)) for r in res if r["group"].startswith("synth:")])
fmt = lambda v: ("%.2f" % v) if v < 10 else ("%d" % int(np.floor(v + 0.5)))
print("Adaptive pivot (CCK) & " + " & ".join("$%s$" % fmt(pool(g[f], "cck")) for f in fams) + " \\\\")
print("[exact] Adaptive pivot (CCK) & " + " & ".join("$%s$" % fmt(pool(g[f], "cck_exact")) for f in fams) + " \\\\")
print()
print("%-48s %4s %4s %4s %9s %9s %9s | %9s %9s %9s | %9s %9s %8s" % ("table", "n", "m", "m'", "RR%", "Unif%", "CCK%", "RRex%", "Unex%", "CCKex%", "CCKmed%", "CCKbest%", "kappa"))
for r in res:
    if r["group"].startswith("synth:"):
        continue
    print("%-48s %4d %4d %4d %9.3f %9.3f %9.3f | %9.3f %9.3f %9.3f | %9.3f %9.3f %8.3f" % (r["name"], r["n"], r["m"], r["m_prime"], 100*r["rr"], 100*r["un"], 100*r["cck"], 100*r["rr_exact"], 100*r["un_exact"], 100*r["cck_exact"], 100*r["cck_medoid_exact"], 100*r["cck_bestpivot_exact"], r["kappa"]))
gr = collections.defaultdict(list)
for r in res:
    if not r["group"].startswith("synth:"):
        gr[r["group"]].append(r)
for grp, rows in gr.items():
    print("pooled", grp, len(rows), " ".join("%s=%.3f" % (k, pool(rows, k)) for k in ["rr", "un", "cck", "rr_exact", "un_exact", "cck_exact"]),
          "| CCK/RR exact ratio range %.2f-%.2f" % (min(x["cck_exact"]/x["rr_exact"] for x in rows), max(x["cck_exact"]/x["rr_exact"] for x in rows)))
