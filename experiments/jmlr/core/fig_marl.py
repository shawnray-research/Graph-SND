import sys, os, csv, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *

setup()
outdir, anat_csv, e1_csv, e2_csv = sys.argv[1:5]
A = [r for r in csv.DictReader(open(anat_csv)) if r["kind"] == "trained" and int(r["n"]) <= 64]
for r in A:
    for k in ["n", "iter", "seed"]: r[k] = int(r[k])
    for k in ["mu", "share_distinct", "share_rel", "kappa", "eff"]: r[k] = float(r[k])
groups = collections.defaultdict(lambda: collections.defaultdict(list))
for r in A:
    groups[(r["task"], r["n"])][r["iter"]].append(r)
GSTYLE = {("dispersion", 16): ("Dispersion, $n$=16", "#009E73", "-", "o"),
          ("navigation", 8): ("Navigation, $n$=8", "#56B4E9", "--", "s"),
          ("navigation", 16): ("Navigation, $n$=16", "#0072B2", "-", "s"),
          ("navigation", 32): ("Navigation, $n$=32", "#000000", "-", "D")}

fig, axes = plt.subplots(1, 4, figsize=(6.0, 2.05), gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1, 1, 1.05]})
# (a) level mu over training
ax = axes[0]
for g, st in GSTYLE.items():
    if g not in groups: continue
    its = sorted(groups[g]); its = [i for i in its if len(groups[g][i]) > 0]
    y = [np.mean([r["mu"] for r in groups[g][i]]) for i in its]
    x = [max(i, 0.5) for i in its]
    ax.plot(x, y, color=st[1], ls=st[2], marker=st[3], markersize=2.4, lw=1.0, label=st[0])
ax.set_xscale("symlog", linthresh=1); ax.set_yscale("log")
ax.set_xlabel("PPO iteration"); ax.set_ylabel(r"level $\mu$ (= SND)")
ax.set_xlim(-0.2, None)
panel_label(ax, "a")
# (b) distinctiveness share over training
ax = axes[1]
for g, st in GSTYLE.items():
    if g not in groups: continue
    its = sorted(groups[g])
    m = np.array([np.mean([r["share_distinct"] for r in groups[g][i]]) for i in its])
    lo = np.array([np.min([r["share_distinct"] for r in groups[g][i]]) for i in its])
    hi = np.array([np.max([r["share_distinct"] for r in groups[g][i]]) for i in its])
    x = np.array([max(i, 0.5) for i in its], dtype=float)
    ax.fill_between(x, lo, hi, color=st[1], alpha=0.15, lw=0)
    ax.plot(x, m, color=st[1], ls=st[2], marker=st[3], markersize=2.4, lw=1.0, label=st[0])
ax.set_xscale("symlog", linthresh=1); ax.set_ylim(0, 1)
ax.set_xlabel("PPO iteration"); ax.set_ylabel(r"distinctiveness share $\sigma^2_{\rm D}/\sigma^2$")
ax.axhline(0.5, color="#BBBBBB", lw=0.5, ls=":")
ax.text(0.97, 0.03, "relational = 1 $-$ share", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="#555555")
panel_label(ax, "b")
# (c) measured vs predicted gain on all real matrices
ax = axes[2]
E = list(csv.DictReader(open(e1_csv)))
eff = collections.defaultdict(dict)
for r in E:
    key = (r["matrix"], r["k"]); eff[key][r["design"]] = float(r["var_meas"]); eff[key]["pred"] = float(r["eff_pred"])
    eff[key]["fam"] = r["family"]
xs, ys, cs = [], [], []
for key, d in eff.items():
    if "uniform" in d and "round_robin" in d:
        xs.append(d["pred"]); ys.append(d["uniform"] / d["round_robin"])
        f = d["fam"]
        cs.append("#009E73" if f.startswith("dispersion") else ("#7F7F7F" if "zoo" in f else "#0072B2"))
xs, ys = np.array(xs), np.array(ys)
ax.scatter(xs, ys, s=6, c=cs, edgecolor="white", linewidth=0.25, zorder=3)
lo, hi = min(xs.min(), ys.min()) / 1.4, max(xs.max(), ys.max()) * 1.4
ax.plot([lo, hi], [lo, hi], color="black", lw=0.6, zorder=1)
ax.axhline(1, color="#BBBBBB", lw=0.5, ls="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("predicted gain"); ax.set_ylabel(r"measured $\mathrm{Var}_{\rm unif}/\mathrm{Var}_{\rm RR}$")
rat = ys / xs
ax.text(0.04, 0.96, f"{len(xs)} real cells\nratio {np.median(rat):.2f} (median)", transform=ax.transAxes,
        va="top", ha="left", fontsize=5.8)
panel_label(ax, "c")
# (d) relative RMSE per design at k=4 on late-training Dispersion
ax = axes[3]
R2 = [r for r in csv.DictReader(open(e2_csv)) if r["k"] == "4"]
late = collections.defaultdict(list)
sel_fams = [f for f in set(r["family"] for r in R2)]
target = [r for r in R2 if r["family"].startswith("dispersion") and "trained" in r["family"]]
lastit = {}
for r in target:
    key = r["matrix"].rsplit("_it", 1)[0]
    it = int(r["matrix"].rsplit("_it", 1)[1])
    lastit[key] = max(lastit.get(key, -1), it)
for r in target:
    key = r["matrix"].rsplit("_it", 1)[0]
    if int(r["matrix"].rsplit("_it", 1)[1]) == lastit[key]:
        late[r["design"]].append(float(r["rel_rmse"]) ** 2)
order = ["round_robin", "expander_fixed", "uniform", "bernoulli_ratio", "bernoulli_ht", "agent_subsample", "landmark", "local_knn"]
order = [d for d in order if d in late]
vals = [100 * np.sqrt(np.mean(late[d])) for d in order]
yy = np.arange(len(order))[::-1]
for y0, d, v in zip(yy, order, vals):
    lab, col, ls, mk, op = DESIGN_STYLE[d]
    ax.barh(y0, v, color=col if not op else "white", edgecolor=col, lw=0.7, height=0.7)
    ax.text(v * 1.08, y0, f"{v:.1f}", va="center", ha="left", fontsize=5.6)
ax.set_yticks(yy); ax.set_yticklabels([DESIGN_STYLE[d][0].replace(" (ours)", "*") for d in order], fontsize=5.8)
ax.set_xscale("log"); ax.set_xlabel("rel. RMSE (%) at $m=2n$")
ax.grid(axis="y", visible=False)
ax.set_xlim(min(vals) / 2, max(vals) * 4)
panel_label(ax, "d")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.45, 1.10), columnspacing=1.2)
save(fig, os.path.join(outdir, "fig_marl"))
print("ok", len(xs))
