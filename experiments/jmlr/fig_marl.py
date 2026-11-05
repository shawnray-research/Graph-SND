"""Figure: anatomy of trained VMAS teams over training + measured vs predicted efficiency factor."""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
import numpy as np

root, out = sys.argv[1], sys.argv[2]
# optional: tables of the long runs (rows written by experiments/jmlr/core/long_runs.py)
long_rows = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else []
long_runs = {}
for r in long_rows:
    if r["kind"] == "trained":
        long_runs.setdefault((r["task"], r["n"], r["seed"], r["algo"]), []).append(r)
rng = np.random.default_rng(0)
C_NAV, C_DIS = "#332288", "#AA4499"  # Navigation / Dispersion task colors (Tol indigo / purple; unused elsewhere)


def anatomy(D):
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    mu = D[iu].mean()
    a = (D.sum(1) - (n - 1) * mu) / (n - 2)
    R = D - mu - a[:, None] - a[None, :]
    np.fill_diagonal(R, 0)
    tot = ((D[iu] - mu) ** 2).mean()
    inter = (R[iu] ** 2).mean()
    return mu, tot, inter


def measured_ratio(D, k=4, reps=4000):
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    dvec = D[iu]
    N = len(dvec)
    m = n * k // 2
    base_i = np.array([t for t in range(n) for s in range(1, k // 2 + 1)])
    base_j = np.array([(t + s) % n for t in range(n) for s in range(1, k // 2 + 1)])
    eb = np.empty(reps)
    eu = np.empty(reps)
    for r in range(reps):
        p = rng.permutation(n)
        eb[r] = D[p[base_i], p[base_j]].mean()
        eu[r] = dvec[rng.choice(N, m, replace=False)].mean()
    return eu.var() / eb.var()


rows = []
files = [] if root.endswith(".json") else sorted(glob.glob(os.path.join(root, "*.npz")))
if root.endswith(".json"):  # reuse the cached table written by a previous run
    rows = json.load(open(root))
for f in files:
    z = np.load(f, allow_pickle=True)
    kind = str(z["kind"])
    D = z["D"].astype(float)
    n = D.shape[0]
    mu, tot, inter = anatomy(D)
    kappa = (tot - inter) / inter
    pred = (1 + kappa) * n * (n - 3) / ((n - 2) * (n + 1))
    meas = measured_ratio(D) if n >= 8 else np.nan
    rows.append(dict(task=str(z["task"]), n=n, seed=int(z["seed"]), it=int(z["iter"]), kind=kind,
                     mu=mu, share_main=(tot - inter) / tot, pred=pred, meas=meas))
if files:
    json.dump(rows, open(out + "_data.json", "w"), indent=0, default=float)

# ---------------------------------------------------------------- presentation (per-figure style overrides)
INK, TICK, SPINE, GRID = "#222222", "#555555", "#888888", "#E6E6E6"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "xtick.color": TICK, "ytick.color": TICK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 3.0,
    "axes.edgecolor": SPINE, "axes.linewidth": 0.6,
})

FW, FH = 6.0, 2.42
TITLE_PAD = 7.0  # pt; keeps the panel letters clear of the top y tick label of (b)
fig = plt.figure(figsize=(FW, FH))
# panel boxes in inches (x0, width); the gaps hold each panel's y tick labels and y label
AX_Y, AX_H = 0.42, 1.76
boxes = [(0.53, 1.60), (2.60, 1.60), (4.66, 1.33)]
axes = [fig.add_axes([x / FW, AX_Y / FH, w / FW, AX_H / FH]) for x, w in boxes]

# (a, b) one thin, translucent line per seed; markers only at the recorded iterations.
# Task: color + marker (circle / square); Navigation n=32 vs n=16: dashed vs solid line.
DASH32 = (0, (3.2, 1.6))
groups = [("dispersion", 16, C_DIS, "s", "-", "Dispersion, $n=16$"),
          ("navigation", 32, C_NAV, "o", DASH32, "Navigation, $n=32$"),
          ("navigation", 16, C_NAV, "o", "-", "Navigation, $n=16$")]
for gi, (task, n, col, mk, ls, lab) in enumerate(groups):
    tr = [r for r in rows if r["task"] == task and r["n"] == n and r["kind"] == "trained"]
    for sd in sorted(set(r["seed"] for r in tr)):
        rr = sorted([r for r in tr if r["seed"] == sd], key=lambda r: r["it"])
        its = [r["it"] for r in rr]
        for ax_, ys in ((axes[0], [r["mu"] for r in rr]), (axes[1], [100 * r["share_main"] for r in rr])):
            ax_.plot(its, ys, color=col, ls=ls, lw=0.9 if not long_runs else 0.7,
                     alpha=0.7 if not long_runs else 0.45, zorder=3 + gi, solid_capstyle="round",
                     dash_capstyle="butt")
            ax_.plot(its, ys, ls="", marker=mk, ms=3.0 if not long_runs else 2.3, mfc=col, mec="white", mew=0.4,
                     alpha=0.95 if not long_runs else 0.6, zorder=3 + gi)
# long runs (n = 16): bold; independent critics solid, centralized critic (MAPPO-style) dashed
DASH_C = (0, (2.6, 1.3))
long_max_it = 0
for (task, n_, seed_, algo), traj in sorted(long_runs.items()):
    traj = sorted(traj, key=lambda r: r["it"])
    its = [r["it"] for r in traj]
    long_max_it = max(long_max_it, its[-1])
    col = C_DIS if task == "dispersion" else C_NAV
    for ax_, ys in ((axes[0], [r["mu"] for r in traj]), (axes[1], [100 * r["share_main"] for r in traj])):
        ax_.plot(its, ys, color=col, ls="-" if algo == "independent" else DASH_C, lw=1.6, alpha=0.95, zorder=9,
                 solid_capstyle="round", dash_capstyle="butt")

XT = [0, 5, 10, 20, 50, 100, 200] if not long_runs else [0, 5, 10, 50, 200, 1000]
XMAX = 226 if not long_runs else max(1.12 * long_max_it, 1080)
for ax_ in (axes[0], axes[1]):
    ax_.set_xscale("symlog", linthresh=10, linscale=0.4)
    ax_.set_xlim(-1.1, XMAX)
    ax_.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(XT))
    ax_.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(v) for v in XT]))
    ax_.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_.set_xlabel("PPO iteration")
axes[0].set_yscale("log")
axes[0].set_ylim(0.002, 0.4 if not long_runs else 1.4)
axes[0].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.003, 0.01, 0.03, 0.1, 0.3] + ([1] if long_runs else [])))
axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axes[0].set_ylabel(r"Diversity $\mu$ (SND)")
# growth of mu from initialization to the last recorded iteration, over every plotted run
growth = []
for task, n, *_ in groups:
    tr = [r for r in rows if r["task"] == task and r["n"] == n and r["kind"] == "trained"]
    for sd in set(r["seed"] for r in tr):
        rr = sorted([r for r in tr if r["seed"] == sd], key=lambda r: r["it"])
        if rr[0]["it"] == 0 and len(rr) > 1:
            growth.append(rr[-1]["mu"] / rr[0]["mu"])
for traj in long_runs.values():
    traj = sorted(traj, key=lambda r: r["it"])
    growth.append(traj[-1]["mu"] / traj[0]["mu"])
axes[0].set_title(f"Level grows {min(growth):.0f}–{max(growth):.0f}$\\times$", pad=TITLE_PAD)
axes[1].set_ylabel("Distinctiveness share (%)")
axes[1].set_title("…and changes form", pad=TITLE_PAD)
axes[1].set_ylim(0, 100)
axes[1].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
axes[1].axhline(50, color="#BBBBBB", lw=0.8, zorder=1)
# legend entries top to bottom in the order the curve groups stack in (a)
leg_h = [Line2D([], [], color=C_DIS, ls="-", lw=1.0, marker="s", ms=3.0, mec="white", mew=0.4),
         Line2D([], [], color=C_NAV, ls=DASH32, lw=1.0, marker="o", ms=3.0, mec="white", mew=0.4),
         Line2D([], [], color=C_NAV, ls="-", lw=1.0, marker="o", ms=3.0, mec="white", mew=0.4)]
LEG = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=1.0, fancybox=False)  # hides grid only
leg_l = ["Dispersion, $n=16$", "Navigation, $n=32$", "Navigation, $n=16$"]
if long_runs:
    leg_h += [Line2D([], [], color="#444444", ls="-", lw=1.6),
              Line2D([], [], color="#444444", ls=DASH_C, lw=1.6)]
    leg_l += ["Long run", "Long, central critic"]
axes[0].legend(leg_h, leg_l, **LEG,
               loc="lower right", handlelength=2.2, borderaxespad=0.2, labelspacing=0.3, handletextpad=0.5,
               borderpad=0.2)

# (c) measured vs predicted efficiency factor on all real matrices
kinds = {"trained": ("o", "Unmodified teams"), "outlier1": ("s", "One agent reset"), "twogroup": ("^", "Half reset")}
allv = []
for kind, (mk, lab) in kinds.items():
    rr = [r for r in rows + long_rows if r["kind"] == kind and np.isfinite(r["meas"])]
    xs = [r["pred"] for r in rr]
    ys = [r["meas"] for r in rr]
    cs = [C_NAV if r["task"] == "navigation" else C_DIS for r in rr]
    axes[2].scatter(xs, ys, marker=mk, s=15, c=cs, edgecolor="white", linewidth=0.4, label=lab, zorder=3)
    allv += xs + ys
lo, hi = 0.7, max(allv) * 1.3
axes[2].plot([lo, hi], [lo, hi], color="black", lw=0.9, ls=(0, (1, 1.5)), zorder=2)
axes[2].axhline(1.0, color="#999999", lw=0.7, ls=(0, (4, 2.5)), zorder=1)  # equal efficiency
axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_xlim(lo, hi)
axes[2].set_ylim(lo, hi)
for axis in (axes[2].xaxis, axes[2].yaxis):
    axis.set_major_locator(matplotlib.ticker.FixedLocator([1, 3, 10, 30]))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(matplotlib.ticker.NullLocator())
axes[2].set_xlabel(r"Predicted $\mathrm{Var}_{\mathrm{U}}/\mathrm{Var}_{\mathrm{B}}$")
axes[2].set_ylabel(r"Measured $\mathrm{Var}_{\mathrm{U}}/\mathrm{Var}_{\mathrm{B}}$")
axes[2].set_title(f"{sum(np.isfinite(r['meas']) for r in rows + long_rows)} VMAS tables", pad=TITLE_PAD)
GRAYMK = "#444444"
# shape key: neutral open markers (shape only; the fill color of the data encodes the task)
hs = [Line2D([], [], marker="o", ls="", mfc="white", mec=GRAYMK, mew=0.7, ms=3.8, label="Unmodified teams"),
      Line2D([], [], marker="s", ls="", mfc="white", mec=GRAYMK, mew=0.7, ms=3.8, label="One agent reset"),
      Line2D([], [], marker="^", ls="", mfc="white", mec=GRAYMK, mew=0.7, ms=3.8, label="Half reset"),
      Line2D([], [], marker="o", ls="", color=C_NAV, mec="white", mew=0.4, ms=4, label="Navigation"),
      Line2D([], [], marker="o", ls="", color=C_DIS, mec="white", mew=0.4, ms=4, label="Dispersion")]
leg_shape = axes[2].legend(handles=hs[:3], **LEG, loc="upper left", handletextpad=0.2, handlelength=1.0,
                           borderaxespad=0.15, labelspacing=0.3, borderpad=0.2)
axes[2].add_artist(leg_shape)
axes[2].legend(handles=hs[3:], **LEG, loc="lower right", handletextpad=0.2, handlelength=1.0,
               borderaxespad=0.15, labelspacing=0.3, borderpad=0.2)
for ax in axes:
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
# panel letters: bold, on the titles' baseline, left-aligned at the same offset LET_DX (inches) left of every panel
fig.canvas.draw()
inv = fig.transFigure.inverted()
LET_DX = 0.40
y_let = inv.transform(axes[0].title.get_transform().transform(axes[0].title.get_position()))[1]
for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
    fig.text(ax.get_position().x0 - LET_DX / FW, y_let, lab, fontsize=9, fontweight="bold", va="baseline",
             ha="left")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)
ratios = [r["meas"] / r["pred"] for r in rows + long_rows if np.isfinite(r["meas"])]
print("n matrices", len(rows), "meas/pred median", np.median(ratios), "range", min(ratios), max(ratios))
