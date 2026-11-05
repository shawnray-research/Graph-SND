import sys, os, csv, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator, LogLocator

setup()
plt.rcParams.update({  # per-figure overrides of the shared style (figure standard: axes, ticks, grid)
    "axes.edgecolor": "#888888", "axes.linewidth": 0.6,
    "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
    "xtick.color": "#555555", "ytick.color": "#555555",
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
})
files = sys.argv[2:]
outdir = sys.argv[1]
rows = []
for f in files:
    if not os.path.exists(f):
        sys.exit(f"missing input table: {f}")  # the cell count in the annotation would silently change
    rows += list(csv.DictReader(open(f)))
for r in rows:
    for k in ["var_pred", "var_meas", "eff_pred", "kappa", "mu"]:
        r[k] = float(r[k])

# Layout in inches: two square panels (equal log ranges on both axes, so y = x runs at 45 degrees).
# S is chosen so that the tight-bbox PDF is exactly 6.00 in wide (\textwidth).
S = 2.3305                # side of each square axes
ML, GAP, MB, MT, MR = 0.50, 0.66, 0.36, 0.20, 0.14   # left, between spines, bottom, top, right
W, H = ML + S + GAP + S + MR, MB + S + MT
fig = plt.figure(figsize=(W, H))
axA = fig.add_axes([ML / W, MB / H, S / W, S / H])
axB = fig.add_axes([(ML + S + GAP) / W, MB / H, S / W, S / H])
YLAB_OFF = 0.36         # ylabel (its axis-side edge) sits this far left of the spine in both panels [in]
LETTER_OFF = 0.50       # panel letters: left edge this far left of the spine in both panels [in]


def theory_line(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.9, ls=(0, (1, 1.5)), zorder=1.5, solid_capstyle="butt")


# (a) measured / predicted variance against the predicted relative variance (log x, linear y). On a log-log
# y = x scatter a 2.6% deviation is invisible; the ratio view shows directly that every design agrees with its
# closed form to within Monte Carlo error over six decades. Band: +/- 2 Monte Carlo s.e. of a variance estimated
# from R = 3,000 draws, sqrt(2/(R-1)).
ax = axA
A_STYLE = {  # design: (label, color, marker, size [pt^2], edge width, alpha, zorder, filled)
    "uniform":            ("Uniform pairs", UNIFORM_LIGHT, "s", 11, 0.5, 0.8, 2, False),
    "bernoulli_ht":       ("Bernoulli (HT)", DESIGN_STYLE["bernoulli_ht"][1], "^", 13, 0.5, 0.8, 3, False),
    "round_robin":        ("Balanced (round-robin)", DESIGN_STYLE["round_robin"][1], "o", 10, 0.6, 0.85, 4, False),
    "shuffled_circulant": ("Balanced (permuted circulant)", DESIGN_STYLE["shuffled_circulant"][1], "D", 3.5, 0.0, 0.9, 5, True),
}
R_DRAWS = 3000
se = np.sqrt(2.0 / (R_DRAWS - 1))
lo, hi = 10 ** -6.5, 1.0
ylo, yhi = 0.85, 1.13
ax.axhspan(1 - 2 * se, 1 + 2 * se, color="#EDEDED", zorder=0.5, lw=0)
ax.axhline(1.0, color="black", lw=0.9, ls=(0, (1, 1.5)), zorder=1.5)
allx, ally = [], []
for dname, (lab, col, mk, sz, lw, al, z, filled) in A_STYLE.items():
    sel = [r for r in rows if r["design"] == dname and r["var_pred"] > 0 and r["var_meas"] > 0]
    x = np.array([r["var_pred"] / r["mu"] ** 2 for r in sel]); y = np.array([r["var_meas"] / r["var_pred"] for r in sel])
    allx += list(x); ally += list(y)
    if filled:
        ax.scatter(x, y, s=sz, marker=mk, facecolor=col, edgecolor="none", alpha=al, zorder=z)
    else:
        ax.scatter(x, y, s=sz, marker=mk, facecolor="none", edgecolor=col, linewidth=lw, alpha=al, zorder=z)
allx, ally = np.array(allx), np.array(ally)
assert allx.min() > lo and allx.max() < hi, (allx.min(), allx.max())
n_out = int(np.sum((ally < ylo) | (ally > yhi)))
assert n_out == 0, f"{n_out} cells outside the y-range"
ax.set_xscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(ylo, yhi)
decades = [10.0 ** e for e in range(-6, 1)]
fmt_dec = FuncFormatter(lambda v, _: "" if round(np.log10(v)) % 2 else f"$10^{{{round(np.log10(v))}}}$")
ax.xaxis.set_major_locator(FixedLocator(decades)); ax.xaxis.set_major_formatter(fmt_dec)
ax.xaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_locator(FixedLocator([0.9, 0.95, 1.0, 1.05, 1.1]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.set_xlabel(r"Predicted relative variance $\mathrm{Var}/\mu^2$", labelpad=2.5)
ax.set_ylabel("Measured / predicted variance")
ratios = np.array([r["var_meas"] / r["var_pred"] for r in rows if r["var_pred"] > 0])
inside = float(np.mean(np.abs(ratios - 1) <= 2 * se))
nsyn = len(set(r["matrix"] for r in rows if r["source"] == "synthetic"))
nreal = len(set(r["matrix"] for r in rows if r["source"] != "synthetic"))
ax.text(0.03, 0.97, f"{len(ratios):,} cells, {nsyn} synthetic + {nreal} real tables\n"
        f"ratio {ratios.mean():.3f} $\\pm$ {ratios.std():.3f}; {100 * inside:.0f}% inside the band",
        transform=ax.transAxes, va="top", ha="left", fontsize=7, color=INK, linespacing=1.25, zorder=6)
leg_order = ["bernoulli_ht", "uniform", "round_robin", "shuffled_circulant"]
handles = []
for d in leg_order:
    lab, col, mk, sz, lw, al, z, filled = A_STYLE[d]
    ms = np.sqrt(sz) * 1.5
    handles.append(Line2D([], [], ls="none", marker=mk, markersize=ms, markerfacecolor=col if filled else "none",
                          markeredgecolor=col if not filled else "none", markeredgewidth=lw * 1.4))
from matplotlib.patches import Patch
handles.append(Patch(facecolor="#EDEDED", edgecolor="none"))
SHORT = {"bernoulli_ht": "Bernoulli (HT)", "uniform": "Uniform pairs", "round_robin": "Round-robin",
         "shuffled_circulant": "Permuted circulant"}  # the two balanced designs share the balanced blue
legA = ax.legend(handles, [SHORT[d] for d in leg_order] + [r"$\pm2$ s.e."], loc="lower center", ncol=2, handletextpad=0.2,
                 borderaxespad=0.25, handlelength=0.9, columnspacing=0.6, labelspacing=0.25, fontsize=7, frameon=True,
                 borderpad=0.2)
legA.get_frame().set(facecolor="white", edgecolor="none", alpha=1.0, linewidth=0)

# (b) measured vs predicted efficiency factor
ax = axB
eff = collections.defaultdict(dict)
for r in rows:
    key = (r["matrix"], r["k"]); eff[key][r["design"]] = r["var_meas"]
    eff[key]["pred"] = r["eff_pred"]
    eff[key]["fam"] = r["family"] if r["source"] == "synthetic" else ("zoo" if "zoo" in r["family"] else "vmas")
pts = collections.defaultdict(list)
for key, d in eff.items():
    if "uniform" in d and "round_robin" in d:
        pts[d["fam"]].append((d["pred"], d["uniform"] / d["round_robin"]))
allp = []
FAM_MK = {"near_simplex": "o", "one_outlier": "s", "hetero_scale": "D", "two_roles": "^",
          "unequal_roles": "v", "ring": "P", "zoo": "X"}
FAM_MS = {"D": 10, "s": 12, "P": 15, "X": 15}  # equalise visual size across marker shapes [pt^2]
order_plot = ["vmas", "zoo", "one_outlier", "hetero_scale", "near_simplex", "unequal_roles", "two_roles", "ring"]
for fam in order_plot:
    if fam not in pts:
        continue
    l = np.array(pts[fam]); allp += list(l.ravel())
    lab, col = FAMILY_STYLE.get(fam, (fam, "#333333"))
    if fam == "vmas":
        ax.scatter(l[:, 0], l[:, 1], s=11, color=col, alpha=0.4, linewidth=0, zorder=2, label=lab)
    else:
        mk = FAM_MK[fam]
        hollow = fam == "two_roles"  # sits on top of the continuum near (1, 1): large open triangles drawn last
        ax.scatter(l[:, 0], l[:, 1], s=FAM_MS.get(mk, 14) * (1.9 if hollow else 1.0), marker=mk,
                   facecolor="none" if hollow else col, edgecolor=col if hollow else "white",
                   linewidth=0.9 if hollow else 0.4, zorder=5 if hollow else (6 if fam == "ring" else 3), label=lab)
lo, hi = 0.5, 1000.0
assert min(allp) > lo and max(allp) < hi, (min(allp), max(allp))
theory_line(ax, lo, hi)
ax.axhline(1.0, color="#999999", lw=0.7, ls=(0, (4, 2.5)), zorder=1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_aspect("equal", adjustable="box")
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    axis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"Predicted $\mathrm{Var}_{\mathrm{U}}/\mathrm{Var}_{\mathrm{B}}$", labelpad=2.5)
ax.set_ylabel(r"Measured $\mathrm{Var}_{\mathrm{U}}/\mathrm{Var}_{\mathrm{B}}$")
ax.text(0.97, 0.20, "Predicted:\n" r"$(1+\kappa)\,n(n-3)\,/\,[(n-2)(n+1)]$", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7, color=INK, linespacing=1.3)
order_f = sorted(pts.keys(), key=lambda fm: -float(np.median(np.array(pts[fm])[:, 0])))  # by predicted ratio
hs, ls_ = ax.get_legend_handles_labels()
lab2h = dict(zip(ls_, hs))
names = [FAMILY_STYLE[f][0] for f in order_f if FAMILY_STYLE[f][0] in lab2h]
ax.legend([lab2h[nm] for nm in names], [nm.replace(" (incl. reset variants)", "") for nm in names],  # caption: incl. reset variants
          loc="upper left", handletextpad=0.2, borderaxespad=0.2, handlelength=1.0,
          markerscale=1.3, labelspacing=0.2, fontsize=7, frameon=True, facecolor="white", edgecolor="none",
          framealpha=1.0)
for h in ax.get_legend().legend_handles:
    h.set_alpha(1.0)

# shared ylabel offset and panel letters (bold 9 pt, left-aligned at the same offset from each panel's spine)
for ax, letter in ((axA, "a"), (axB, "b")):
    ax.yaxis.set_label_coords(-YLAB_OFF / S, 0.5)
    ax.text(-LETTER_OFF / S, 1.0 + 0.07 / S, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="left", va="bottom", color=INK)
save(fig, os.path.join(outdir, "fig_laws"))
print("ok", len(rows), f"{len(ratios):,} cells, {nsyn} synthetic + {nreal} real, ratio {ratios.mean():.3f} +- {ratios.std():.3f},",
      f"{100 * inside:.1f}% within +-2 MC s.e. ({2 * se:.4f})")
