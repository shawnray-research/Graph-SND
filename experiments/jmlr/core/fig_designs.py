import sys, os, csv, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator, LogLocator

setup()
# per-figure overrides of the shared style (figstyle.py is left untouched)
plt.rcParams.update({
    "axes.edgecolor": "#888888", "axes.linewidth": 0.6,
    "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
    "xtick.color": "#555555", "ytick.color": "#555555",
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
})

outdir, f = sys.argv[1], sys.argv[2]  # Table tab:e2 and this figure use e2_synth_v2.csv
rows = [r for r in csv.DictReader(open(f)) if r["source"] == "synthetic"]
fams = ["near_simplex", "hetero_scale", "one_outlier", "two_roles", "unequal_roles", "ring"]
designs = ["uniform", "bernoulli_ht", "bernoulli_ratio", "agent_subsample", "landmark",
           "local_knn", "ring_fixed", "expander_fixed", "round_robin"]

# Series styling. Two pairs of series coincide: Bernoulli (ratio) lies on uniform pairs in every
# panel, and the fixed random expander lies on the balanced design. In each pair the first series is
# drawn underneath with a wider line and a larger marker, the second on top with a thinner line of a
# dotted pattern and a smaller marker of a different shape, so both stay visible (no jitter); the
# uniform squares are slightly larger than the balanced circles, so their corners show where they coincide.
# Both gray series are drawn behind the blue ones (where kappa ~ 0 they coincide with the balanced design).
#            label, color, linestyle, marker, open, lw, markersize, zorder
STYLE = {
    "uniform":         ("Uniform pairs",             UNIFORM_LIGHT, "-",  "s", False, 1.6, 4.0, 2.0),
    "bernoulli_ratio": ("Bernoulli (ratio)",         "#C6C6C6", (0, (1, 1.3)), "D", False, 1.2, 2.6, 2.1),
    "bernoulli_ht":    ("Bernoulli (HT)",            "#999999", (0, (3.2, 1.6)), "^", False, 1.2, 3.6, 2.75),
    "agent_subsample": ("Agent subsampling",         "#CC79A7", "-",  "P", False, 1.2, 3.8, 2.6),
    "landmark":        ("Landmarks",                 "#882255", "-",  "X", False, 1.2, 3.8, 2.7),
    "local_knn":       ("Nearest neighbors (fixed)", "#D55E00", "-",  "o", True,  1.2, 3.4, 2.8),
    "ring_fixed":      ("Fixed ring",                "#CC6677", (0, (4.0, 1.4, 1.0, 1.4)), "s", True, 1.2, 3.1, 2.9),
    "round_robin":     ("Balanced",                  "#0072B2", "-",  "o", False, 1.6, 3.5, 4.0),
    "expander_fixed":  ("Random expander (fixed)",   "#56B4E9", (0, (1, 1.3)), "v", False, 1.2, 2.5, 4.1),
}

W, H = 6.0, 3.85
LEFT, RIGHT, BOTTOM, TOP = 0.075, 0.995, 0.0975, 0.832
fig, axes = plt.subplots(2, 3, figsize=(W, H), sharex=True,
                         gridspec_kw={"hspace": 0.31, "wspace": 0.30})
handles, ylims, xmax = {}, [], 0.0
for ax, fam in zip(axes.ravel(), fams):
    sub = [r for r in rows if r["family"] == fam]
    kap = np.mean([float(r["kappa"]) for r in sub])
    ylo, yhi = np.inf, 0.0
    for d in designs:
        lab, col, ls, mk, op, lw, ms, z = STYLE[d]
        agg = collections.defaultdict(list)
        for r in sub:
            if r["design"] == d:
                agg[int(r["k"])].append((float(r["m"]), float(r["rel_rmse"])))
        ks = sorted(agg)
        if not ks:
            continue
        x = np.array([np.mean([a for a, b in agg[k]]) for k in ks])
        # RMS over matrices (for fixed designs: RMS of the deterministic errors)
        y = np.array([np.sqrt(np.mean([b ** 2 for a, b in agg[k]])) for k in ks])
        y = 100 * np.maximum(y, 1e-6)
        # landmarks: k=1 and k=2 are the same one-pivot design (m=199); draw one point, not two stacked markers
        keep = np.r_[True, np.diff(x) > 0]
        x, y = x[keep], y[keep]
        ylo, yhi = min(ylo, y.min()), max(yhi, y.max())
        xmax = max(xmax, x.max())
        if op:   # open marker: white face, coloured edge
            mkw = dict(markerfacecolor="white", markeredgecolor=col, markeredgewidth=0.8)
        else:    # filled marker with a thin white edge so it separates from lines and other markers
            mkw = dict(markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.4)
        h, = ax.plot(x, y, ls=ls, color=col, lw=lw, marker=mk, markersize=ms, zorder=z,
                     label=lab, solid_capstyle="round", dash_capstyle="butt", **mkw)
        handles[d] = h
    ax.set_xscale("log"); ax.set_yscale("log")
    e0, e1 = int(np.floor(np.log10(ylo))), int(np.ceil(np.log10(yhi)))
    ylims.append((ax, e0, e1))
    ax.yaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(e0, e1 + 1)]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(FixedLocator([100, 200, 400, 800, 1600]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", color="#E6E6E6", lw=0.6)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    name = {"ring": "Continuum"}.get(fam, FAMILY_STYLE[fam][0])
    kstr = f"{kap:.0f}" if kap >= 10 else (f"{kap:.2f}" if kap >= 0.01 else f"{kap:.1g}")
    ax.set_title(name, fontsize=8.5, pad=4)
    # kappa sits in the bottom-left corner, which is free of data in all six panels
    ax.annotate(f"$\\kappa={kstr}$", xy=(0, 0), xycoords="axes fraction", xytext=(4, 5.6),
                textcoords="offset points", ha="left", va="bottom", fontsize=7.5, zorder=10,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))

fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP)


def set_limits():
    """Decade ticks at both ends; the limits extend MARGIN points beyond the end decades
    so that no marker touches or crosses the frame."""
    MARGIN = 3.2  # points
    for ax, e0, e1 in ylims:
        hpt = ax.get_position().height * H * 72
        n = e1 - e0
        m = MARGIN * n / (hpt - 2 * MARGIN)
        ax.set_ylim(10.0 ** (e0 - m), 10.0 ** (e1 + m))
    ax = axes[0, 0]
    wpt = ax.get_position().width * W * 72
    n = np.log10(xmax / 100.0)
    m = 3.6 * n / (wpt - 2 * 3.6)
    ax.set_xlim(10.0 ** (2 - m), xmax * 10.0 ** m)


set_limits()
xlab = fig.supxlabel("Number of pairs evaluated (of $n(n-1)/2=19{,}900$ for $n=200$)", fontsize=8,
                     x=0.5, y=0.0, va="bottom")
for ax in axes[:, 0]:
    ax.set_ylabel("Relative RMSE (%)", fontsize=8, labelpad=3)

LETTER_DX = -27  # points from the axes' left edge: identical for every panel
for ax, s in zip(axes.ravel(), "abcdef"):
    ax.annotate(f"({s})", xy=(0, 1), xycoords="axes fraction", xytext=(LETTER_DX, 4),
                textcoords="offset points", ha="left", va="baseline", fontsize=9, fontweight="bold")

# Legend: three one-column blocks, each left-aligned with the panel column beneath it.
groups = [["round_robin", "expander_fixed", "uniform"],
          ["bernoulli_ratio", "bernoulli_ht", "agent_subsample"],
          ["landmark", "local_knn", "ring_fixed"]]
legends = []
for j, g in enumerate(groups):
    x0 = axes[0, j].get_position().x0
    legends.append(fig.legend([handles[d] for d in g], [STYLE[d][0] for d in g], loc="upper left",
               bbox_to_anchor=(x0 + LETTER_DX / 72 / W, 1.0), bbox_transform=fig.transFigure,
               ncol=1, handlelength=2.6, handletextpad=0.6, labelspacing=0.32, borderpad=0.0,
               borderaxespad=0.0, fontsize=7))

# Width control: the saved PDF (tight bbox + 0.02 in pad on both sides) must be exactly 6.00 in wide,
# so move the right edge of the panel grid until the drawn content spans 5.96 in.
xlab.set_x((axes[1, 0].get_position().x0 + axes[1, 2].get_position().x1) / 2)
for _ in range(4):
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    excess = bb.width - (6.0 - 2 * 0.02)
    if abs(excess) < 0.002:
        break
    RIGHT -= excess / W
    fig.subplots_adjust(right=RIGHT)
    set_limits()
    xlab.set_x((axes[1, 0].get_position().x0 + axes[1, 2].get_position().x1) / 2)  # centred on the grid
    for j, lg in enumerate(legends):
        lg.set_bbox_to_anchor((axes[0, j].get_position().x0 + LETTER_DX / 72 / W, 1.0), fig.transFigure)

save(fig, os.path.join(outdir, "fig_designs"))
print("ok")
