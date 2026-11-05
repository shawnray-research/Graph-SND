"""Figure: cost and accuracy of all-pairs vs balanced SND as n grows.

Usage: python fig_scale.py TIMING_JSON OUT_STEM ACCURACY_JSON
"""
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

src, out = sys.argv[1], sys.argv[2]
rows = json.load(open(src))
acc = json.load(open(sys.argv[3]))
INK, MUTED = "#222222", "#555555"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": "#888888", "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": "#555555", "ytick.color": "#555555", "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "legend.frameon": False, "pdf.fonttype": 42,
})
ks = [4, 8, 16]
# balanced design = blue; k encoded by shade (light -> dark, k=8 is the default balanced blue)
# and, for grayscale, by marker shape
shades = {4: "#6BAED6", 8: "#0072B2", 16: "#08306B"}
marks = {4: ("s", 3.0), 8: ("o", 3.3), 16: ("D", 2.8)}  # (marker, size): visually matched areas


def metric_bound(n, k):
    """Metric-universal bound v_{n,k} on Var(T)/mu^2 (Theorem metric; tight form).

    Computed here rather than read from the `k*_bound` fields of the accuracy JSON,
    which hold the earlier, looser 2(n-1-k)/((n-2)(n-3)k).
    """
    return 2.0 * (n - 2) * (n - 1 - k) / (n ** 2 * (n - 3) * k)


def plain_log_ticks(axis, ticks):
    axis.set_major_locator(mt.FixedLocator(ticks))
    axis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:,g}"))
    axis.set_minor_formatter(mt.NullFormatter())


DOT = dict(lw=1.2, ls=(0, (0.01, 2.1)), dash_capstyle="round")  # round dots; every line ends on a whole dot


def kmark(k):
    m, s = marks[k]
    return dict(marker=m, ms=s, mfc=shades[k], mec="white", mew=0.4)


FW, FH = 6.0, 2.44          # figure canvas (in); final size is set by the tight bbox below
AX_Y, AX_H = 0.43, 1.68     # axes bottom / height (in)
LEFT, GAP = 0.56, 0.80      # left edge of (a), gap between (a) and (b) (in)
TARGET_W = 6.00 - 2 * 0.02  # tight-bbox content width so that the saved PDF is 6.00 in wide
fig = plt.figure(figsize=(FW, FH))
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0, 0, 1, 1])

# ---------------- (a) cost ----------------
nf = [r["n"] for r in rows if "t_full_s" in r]
tf = [r["t_full_s"] * 1e3 for r in rows if "t_full_s" in r]
for k in reversed(ks):  # dark (k=16, highest) drawn first, light on top
    n_ = [r["n"] for r in rows]
    t_ = [r["t_bal_k%d_s" % k] * 1e3 for r in rows]
    ax1.loglog(n_, t_, "-", color=shades[k], lw=1.3, zorder=3, **kmark(k), label=f"$k={k}$")
ax1.loglog(nf, tf, "-o", color="black", ms=3.3, mec="white", mew=0.4, lw=1.5, zorder=4)
ax1.annotate("All pairs", xy=(nf[-1], tf[-1]), xytext=(-9, -2), textcoords="offset points",
             ha="right", va="center", fontsize=7, color=INK)
# legend rows in the vertical order of the curves (k=16 on top); white, frameless box over the grid
LEG_BOX = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=1.0, fancybox=False)
# (box flush with the axes' right and bottom edges, so no grid-line stubs show around it; drawn below the
# spines, which therefore stay on top of it; no data enter this corner)
leg1 = ax1.legend(*ax1.get_legend_handles_labels(), title="Balanced", loc="lower right",
                  bbox_to_anchor=(1.004, 0.0), handlelength=2.4, handletextpad=0.5, labelspacing=0.3,
                  borderaxespad=0.0, borderpad=0.25, title_fontsize=7, alignment="left", **LEG_BOX)
leg1.get_title().set_color(INK)
leg1.set_zorder(2.4)
for r in rows:
    if r["n"] == 5000:
        sp = r["t_full_s"] / r["t_bal_k8_s"]
        sp_full_ms, sp_bal_ms = 1e3 * r["t_full_s"], 1e3 * r["t_bal_k8_s"]
# double arrow between the all-pairs and the balanced (k=8) points at n = 5,000; it stops short of the
# markers (the k=16 marker sits 2.8 pt above the k=8 one, hence the larger lower shrink)
ax1.annotate("", xy=(5000, sp_bal_ms), xytext=(5000, sp_full_ms),
             arrowprops=dict(arrowstyle="<|-|>,head_length=0.35,head_width=0.16", lw=0.7, color=MUTED,
                             shrinkA=3.2, shrinkB=6.2), zorder=2)
ax1.annotate(f"{sp:.0f}$\\times$\n($k=8$)", xy=(5000, math.sqrt(sp_bal_ms * sp_full_ms)), xytext=(4, 0),
             textcoords="offset points", ha="left", va="center", fontsize=7, color=INK, linespacing=1.25)
ax1.set_xlabel("Population size $n$")
ax1.set_ylabel("Time per evaluation (ms)")
ax1.set_title("Cost of one evaluation (GPU)", pad=4)
ax1.set_xlim(42, 14000)
ax1.set_ylim(0.38, 3800)  # 4 decades
plain_log_ticks(ax1.xaxis, [100, 300, 1000, 3000, 10000])
plain_log_ticks(ax1.yaxis, [1, 10, 100, 1000])

# ---------------- (b) accuracy ----------------
for k in ks:
    n_ = [r["n"] for r in acc]
    meas = [100 * r["k%d_meas" % k] for r in acc]
    se = [100 * r["k%d_meas_se" % k] for r in acc]
    pred = [100 * r["k%d_pred" % k] for r in acc]
    bound = [100 * math.sqrt(metric_bound(n, k)) for n in n_]
    ax2.loglog(n_, pred, "-", color=shades[k], lw=1.3, zorder=2)
    ax2.errorbar(n_, meas, yerr=se, fmt="none", ecolor=shades[k], elinewidth=0.8, capsize=1.2, capthick=0.6,
                 zorder=3)
    ax2.loglog(n_, meas, ls="none", zorder=4, **kmark(k))
    ax2.loglog(n_, bound, color=shades[k], zorder=1, **DOT)
    ax2.annotate(f"$k={k}$", xy=(n_[0], meas[0]), xytext=(-5, 0), textcoords="offset points",
                 va="center", ha="right", fontsize=7, color=INK)
mk_proxy = tuple(Line2D([], [], ls="none", color=MUTED, **dict(kmark(k), mfc=MUTED)) for k in ks)
h_th = Line2D([], [], ls="-", color=MUTED, lw=1.3)
h_bd = Line2D([], [], color=MUTED, **DOT)
ax2.legend([mk_proxy, h_th, h_bd], ["Measured", "Theory", "Metric bound"],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, loc="upper right",
           bbox_to_anchor=(1.004, 1.004), handlelength=2.3, handletextpad=0.45, labelspacing=0.25,
           borderaxespad=0.0, borderpad=0.25, **LEG_BOX)
ax2.set_xlabel("Population size $n$")
ax2.set_ylabel("Relative s.d. of estimate (%)")
ax2.set_title("Accuracy of the balanced estimate", pad=4)
ax2.set_xlim(16, 6200)  # room left of n=50 for the direct k labels
ax2.set_ylim(0.03, 16)
plain_log_ticks(ax2.xaxis, [30, 100, 300, 1000, 3000])
plain_log_ticks(ax2.yaxis, [0.1, 1, 10])

for ax in (ax1, ax2):
    ax.grid(True, which="major", color="#E6E6E6", lw=0.6)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    for sp_ in ["top", "right"]:
        ax.spines[sp_].set_visible(False)
    for sp_ in ["left", "bottom"]:
        ax.spines[sp_].set_color("#888888")
        ax.spines[sp_].set_linewidth(0.6)


def place(w):
    ax1.set_position([LEFT / FW, AX_Y / FH, w / FW, AX_H / FH])
    ax2.set_position([(LEFT + w + GAP) / FW, AX_Y / FH, w / FW, AX_H / FH])


# equal panel widths, solved so that the tight bounding box is exactly TARGET_W wide
w = (FW - LEFT - GAP - 0.1) / 2
for _ in range(4):
    place(w)
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w += (TARGET_W - bb.width) / 2
place(w)
fig.canvas.draw()

# panel letters: bold, outside the axes, same offset left of each axes, on the titles' baseline
r_ = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
off = max(ax.get_window_extent(r_).x0 - ax.yaxis.label.get_window_extent(r_).x0 for ax in (ax1, ax2))
for ax, lab in ((ax1, "(a)"), (ax2, "(b)")):
    x0 = ax.get_window_extent(r_).x0 - off
    y_base = ax.title.get_transform().transform(ax.title.get_position())[1]
    fig.text(*inv.transform((x0, y_base)), lab, fontsize=9, fontweight="bold", ha="left", va="baseline",
             color=INK)
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("speedup n=5000 k=8:", sp)
