"""Figure: simulated diversity control with drifting policies, 10 populations per setting."""
import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from matplotlib.lines import Line2D

src_dir, out = sys.argv[1], sys.argv[2]
runs = [json.load(open(f)) for f in sorted(glob.glob(src_dir + "/seed*.json"))]
C = {"balanced": "#0072B2", "uniform": "#7F7F7F", "bernoulli_ratio": "#B3B3B3", "bernoulli_ht": "#999999"}
C_UNI_TRACE = C["uniform"]  # same grey as the uniform-pairs markers in (b); the blue trace is drawn on top
LAB = {"balanced": "Balanced", "uniform": "Uniform pairs", "bernoulli_ratio": "Bernoulli (ratio)",
       "bernoulli_ht": "Bernoulli (HT)"}
MK = {"balanced": "o", "uniform": "s", "bernoulli_ratio": "D", "bernoulli_ht": "^"}
MS = {"balanced": 4.0, "uniform": 3.7, "bernoulli_ratio": 3.6, "bernoulli_ht": 4.0}  # similar visual weight
INK, SPINE, TICK, GRID = "#222222", "#888888", "#555555", "#E6E6E6"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK,
    "axes.titlecolor": INK, "axes.edgecolor": SPINE, "axes.linewidth": 0.6,
    "xtick.color": TICK, "ytick.color": TICK, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 3.0,
    "legend.frameon": False, "pdf.fonttype": 42,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.103, 2.62), gridspec_kw={"width_ratios": [1.12, 1.0]})
fig.subplots_adjust(left=0.08, right=0.995, bottom=0.25, top=0.9, wspace=0.27)

# (a) one trace of the realized/target ratio
first = [r for r in runs[0] if r["n"] == 500 and r["setting"] == "heterogeneous"][0]
t = np.arange(len(first["trace_uniform"]))
ax1.axhline(0, color="black", lw=0.8, zorder=1.5)
ax1.plot(t, 100 * np.array(first["trace_uniform"]), color=C_UNI_TRACE, lw=1.2, zorder=2,
         solid_joinstyle="round")
ax1.plot(t, 100 * np.array(first["trace_balanced"]), color=C["balanced"], lw=1.4, zorder=3,
         solid_joinstyle="round")
ax1.set_xlabel("Control step (first 150 of 300)")
ax1.set_ylabel("Realized/target $-$ 1 (%)")
ax1.set_title("$n=500$, unequal scales, $2{,}000$ pairs per step", pad=4)
ax1.set_xlim(-4, 153)
ax1.set_ylim(-4.5, 7.0)
ax1.xaxis.set_major_locator(mt.MultipleLocator(25))
ax1.yaxis.set_major_locator(mt.MultipleLocator(2))
h_a = [Line2D([], [], color=C["balanced"], lw=1.4), Line2D([], [], color=C_UNI_TRACE, lw=1.2)]
ax1.legend(h_a, [LAB["balanced"], LAB["uniform"]], loc="upper center", ncol=2, handlelength=1.6,
           columnspacing=1.2, handletextpad=0.5, borderaxespad=0.15, borderpad=0.25,
           frameon=True, facecolor="white", edgecolor="none", framealpha=1.0, fancybox=False)

# (b) RMS tracking error per setting and design
settings = [(50, "homogeneous"), (50, "heterogeneous"), (500, "homogeneous"), (500, "heterogeneous")]
designs = ["bernoulli_ht", "bernoulli_ratio", "uniform", "balanced"]
summary = {}
for xi, (n, s) in enumerate(settings):
    for j, d in enumerate(designs):
        meas = np.array([[r for r in run if r["n"] == n and r["setting"] == s][0]["rms_" + d] for run in runs])
        pred = np.array([[r for r in run if r["n"] == n and r["setting"] == s][0]["pred_" + d] for run in runs])
        mean_track = np.array([[r for r in run if r["n"] == n and r["setting"] == s][0]["mean_" + d] for run in runs])
        x = xi + (j - 1.5) * 0.21
        m_, se = 100 * meas.mean(), 100 * meas.std(ddof=1) / np.sqrt(len(meas))
        ax2.plot([x - 0.095, x + 0.095], [100 * pred.mean()] * 2, color="black", lw=1.2, zorder=6,
                 solid_capstyle="butt")
        # no caps: every s.e. bar is shorter than its marker's half-height (verified), and caps
        # would protrude beside the apexes of the triangles and diamonds
        eb = ax2.errorbar([x], [m_], yerr=[se], fmt=MK[d], color=C[d], ms=MS[d], mec="white", mew=0.4,
                          elinewidth=0.8, ecolor=C[d], capsize=0, zorder=5)
        eb[2][0].set_capstyle("butt")
        summary[f"{n}-{s}-{d}"] = dict(rms_mean=m_, rms_se=se, pred=100 * pred.mean(),
                                       bias_mean=100 * mean_track.mean(),
                                       bias_se=100 * mean_track.std(ddof=1) / np.sqrt(len(mean_track)))
ax2.set_yscale("log")
ax2.set_ylim(0.2, 30)
ax2.yaxis.set_major_locator(mt.FixedLocator([0.3, 1, 3, 10, 30]))
ax2.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
# minor ticks strictly inside the limits (none at the 0.2 corner)
ax2.yaxis.set_minor_locator(mt.FixedLocator([b * f for b in (0.1, 1, 10) for f in (2, 4, 5, 6, 7, 8, 9)
                                             if 0.2 < b * f < 30]))
ax2.yaxis.set_minor_formatter(mt.NullFormatter())
ax2.set_xlim(-0.5, 3.5)
ax2.set_xticks(range(4))
ax2.set_xticklabels(["Equal\nscales", "Unequal\nscales"] * 2, linespacing=1.05)
ax2.set_ylabel("RMS tracking error (%)")
ax2.set_title("Measured vs. predicted", pad=4)
h_b = [Line2D([], [], ls="none", marker=MK[d], color=C[d], ms=MS[d], mec="white", mew=0.4)
       for d in designs] + [Line2D([], [], color="black", lw=1.1, solid_capstyle="butt")]
ax2.legend(h_b, [LAB[d] for d in designs] + ["Prediction"], loc="upper right", handlelength=1.3,
           handletextpad=0.5, borderaxespad=0.15, borderpad=0.3, labelspacing=0.3,
           frameon=True, facecolor="white", edgecolor="none", framealpha=1.0, fancybox=False)
# note sits between the 0.3 and 1 gridlines, left of the n = 500 groups
ax2.text(-0.44, np.sqrt(0.3 * 1.0), "The s.e. bars are smaller\nthan the markers", ha="left", va="center",
         fontsize=7, color=INK, linespacing=1.15)

for ax in (ax1, ax2):
    ax.grid(True, axis="y" if ax is ax2 else "both", which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp_ in ["top", "right"]:
        ax.spines[sp_].set_visible(False)

fig.canvas.draw()
inv = fig.transFigure.inverted()
# group labels n = 50 / n = 500 under the (b) tick labels, with thin brackets
tl_bottom = min(lab.get_window_extent().y0 for lab in ax2.get_xticklabels())
pt = fig.dpi / 72.0
y_br = inv.transform((0, tl_bottom - 2.5 * pt))[1]
y_gl = inv.transform((0, tl_bottom - 4.0 * pt))[1]
for (x0, x1), lab in (((-0.4, 1.4), "$n=50$"), ((1.6, 3.4), "$n=500$")):
    fx0 = inv.transform(ax2.transData.transform((x0, 1)))[0]
    fx1 = inv.transform(ax2.transData.transform((x1, 1)))[0]
    tick_h = inv.transform((0, 2.0 * pt))[1] - inv.transform((0, 0))[1]
    fig.add_artist(Line2D([fx0, fx0, fx1, fx1], [y_br + tick_h, y_br, y_br, y_br + tick_h],
                          color=SPINE, lw=0.6, solid_joinstyle="miter"))
    fig.text((fx0 + fx1) / 2, y_gl, lab, ha="center", va="top", fontsize=8)
# panel letters: same offset left of each axes, baseline-aligned with the titles
off = max(ax.get_window_extent().x0 - ax.yaxis.label.get_window_extent().x0 for ax in (ax1, ax2))
for ax, lab in ((ax1, "(a)"), (ax2, "(b)")):
    x_disp = ax.get_window_extent().x0 - off
    y_disp = ax.title.get_transform().transform(ax.title.get_position())[1]
    fx, fy = inv.transform((x_disp, y_disp))
    fig.text(fx, fy, lab, fontsize=9, fontweight="bold", ha="left", va="baseline")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)
json.dump(summary, open(out + "_summary.json", "w"), indent=1)
for k, v in summary.items():
    print(k, {a: round(b, 3) for a, b in v.items()})
