"""Figure: pairwise diversity of generated text scored by neural judges.
(a) where judged diversity lives (level vs distinctiveness share, every population);
(b) error vs judge calls with Claude Opus 5.5 as the judge (top axis: API cost);
(c) a Claude-judged diversity curve over six temperatures, one balanced design per point
    (pass the curve file with --curve claude_curve.json; the measured-vs-closed-form check over all
    276 judge cells is reported in the text, see judge_numbers.py)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from matplotlib.lines import Line2D

out = sys.argv[1]
FIGW = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].endswith(".json") else 6.038
argv = sys.argv[2:]
curve_file = argv[argv.index("--curve") + 1] if "--curve" in argv else None
files = [f for f in argv if f.endswith(".json") and f != curve_file]
pops = {}
for f in files:
    for p in json.load(open(f))["populations"]:
        pops[p["name"]] = p
C_BAL, C_UNI, C_UNI_LIGHT, C_HT, C_SUB, C_STAR = "#0072B2", "#7F7F7F", "#AAAAAA", "#999999", "#CC79A7", "#E69F00"
INK = "#222222"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#555555", "ytick.color": "#555555", "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titlepad": 15,
})
fig, axs = plt.subplots(1, 3, figsize=(FIGW, 2.55), gridspec_kw=dict(width_ratios=[1, 1, 1.18]))
fig.subplots_adjust(left=0.08, right=0.985, bottom=0.17, top=0.83, wspace=0.52)


def plain_log(ax, axis, ticks):
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(mt.FixedLocator(ticks))
    a.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:,g}"))
    a.set_minor_locator(mt.NullLocator())


def letter(ax, s):
    ax.text(-0.27, 1.075, f"({s})", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold",
            color=INK)
    ax.title.set_position((0.5, 1.0))


# ---------------- (a) level vs distinctiveness share
ax = axs[0]
TCOL = {0.5: "#BDBDBD", 1.0: "#7A7A7A", 1.5: "#1A1A1A"}
for name, p in pops.items():
    if name.startswith(("claude_", "mixture", "crossmodel", "contaminated")):
        continue
    llama = name.startswith("llama_")
    T = float(name.rsplit("_T", 1)[1])
    if llama:
        ax.plot(p["mu"], 100 * p["distinct_share"], "s", ms=3.6, mfc="none", mec=TCOL[T], mew=1.0, zorder=4)
    else:
        ax.plot(p["mu"], 100 * p["distinct_share"], "o", ms=3.8, mfc=TCOL[T], mec="white", mew=0.4, zorder=3)
COMP = {"contaminated_story_T0.5": ("2% degenerate", (0.70, 87), "left"),
        "crossmodel_story_T1.0": ("Two\nmodels", (0.81, 24.0), "left"),
        "mixture_T1.0": ("Mixed\nprompts", (0.82, 8.0), "left")}
for name, (lab, xy, ha) in COMP.items():
    if name not in pops:
        continue
    p = pops[name]
    pt = (p["mu"], 100 * p["distinct_share"])
    ax.plot(*pt, "D", ms=3.8, mfc="white", mec=INK, mew=0.9, zorder=4)
    ax.annotate(lab, pt, xytext=xy, fontsize=7, color=INK, ha=ha, va="center", linespacing=1.0,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555", shrinkA=1.0, shrinkB=2.5))
for name, dx, dy, ha in (("claude_story_T0.5", -0.06, 9.0, "right"), ("claude_story_T1.0", 0.03, -9.5, "left")):
    if name in pops:
        p = pops[name]
        ax.plot(p["mu"], 100 * p["distinct_share"], "*", ms=7.5, mfc=C_STAR, mec=INK, mew=0.5, zorder=5)
        T = float(name.rsplit("_T", 1)[1])
        ax.annotate(f"$T={T:g}$", (p["mu"], 100 * p["distinct_share"]),
                    xytext=(p["mu"] + dx, 100 * p["distinct_share"] + dy), fontsize=7, color=INK, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555", shrinkA=0.5, shrinkB=4.5))
ax.set_xlim(0.2, 1.09); ax.set_ylim(0, 104)
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel(r"Judged diversity $\mu$")
ax.set_ylabel("Distinctiveness share (%)")
ax.set_title("Where diversity lives")
hT = [Line2D([], [], ls="", marker="o", mfc=TCOL[t], mec="white", mew=0.4, ms=4, label=f"$T={t:g}$") for t in (0.5, 1.0, 1.5)]
legT = ax.legend(handles=hT, loc="upper left", ncol=3, handlelength=0.7, columnspacing=0.6, borderaxespad=0.1,
                 handletextpad=0.25, fontsize=7)
ax.add_artist(legT)
hK = [Line2D([], [], ls="", marker="o", mfc="#7A7A7A", mec="white", mew=0.4, ms=4, label="Qwen"),
      Line2D([], [], ls="", marker="s", mfc="none", mec="#7A7A7A", mew=1.0, ms=3.6, label="Llama"),
      Line2D([], [], ls="", marker="*", mfc=C_STAR, mec=INK, mew=0.5, ms=7, label="Claude"),
      Line2D([], [], ls="", marker="D", mfc="white", mec=INK, mew=0.9, ms=3.6, label="Composite")]
ax.legend(handles=hK, loc="lower left", ncol=1, handlelength=0.7, borderaxespad=0.1, handletextpad=0.3,
          labelspacing=0.2, fontsize=7)
letter(ax, "a")

# ---------------- (b) error vs judge calls, Claude judge
ax = axs[1]
p = pops["claude_story_T1.0"]
calls = np.array([b["judge_calls"] // 2 for b in p["budgets"]])  # one Claude call per pair
usd_per_call = 3.930808 / 2016  # measured: results/jmlr/samples/claude_extra.json
hb = []
SER = (("bernoulli_ht", C_HT, "--", "^", "Bernoulli (HT)"), ("subset", C_SUB, "-", "P", "Sample subset"),
       ("uniform", C_UNI, "-", "s", "Uniform pairs"), ("balanced", C_BAL, "-", "o", "Balanced"))
for dz, col, ls, mk, lab in SER:
    meas = np.array([100 * b[dz]["rel_rmse"] for b in p["budgets"]])
    ax.plot(calls, meas, ls, color=col, lw=1.4, marker=mk, ms=3.8, mec="white", mew=0.4, zorder=3)
    if dz != "subset":
        pred = np.array([100 * b[dz]["rel_sd_pred"] for b in p["budgets"]])
        ax.plot(calls, pred, ":", color="black", lw=0.9, zorder=4)
    hb.append(Line2D([], [], ls=ls, color=col, lw=1.4, marker=mk, ms=3.8, mec="white", mew=0.4, label=lab))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(calls[0] / 1.22, calls[-1] * 1.22)
ax.set_ylim(0.12, 30)
plain_log(ax, "x", list(calls)); plain_log(ax, "y", [0.2, 0.5, 1, 2, 5, 10, 20])
ax.grid(True, which="major")
ax.set_xlabel("Judge calls (Claude Opus 5.5)")
ax.set_ylabel("Relative RMSE (%)")
ax.set_title("Error at equal cost")
top = ax.secondary_xaxis("top", functions=(lambda x: x * usd_per_call, lambda y: y / usd_per_call))
top.xaxis.set_major_locator(mt.FixedLocator([c * usd_per_call for c in calls]))
top.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"\\${v:.2f}"))
top.xaxis.set_minor_locator(mt.NullLocator())
top.tick_params(labelsize=7, colors="#555555", labelcolor=INK, length=2.5, width=0.6, pad=1.5)
top.spines["top"].set_color("#888888"); top.spines["top"].set_linewidth(0.6)
hb = hb + [Line2D([], [], ls=":", color="black", lw=0.9, label="Theory")]
ax.legend(handles=hb, loc="lower left", handlelength=1.7, borderaxespad=0.1, handletextpad=0.4, labelspacing=0.15,
          fontsize=7)
ax.text(0.97, 0.97, f"All {p['pairs']:,} pairs:\n\\${p['pairs'] * usd_per_call:.2f}", transform=ax.transAxes,
        ha="right", va="top", fontsize=7, color=INK, linespacing=1.15)
letter(ax, "b")

# ---------------- (c) diversity curve: one balanced design (k = 4 rounds) per temperature, n = 1,000
ax = axs[2]
C = json.load(open(curve_file))
d0 = sorted([d for d in C["designs"] if d["rep"] == 0], key=lambda d: d["temperature"])
T = np.array([d["temperature"] for d in d0])
mu = np.array([d["mu_hat"] for d in d0])
clo = np.array([d["ci95"][0] for d in d0]); chi = np.array([d["ci95"][1] for d in d0])
xs = [0.5, 1.0, 1.5]
if all(f"story_T{t}" in pops for t in xs):
    ax.plot(xs, [pops[f"story_T{t}"]["mu"] for t in xs], "--", color=INK, lw=1.1, marker="x", ms=3.6, mfc="white",
            mec=INK, mew=0.9, zorder=2, label="Cross-encoder")
ax.plot(T, mu, "-", color=C_BAL, lw=1.4, zorder=3)
ax.errorbar(T, mu, yerr=[mu - clo, chi - mu], fmt="o", color=C_BAL, ms=3.8, mec="white", mew=0.4,
            elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=4, label="Claude, balanced")
for d in [d for d in C["designs"] if d["rep"] == 1]:
    m = d["mu_hat"]
    ax.errorbar([d["temperature"] + 0.06], [m], yerr=[[m - d["ci95"][0]], [d["ci95"][1] - m]], fmt="o", mfc="white",
                mec=C_BAL, color=C_BAL, ms=3.4, mew=0.9, elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=4,
                label="Repeat design")
cost = np.mean([d["cost_usd"] for d in d0])  # per point; all pairs would cost 499,500 / 2,000 times more
ax.text(0.97, 0.03, f"{d0[0]['calls']:,} calls per point\n(all pairs: {499500:,})",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK, linespacing=1.15)
ax.set_xlabel("Sampling temperature")
ax.set_ylabel(r"Judged diversity $\mu$")
ax.set_xticks(T); ax.set_xticklabels([f"{t:g}" for t in T])
ax.set_xlim(0.12, 1.64); ax.set_ylim(0.38, 0.92)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_title(f"Curve at \\${cost:.0f} per point")
hh, ll = ax.get_legend_handles_labels()
order = [ll.index(x) for x in ("Claude, balanced", "Repeat design", "Cross-encoder") if x in ll]
ax.legend([hh[i] for i in order], [ll[i] for i in order], loc="upper left", handlelength=1.5, borderaxespad=0.1, labelspacing=0.25, fontsize=7,
          handletextpad=0.4)
letter(ax, "c")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("saved", out)
