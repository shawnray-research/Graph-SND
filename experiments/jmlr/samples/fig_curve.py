"""Figure: Claude-judged diversity of 1,000 LM responses per temperature from one balanced design each,
with the all-pairs cross-encoder diversity of separate 300-response samples for reference."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
CE = {p["name"]: p for p in json.load(open(sys.argv[3]))["populations"]} if len(sys.argv) > 3 else {}
FIGW = float(sys.argv[4]) if len(sys.argv) > 4 else 4.054
C = json.load(open(src))
C_BAL, INK = "#0072B2", "#222222"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#555555", "ytick.color": "#555555", "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(FIGW, 2.45))
d0 = sorted([d for d in C["designs"] if d["rep"] == 0], key=lambda d: d["temperature"])
T = np.array([d["temperature"] for d in d0])
mu = np.array([d["mu_hat"] for d in d0])
lo = np.array([d["ci95"][0] for d in d0]); hi = np.array([d["ci95"][1] for d in d0])
if CE:
    xs = [0.5, 1.0, 1.5]
    ax.plot(xs, [CE[f"story_T{t}"]["mu"] for t in xs], "--", color=INK, lw=1.2, marker="s", ms=3.6, mfc="white",
            mec=INK, mew=0.9, zorder=2, label="Cross-encoder, all pairs ($n=300$)")
ax.plot(T, mu, "-", color=C_BAL, lw=1.4, zorder=3)
ax.errorbar(T, mu, yerr=[mu - lo, hi - mu], fmt="o", color=C_BAL, ms=3.8, mec="white", mew=0.4,
            elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4, label="Claude, one balanced design")
for d in [d for d in C["designs"] if d["rep"] == 1]:
    m = d["mu_hat"]
    ax.errorbar([d["temperature"] + 0.05], [m], yerr=[[m - d["ci95"][0]], [d["ci95"][1] - m]], fmt="o", mfc="white",
                mec=C_BAL, color=C_BAL, ms=3.6, mew=0.9, elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4,
                label="Independent repeat")
cost = np.mean([d["cost_usd"] for d in d0])
ax.text(0.03, 0.97, f"Claude, $n=1{{,}}000$ per point:\n{d0[0]['calls']:,} calls (\\${cost:.2f}) per point;\n"
        f"all pairs: 499,500 calls (\\${cost * 499500 / d0[0]['calls']:,.0f})", transform=ax.transAxes, ha="left",
        va="top", fontsize=7, color=INK, linespacing=1.15)
ax.set_xlabel("Sampling temperature")
ax.set_ylabel(r"Judged diversity $\widehat{\mu}$")
ax.set_xticks(T); ax.set_xticklabels([f"{t:g}" for t in T])
ax.set_xlim(0.15, 1.6); ax.set_ylim(0.38, 0.92)
ax.legend(loc="lower right", handlelength=1.6, borderaxespad=0.1, labelspacing=0.25, fontsize=7)
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("saved", out)
