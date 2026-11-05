import sys, os, csv, io, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
setup()
outdir, f = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(f)))
ks = [2, 4, 8, 16]
def agg(key, src=None, fn=np.mean):
    out = []
    for k in ks:
        v = np.array([float(r[key]) for r in rows if int(r["k"]) == k and (src is None or
                      (r["source"] == src and "zoo" not in r["family"]))])
        out.append((fn(v), v.min(), v.max()))
    return np.array(out)

# ---- per-figure style overrides (figstyle.py is shared and left untouched) ----
SPINE, TICK, GRID = "#888888", "#555555", "#E6E6E6"
LW, MS, ELW = 1.3, 3.6, 0.8                 # data lines, markers, error bars (pt)
TARGET_W = 6.00                              # final PDF width (in), = \textwidth
plt.rcParams.update({"grid.linewidth": 0.6})

BLUE, GRAY = DESIGN_STYLE["round_robin"][1], UNIFORM_LIGHT  # uniform lighter: it overlaps the blue series
SERIES = {"rr": ("Balanced (round-robin)", BLUE, "o"), "u": (DESIGN_STYLE["uniform"][0], GRAY, "s")}
# legends sit over gridlines: white box without frame, reaching just past the axes' right edge (no grid stubs)
# and drawn below the spines (zorder 2.4) so that the bottom spine stays intact
LEG = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=1, borderpad=0.3, borderaxespad=0)
REF = dict(color="black", lw=0.9, ls=":", zorder=1.5)  # reference lines dotted: dashes mark k=16


def kline(ax, x, y, col, mk, z=3):
    """Solid line and filled markers for k <= 8; the k=16 segment dashed with an open marker,
    because only the tables with n >= 32 admit k=16 (13 of 26; 6 of 19 VMAS tables)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ax.plot(x[:3], y[:3], color=col, lw=LW, ls="-", solid_capstyle="butt", zorder=z)
    ax.plot(x[2:], y[2:], color=col, lw=LW, ls=(0, (3, 1.6)), dash_capstyle="butt", zorder=z)
    ax.plot(x[:3], y[:3], ls="none", marker=mk, ms=MS, markerfacecolor=col, markeredgecolor="white",
            markeredgewidth=0.4, zorder=z + 0.5)
    ax.plot(x[3:], y[3:], ls="none", marker=mk, ms=MS - 0.3, markerfacecolor="white", markeredgecolor=col,
            markeredgewidth=0.9, zorder=z + 0.5)


def series_handles():  # labels wrapped to fit inside the narrow panels
    return [Line2D([], [], color=c, lw=LW, marker=m, ms=MS, markerfacecolor=c, markeredgecolor="white",
                   markeredgewidth=0.4) for _, c, m in SERIES.values()], \
           [lab.replace(" (", "\n(") for lab, _, _ in SERIES.values()]


def style(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(which="major", direction="out", length=2.5, width=0.6, color=TICK, labelcolor=INK, pad=2)
    ax.tick_params(which="minor", direction="out", length=1.5, width=0.5, color=TICK)
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.grid(False, which="minor")
    ax.set_axisbelow(True)


def kaxis(ax):
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(2 / 1.19, 16 * 1.19)          # symmetric log margins; no marker or cap clipped
    ax.set_xlabel("Comparisons per individual $k$", labelpad=2)


H = 2.30
fig = plt.figure(figsize=(TARGET_W + 1.0, H))     # oversized canvas; the saved PDF is cropped tight
axes = [fig.add_axes([0.1, 0.1, 0.2, 0.7]) for _ in range(3)]
for ax in axes:
    style(ax)
# (a) coverage (mean and range over tables)
ax = axes[0]
for s, off in [("rr", -0.06), ("u", 0.06)]:  # side by side, so that the k=16 markers do not touch
    lab, col, mk = SERIES[s]
    a = agg("cover_" + s)
    x = np.array(ks) * (1 + off)
    ax.errorbar(x, a[:, 0], yerr=[a[:, 0] - a[:, 1], a[:, 2] - a[:, 0]], fmt="none", ecolor=col,
                elinewidth=ELW, capsize=1.6, capthick=ELW, zorder=2 if s == "u" else 2.2)
    kline(ax, x, a[:, 0], col, mk, z=3 if s == "u" else 4)
ax.axhline(0.95, **REF)
kaxis(ax)
ax.set_ylim(0.852, 0.974); ax.yaxis.set_major_locator(FixedLocator(np.arange(0.86, 0.9601, 0.02)))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.set_ylabel("Coverage of 95% CI")
leg_a = ax.legend(*series_handles(), loc="lower right", bbox_to_anchor=(1.02, 0), handlelength=1.4, handletextpad=0.5,
                  labelspacing=0.35, **LEG)
leg_a.set_zorder(2.4)
# (b) CI width ratio, median and range over tables
ax = axes[1]
w = []
for k in ks:
    v = np.array([float(r["width_rr"]) / float(r["width_u"]) for r in rows if int(r["k"]) == k])
    w.append((np.median(v), v.min(), v.max()))
w = np.array(w)
ax.fill_between(ks[:3], w[:3, 1], w[:3, 2], color=BLUE, alpha=0.17, lw=0, zorder=1)
ax.fill_between(ks[2:], w[2:, 1], w[2:, 2], color=BLUE, alpha=0.08, lw=0, zorder=1)  # k=16: fewer tables
kline(ax, ks, w[:, 0], BLUE, "o")
ax.axhline(1, **REF)
kaxis(ax); ax.set_yscale("log")
plain_log_ticks(ax.yaxis, [0.1, 0.3, 1, 3])
ax.set_ylim(0.08, 5)
ax.set_ylabel("CI width ratio (bal./unif.)")
# (c) Spearman correlation on the real VMAS tables
ax = axes[2]
for s in ["u", "rr"]:
    lab, col, mk = SERIES[s]
    a = agg("spearman_" + s, src="real")
    kline(ax, ks, a[:, 0], col, mk, z=3 if s == "u" else 4)
kaxis(ax); ax.set_ylim(0.5, 1.0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
ax.set_ylabel(r"Spearman($\hat a$, $a$), VMAS teams")
leg_c = ax.legend(*series_handles(), loc="lower right", bbox_to_anchor=(1.02, 0), handlelength=1.4, handletextpad=0.5,
                  labelspacing=0.35, **LEG)
leg_c.set_zorder(2.4)

# ---- layout in inches: equal axes widths, identical y-label offset, identical gaps between panels ----
FW = fig.get_figwidth()
B, TOP = 0.36, 0.20          # bottom: tick labels + x label; top: room for the panel letters
YLAB = 0.315                 # y-label anchor (its right edge) left of the spine, same in every panel
YLAB_H = 0.125               # extent of the rotated 8 pt y label
GAP = 0.30                   # spine-free right edge of a panel -> left edge of the next panel's y label
LET_DX, LET_DY = YLAB + YLAB_H, 0.07  # letters: flush with the y labels' left edge, baseline above the top
ax_h = H - B - TOP

def layout(axw):
    x = 0.6
    for ax in axes:
        ax.set_position([x / FW, B / H, axw / FW, ax_h / H])
        ax.yaxis.set_label_coords(-YLAB / axw, 0.5)
        x += axw + GAP + YLAB + YLAB_H

letters = [fig.text(0, 0, f"({s})", fontsize=9, fontweight="bold", ha="left", va="baseline", color=INK)
           for s in "abc"]  # figure-fraction coordinates, so that the tight crop moves them with the axes

def measure(axw):
    layout(axw)
    for ax, t in zip(axes, letters):
        p = ax.get_position()
        t.set_position((p.x0 - LET_DX / FW, p.y1 + LET_DY / H))
    buf = io.BytesIO()  # measure the PDF exactly as it will be saved (PDF text metrics differ from Agg)
    fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    w, h = re.search(rb"/MediaBox \[ *0 0 ([\d.]+) ([\d.]+) *\]", buf.getvalue()).groups()
    return float(w) / 72, float(h) / 72

axw = 1.40
for _ in range(8):  # widen the three axes equally until the tight-cropped PDF is exactly TARGET_W wide
    width, height = measure(axw)
    if abs(width - TARGET_W) < 1e-4:
        break
    axw += (TARGET_W - width) / 3
width, height = measure(axw)
save(fig, os.path.join(outdir, "fig_ci"))
print("ok", round(width, 4), round(height, 4), round(axw, 4))
