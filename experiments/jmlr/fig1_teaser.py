"""Figure 1: anatomy of a toy team, two designs, and error vs budget."""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from matplotlib.lines import Line2D

OUT = sys.argv[1] if len(sys.argv) > 1 else "fig1_teaser"

C_BAL = "#0072B2"
C_UNI = "#7F7F7F"
C_BER = "#999999"
C_SUB = "#CC79A7"  # reddish purple: orange is reserved for distinctiveness
C_DIST = "#E69F00"
C_REL = "#009E73"
INK = "#222222"
MUTED = "#666666"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.6,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.labelcolor": INK,
    "text.color": INK,
    "pdf.fonttype": 42,
})


def anatomy(D):
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    mu = D[iu].mean()
    S = D.sum(1)
    a = (S - (n - 1) * mu) / (n - 2)
    R = D - mu - a[:, None] - a[None, :]
    np.fill_diagonal(R, 0.0)
    return mu, a, R


rng = np.random.default_rng(3)

# ---------- (a) toy team: two roles + one specialist ----------
n = 16
X = rng.normal(scale=0.35, size=(n, 6))
X[: n // 2, 0] += 1.6          # role A
X[n // 2:, 0] -= 1.6           # role B
X[0] = X[0] * 0.3 + np.array([0.0, 4.0, 0, 0, 0, 0])  # specialist (agent 1)
D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
mu, a, R = anatomy(D)
M = a[:, None] + a[None, :]
np.fill_diagonal(M, 0)
L = np.full_like(D, mu)
np.fill_diagonal(L, 0)

# ---------- (c) population for error-vs-budget ----------
n2 = 200
Z = rng.normal(size=(n2, 16))
scale = rng.lognormal(0.0, 0.8, size=(n2, 1))
X2 = Z * scale
D2 = np.sqrt(((X2[:, None] - X2[None]) ** 2).sum(-1))
mu2, a2, R2 = anatomy(D2)
N2 = n2 * (n2 - 1) // 2
iu2 = np.triu_indices(n2, 1)
dvec = D2[iu2]
s_tot = ((dvec - mu2) ** 2).mean()
s_int = (R2[iu2] ** 2).mean()
s_main = s_tot - s_int

ks = [2, 4, 8, 16, 32, 64]
reps = 2000
res = {"bal": [], "uni": [], "sub": [], "ber": []}
pred = {"bal": [], "uni": []}
for k in ks:
    m = n2 * k // 2
    eb, eu, es, eh = [], [], [], []
    base = [(t, (t + s) % n2) for t in range(n2) for s in range(1, k // 2 + 1)]
    bi = np.array([b[0] for b in base])
    bj = np.array([b[1] for b in base])
    s_agents = int(round((1 + math.sqrt(1 + 8 * m)) / 2))
    for _ in range(reps):
        perm = rng.permutation(n2)
        eb.append(D2[perm[bi], perm[bj]].mean())
        sel = rng.choice(N2, m, replace=False)
        eu.append(dvec[sel].mean())
        S = rng.choice(n2, s_agents, replace=False)
        sub = D2[np.ix_(S, S)]
        es.append(sub[np.triu_indices(s_agents, 1)].mean())
        keep = rng.random(N2) < m / N2
        eh.append(dvec[keep].sum() / (N2 * m / N2))
    for key, arr in zip(["bal", "uni", "sub", "ber"], [eb, eu, es, eh]):
        arr = np.asarray(arr)
        res[key].append(np.sqrt(((arr / mu2 - 1) ** 2).mean()))
    pred["bal"].append(math.sqrt(s_int / m * (n2 - 1 - k) / (n2 - 3)) / mu2)
    pred["uni"].append(math.sqrt(s_tot / m * (N2 - m) / (N2 - 1)) / mu2)
ms = [n2 * k // 2 for k in ks]


# ---------- figure (laid out in inches) ----------
import matplotlib.transforms as mtrans
from matplotlib.textpath import TextPath

plt.rcParams.update({"xtick.labelcolor": INK, "ytick.labelcolor": INK,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
                     "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
                     "xtick.direction": "out", "ytick.direction": "out"})
FW, FH = 6.0, 3.72
fig = plt.figure(figsize=(FW, FH))


def inch_axes(x, y, w, h):
    """Axes placed in figure inches; (x, y) is the lower-left corner."""
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])


def ftext(x, y, s, **kw):
    """Figure text placed in inches."""
    return fig.text(x / FW, y / FH, s, **kw)


def fline(xs, ys, **kw):
    """Figure-level line placed in inches."""
    fig.add_artist(Line2D(np.asarray(xs) / FW, np.asarray(ys) / FH, transform=fig.transFigure, **kw))


# ---- (a) four equal square heatmaps: D = level + distinctiveness + relations ----
HS, HG, HX0, HY = 1.02, 0.42, 0.11, 2.33     # side, gap, left edge, bottom (in)
vmax_c = np.abs(np.concatenate([M.ravel(), R.ravel()])).max()
panels = [
    (D, r"Distances $d_{ij}$", "Greys", 0, D.max()),
    (L, r"Level $\mu$", "Greys", 0, D.max()),
    (M, r"Distinctiveness $a_i+a_j$", "PuOr_r", -vmax_c, vmax_c),
    (R, r"Relations $r_{ij}$", "PuOr_r", -vmax_c, vmax_c),
]
iu = np.triu_indices(n, 1)
tot = ((D[iu] - mu) ** 2).mean()
sm = ((M[iu]) ** 2).mean() / tot
sr = ((R[iu]) ** 2).mean() / tot
subs = ["Two roles + specialist", f"$\\mu={mu:.2f}$", f"{100*sm:.0f}% of variance", f"{100*sr:.0f}% of variance"]
axes_a = []
for p, (Mat, title, cmap, lo, hi) in enumerate(panels):
    ax = inch_axes(HX0 + p * (HS + HG), HY, HS, HS)
    ax.imshow(Mat, cmap=cmap, vmin=lo, vmax=hi, interpolation="none", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():  # thin frame: the zero diagonal cells would otherwise notch the squares
        sp.set_linewidth(0.5)
        sp.set_edgecolor("#888888")
    ax.set_title(title, pad=4)
    ax.text(0.5, -0.045, subs[p], transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK)
    axes_a.append(ax)
# operators drawn as strokes, so they sit exactly on the heatmaps' vertical centre
yc = HY + HS / 2
OP_W, OP_GAP, OP_LW = 0.10, 0.040, 1.0
for p in range(3):
    xc = HX0 + (p + 1) * HS + p * HG + HG / 2
    if p == 0:
        for dy in (-OP_GAP / 2, OP_GAP / 2):
            fline([xc - OP_W / 2, xc + OP_W / 2], [yc + dy, yc + dy], color=INK, lw=OP_LW, solid_capstyle="butt")
    else:
        fline([xc - OP_W / 2, xc + OP_W / 2], [yc, yc], color=INK, lw=OP_LW, solid_capstyle="butt")
        fline([xc, xc], [yc - OP_W / 2, yc + OP_W / 2], color=INK, lw=OP_LW, solid_capstyle="butt")
# slim shared colour bar for the two diverging panels (orange positive, purple negative)
CB_X = HX0 + 4 * HS + 3 * HG + 0.07
cax = inch_axes(CB_X, HY, 0.07, HS)
# drawn as an image (not a snapped mesh) so the ramp sits exactly inside its 0.5 pt frame, like the heatmaps
cax.imshow(np.linspace(-vmax_c, vmax_c, 256)[:, None], cmap="PuOr_r", vmin=-vmax_c, vmax=vmax_c, aspect="auto",
           origin="lower", extent=[0, 1, -vmax_c, vmax_c], interpolation="nearest")
cax.set_xticks([])
cax.yaxis.tick_right()
cax.set_yticks([-2, 0, 2])
cax.set_yticklabels(["$-2$", "$0$", "$+2$"])
for sp in cax.spines.values():
    sp.set_linewidth(0.5)
    sp.set_edgecolor("#888888")
cax.tick_params(axis="y", length=2.5, width=0.6, color="#555555", labelsize=7, labelcolor=INK, pad=1.5)


# ---- (b) two designs with the same budget ----
BOT_Y, ROW_H = 0.43, 1.30       # bottom row: lower edge and height of panel (c) (in)
RR = 0.46                       # ring radius (in)
RING_LIM = 1.25                 # ring axes half-width in units of the radius
DIGIT_W, DIGIT_H = 4.45, 5.1    # 7 pt DejaVu Serif digit width / cap height (pt)


def draw_design(ax, edges, color):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    P = np.c_[np.cos(ang), np.sin(ang)]
    deg = np.zeros(n, int)
    for i, j in edges:
        ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=color, lw=0.7, alpha=0.75, zorder=1)
        deg[i] += 1
        deg[j] += 1
    for i in range(n):
        if i == 0:  # the specialist: an orange diamond, so it stays distinct in grayscale
            ax.scatter(P[i, 0], P[i, 1], s=46, clip_on=False, marker="D", color=C_DIST, edgecolor="#000000", linewidth=0.9,
                       zorder=3)
            r_node = 0.707 * math.sqrt(46) + 0.45
        else:
            fc = "white" if deg[i] == 0 else "#DDDDDD"
            ax.scatter(P[i, 0], P[i, 1], s=34, clip_on=False, color=fc, edgecolor="#444444", linewidth=0.6, zorder=3)
            r_node = 0.5 * math.sqrt(34) + 0.3
        c, s_ = P[i]
        d = r_node + 2.0 + abs(c) * DIGIT_W / 2 + abs(s_) * DIGIT_H / 2   # clearance from the node (pt)
        tr = mtrans.offset_copy(ax.transData, fig=fig, x=c * d, y=s_ * d - DIGIT_H / 2, units="points")
        ax.text(c, s_, str(deg[i]), transform=tr, ha="center", va="baseline", fontsize=7, color=INK)
    ax.set_xlim(-RING_LIM, RING_LIM)
    ax.set_ylim(-RING_LIM, RING_LIM)
    ax.axis("off")


m_toy = 2 * n
allp = [(i, j) for i in range(n) for j in range(i + 1, n)]
rs = np.random.default_rng(11)
uni_edges = [allp[t] for t in rs.choice(len(allp), m_toy, replace=False)]
perm = rs.permutation(n)
bal_edges = [(perm[t], perm[(t + s) % n]) for t in range(n) for s in (1, 2)]
cx = [HX0 + HS / 2, HX0 + HS + HG + HS / 2]          # under the first two heatmaps
ring_cy = BOT_Y + ROW_H / 2 - 0.02
ax_u = inch_axes(cx[0] - RR * RING_LIM, ring_cy - RR * RING_LIM, 2 * RR * RING_LIM, 2 * RR * RING_LIM)
draw_design(ax_u, uni_edges, C_UNI)
ax_b = inch_axes(cx[1] - RR * RING_LIM, ring_cy - RR * RING_LIM, 2 * RR * RING_LIM, 2 * RR * RING_LIM)
draw_design(ax_b, bal_edges, C_BAL)
TITLE_Y = BOT_Y + ROW_H + 0.075                      # common title baseline of panels (b) and (c)
ftext(cx[0], TITLE_Y, "Uniform pairs", ha="center", va="baseline", fontsize=8.5)
ftext(cx[1], TITLE_Y, "Balanced, $k=4$", ha="center", va="baseline", fontsize=8.5)
# key under both rings
NOTE_Y = BOT_Y - 0.035
xm = (cx[0] + cx[1]) / 2
ftext(xm, NOTE_Y, "$m=2n$ pairs; numbers: comparisons per agent", ha="center", va="top", fontsize=7.5)
t2 = ftext(xm, NOTE_Y - 0.15, "Specialist (agent 1)", ha="left", va="top", fontsize=7.5)

# ---- (c) error vs budget ----
C_X0, C_X1 = 3.296, 5.333
ax = inch_axes(C_X0, BOT_Y, C_X1 - C_X0, ROW_H)
ax.loglog(ms, 100*np.array(res["sub"]), "-P", color=C_SUB, ms=3.8, lw=1.4, mec="white", mew=0.4, zorder=3)
ax.loglog(ms, 100*np.array(res["ber"]), "--^", color=C_BER, ms=3.8, lw=1.2, mec="white", mew=0.4, zorder=3)
ax.loglog(ms, 100*np.array(res["uni"]), "-s", color=C_UNI, ms=3.4, lw=1.4, mec="white", mew=0.4, zorder=3)
ax.loglog(ms, 100*np.array(res["bal"]), "-o", color=C_BAL, ms=3.6, lw=1.6, mec="white", mew=0.4, zorder=3)
ax.loglog(ms, 100*np.array(pred["uni"]), ":", color="black", lw=0.9, zorder=4)
ax.loglog(ms, 100*np.array(pred["bal"]), ":", color="black", lw=0.9, zorder=4)
for key, lab, dy in [("sub", "Agent\nsubsampling", 0.0), ("ber", "Bernoulli\n(HT)", 4.5),
                     ("uni", "Uniform\npairs", -4.5), ("bal", "Balanced", 0.0)]:
    tr = mtrans.offset_copy(ax.transData, fig=fig, x=6.0, y=dy, units="points")
    ax.text(ms[-1], 100 * res[key][-1], lab, transform=tr, fontsize=7, color=INK, va="center", ha="left",
            linespacing=1.0, clip_on=False)
XL = (ms[0] / 1.25, ms[-1] * 1.12)
x400 = (math.log10(400) - math.log10(XL[0])) / (math.log10(XL[1]) - math.log10(XL[0]))
ax.legend(handles=[Line2D([], [], ls=":", color="black", lw=0.9, label="Theory")], loc="lower center",
          bbox_to_anchor=(x400, 0.0), frameon=False,  # centred between the m=200 and m=800 grid lines
          handlelength=1.8, handletextpad=0.4, borderaxespad=0.3)
ax.set_xlim(*XL)
ax.set_ylim(0.1, 30)
ax.xaxis.set_major_locator(mt.FixedLocator([200, 800, 3200]))
ax.xaxis.set_major_formatter(mt.FixedFormatter(["200", "800", "3,200"]))
ax.xaxis.set_minor_locator(mt.FixedLocator([400, 1600, 6400]))
ax.xaxis.set_minor_formatter(mt.NullFormatter())
ax.yaxis.set_major_locator(mt.FixedLocator([0.1, 1, 10]))
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel(r"Comparisons $m$ (all pairs: $19{,}900$)", labelpad=3)
ax.set_ylabel("Relative RMSE (%)", labelpad=3)
ax.grid(True, which="major", color="#E6E6E6", lw=0.6)
ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
gain = np.array(res["uni"]) ** 2 / np.array(res["bal"]) ** 2
kappa = s_main / s_int
ftext((C_X0 + C_X1) / 2, TITLE_Y, f"$n={n2}$, $1+\\kappa\\approx{1 + kappa:.0f}$", ha="center", va="baseline",
      fontsize=8.5)

# ---- specialist key glyph, panel letters ----
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = fig.dpi_scale_trans.inverted()           # display -> inches
KEY_MS, KEY_GAP = 5.2, 0.045                  # diamond size (pt) and gap to its text (in)
key_w = 2 * 0.707 * KEY_MS / 72
bb2 = inv.transform_bbox(t2.get_window_extent(rend))
w2 = TextPath((0, 0), t2.get_text(), size=7.5, prop=t2.get_fontproperties()).get_extents().x1 / 72  # unhinted
x_key = xm - (key_w + KEY_GAP + w2) / 2          # centre the glyph + text group under both rings
t2.set_x((x_key + key_w + KEY_GAP) / FW)
fig.add_artist(Line2D([(x_key + key_w / 2) / FW], [((bb2.y0 + bb2.y1) / 2 + 0.004) / FH], marker="D", ms=KEY_MS,
                      color=C_DIST, mec="#000000", mew=0.8, ls="none", transform=fig.transFigure))


def top_in(artists):
    return max(inv.transform_bbox(a_.get_window_extent(rend)).y1 for a_ in artists)


X_LET = 0.02
y_a = top_in([a_.title for a_ in axes_a]) + 0.06
y_bc = top_in([t for t in fig.texts if abs(t.get_position()[1] * FH - TITLE_Y) < 1e-6]) + 0.06
ftext(X_LET, y_a, "(a)", fontsize=9, fontweight="bold", va="baseline")
ftext(X_LET, y_bc, "(b)", fontsize=9, fontweight="bold", va="baseline")
ftext(inv.transform_bbox(ax.yaxis.label.get_window_extent(rend)).x0, y_bc, "(c)", fontsize=9,
      fontweight="bold", va="baseline")

fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("kappa=", kappa, "1+kappa", 1 + kappa, "gain median", np.median(gain),
      "res", {k: [f"{v:.4f}" for v in vals] for k, vals in res.items()})
