"""Diversity gallery: anatomy of four populations.

Usage: python fig_gallery.py OUT_STEM --col SPEC --col SPEC --col SPEC --col SPEC
  SPEC is 'npz:PATH|TITLE|SUBTITLE' or 'synth:NAME|TITLE|SUBTITLE'.
Row 1: dissimilarity table D, agents ordered by the dominant component.
Row 2: variance shares of the distinctiveness (main-effect) and relational components.
Row 3: sorted distinctiveness a_i / mu, and the level mu.
"""
import argparse
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

C_DIST = "#E69F00"   # distinctiveness (Okabe-Ito orange)
C_REL = "#009E73"    # relational (Okabe-Ito bluish green)
INK = "#222222"      # all information-carrying text
SPINE = "#888888"
TICK = "#555555"
GRID = "#E6E6E6"
HAIR = "#BBBBBB"     # zero baseline of the a_i / mu bars
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.6,
    "xtick.color": TICK,
    "ytick.color": TICK,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.labelcolor": INK,
    "text.color": INK,
    "pdf.fonttype": 42,
})


def anatomy(D):
    n = D.shape[0]
    N = n * (n - 1) / 2
    iu = np.triu_indices(n, 1)
    mu = D[iu].mean()
    S = D.sum(1)
    a = (S - (n - 1) * mu) / (n - 2)
    R = D - mu - a[:, None] - a[None, :]
    np.fill_diagonal(R, 0.0)
    tot = ((D[iu] - mu) ** 2).sum() / N
    main = (n - 2) * (a ** 2).sum() / N
    rel = (R[iu] ** 2).sum() / N
    return mu, a, R, tot, main, rel


def order_agents(R, a, rel_share):
    # Order by the dominant component: the leading eigenvector of R when relational
    # structure carries >= 25% of the pairwise variance, else by distinctiveness.
    if rel_share < 0.25:
        return np.argsort(-a)
    w, V = np.linalg.eigh(R)
    v = V[:, np.argmax(np.abs(w))]
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return np.lexsort((-a, np.round(v, 6)))


def pdist(X, p=2):
    diff = X[:, None, :] - X[None, :, :]
    if p == 1:
        return np.abs(diff).sum(-1)
    return np.sqrt((diff ** 2).sum(-1))


def synth_two_roles(rng, n=24):
    c = np.repeat([0, 1], n // 2)
    C = np.array([[0.0] * 8, [3.0] + [0.0] * 7])
    return pdist(C[c] + 0.3 * rng.standard_normal((n, 8)))


def synth_hetero(rng, n=24):
    s = np.exp(0.6 * rng.standard_normal(n))
    return pdist(s[:, None] * rng.standard_normal((n, 64)))


def synth_outlier(rng, n=24):
    X = np.eye(n) + 0.05 * rng.standard_normal((n, n))
    X[0] *= 6.0
    return pdist(X)


def synth_unequal_roles(rng, n=24):
    sizes = np.maximum(1, np.round(np.array([0.4, 0.3, 0.2, 0.1]) * n).astype(int))
    sizes[0] += n - sizes.sum()
    c = np.repeat(np.arange(4), sizes)
    C = 3.0 * rng.standard_normal((4, 8))
    return pdist(C[c] + 0.3 * rng.standard_normal((n, 8)))


def nice_floor(x):
    """Largest 1-2-5 number not exceeding x (x > 0)."""
    e = math.floor(math.log10(x))
    for f in (5, 2, 1):
        if f * 10.0 ** e <= x * (1 + 1e-9):
            return f * 10.0 ** e
    return 5 * 10.0 ** (e - 1)


def tick_label(v, _pos=None):
    """Plain numbers, '0' for zero, typographic minus: identical style in every panel."""
    if abs(v) < 1e-12:
        return "0"
    return f"{v:g}".replace("-", "\u2212")


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    D = np.asarray(z["D"], dtype=float)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def sentence_case(s):
    """Capitalize the first character only (titles/subtitles are sentence case; the rest is left as given)."""
    s = s.strip()
    return s[:1].upper() + s[1:]


SYNTH = {"one_outlier": synth_outlier, "two_roles": synth_two_roles,
         "hetero_scale": synth_hetero, "unequal_roles": synth_unequal_roles}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--col", action="append", required=True,
                    help="'npz:PATH|TITLE|SUBTITLE' or 'synth:NAME|TITLE|SUBTITLE' (4 times)")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    cols = []
    for spec in args.col:
        src, title, sub = spec.split("|")
        kind, arg = src.split(":", 1)
        D = load_npz(arg) if kind == "npz" else SYNTH[arg](rng)
        # panel letters are drawn separately, never inside the title string
        title = re.sub(r"^\s*\(?[a-h]\)\s*", "", title)
        cols.append((D, sentence_case(title), sentence_case(sub)))

    # Layout in inches (from the bottom). FW is tuned so that the tight-cropped PDF is 6.00 in wide.
    FW = 6.014
    LM, RM, GAP = 0.52, 0.02, 0.36              # LM holds the shared y label; GAP the next column's tick labels
    w = (FW - LM - RM - 3 * GAP) / 4            # heatmap side = column width
    bar_y, bar_h = 0.17, 0.52                   # bottom row: sorted a_i / mu
    sh_y, sh_h = bar_y + bar_h + 0.215, 0.055   # middle row: variance shares
    hm_bot = sh_y + sh_h + 0.055                # top row: dissimilarity table
    hm_top = hm_bot + w
    sub_base = hm_top + 0.075                   # subtitle baseline
    title_base = sub_base + 0.165               # title / panel-letter baseline
    key_y = title_base + 0.215                  # centre line of the key + legend row
    FH = key_y + 0.075
    LETTER_DX = 0.25                            # panel letters hang this far left of each column
    fig = plt.figure(figsize=(FW, FH))

    def axes_in(x, y, w_, h_):
        return fig.add_axes([x / FW, y / FH, w_ / FW, h_ / FH])

    def text_in(x, y, s, **kw):
        return fig.text(x / FW, y / FH, s, **kw)

    for c, (D, title, sub) in enumerate(cols):
        n = D.shape[0]
        mu, a, R, tot, main_e, rel_e = anatomy(D)
        o = order_agents(R, a, rel_e / tot if tot > 0 else 0.0)
        x0 = LM + c * (w + GAP)
        ax = axes_in(x0, hm_bot, w, w)
        ax.imshow(D[np.ix_(o, o)], cmap=SEQ, vmin=0.0, vmax=float(D.max()), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():   # hairline frame: keeps the white corner cells (D_ii = 0) from notching the square
            sp.set_visible(True)
            sp.set_color(SPINE)
            sp.set_linewidth(0.6)
        text_in(x0 - LETTER_DX, title_base, f"({'abcdefgh'[c]})", ha="left", va="baseline",
                fontsize=9, fontweight="bold", color=INK)
        text_in(x0, title_base, title, ha="left", va="baseline", fontsize=8.5, color=INK)
        text_in(x0, sub_base, sub, ha="left", va="baseline", fontsize=7, color=INK)

        sh_main = main_e / tot if tot > 0 else 0.0
        sh_rel = 1.0 - sh_main
        sx = axes_in(x0, sh_y, w, sh_h)
        gap = 0.015 if min(sh_main, sh_rel) > 0.015 else 0.0   # white separator only where two segments meet
        if sh_main > 0:
            sx.barh(0, max(sh_main - gap / 2, 0.0), left=0, height=1.0, color=C_DIST, linewidth=0)
        if sh_rel > 0:
            sx.barh(0, max(sh_rel - gap / 2, 0.0), left=sh_main + gap / 2, height=1.0, color=C_REL, linewidth=0)
        sx.set_xlim(0, 1)
        sx.set_ylim(-0.5, 0.5)
        sx.axis("off")
        sx.text(0.0, -0.9, f"{100*sh_main:.0f}%", transform=sx.transAxes, ha="left", va="top", fontsize=7.5, color=INK)
        sx.text(1.0, -0.9, f"{100*sh_rel:.0f}%", transform=sx.transAxes, ha="right", va="top", fontsize=7.5, color=INK)

        bx = axes_in(x0, bar_y, w, bar_h)
        a_sorted = np.sort(a)[::-1] / mu if mu > 0 else np.sort(a)[::-1]
        xx = np.arange(n)
        bx.bar(xx, a_sorted, width=0.72, color=C_DIST, linewidth=0, zorder=2)
        bx.axhline(0.0, color=HAIR, lw=0.6, zorder=1)
        bx.set_xlim(-0.8, n - 0.2)
        amax = float(np.abs(a_sorted).max())
        lim = max(0.05, amax * 1.12)
        bx.set_ylim(-lim, lim)
        bx.set_xticks([])
        t = nice_floor(amax) if amax > 0 else 0.05
        bx.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([-t, 0.0, t]))
        bx.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(tick_label))
        bx.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        bx.set_axisbelow(True)
        bx.grid(axis="y", which="major", color=GRID, lw=0.6)
        bx.grid(axis="x", visible=False)
        for sp in ("top", "right", "bottom"):
            bx.spines[sp].set_visible(False)
        if c == 0:
            bx.set_ylabel(r"Sorted $a_i/\mu$", labelpad=3)
        bx.text(0.0, -0.06, rf"$\mu$ = {mu:.3g}", transform=bx.transAxes,
                ha="left", va="top", fontsize=7.5, color=INK)

    handles = [matplotlib.patches.Patch(facecolor=C_DIST, edgecolor="none", label="Distinctiveness (main effects $a_i$)"),
               matplotlib.patches.Patch(facecolor=C_REL, edgecolor="none", label="Relational structure ($R$)")]
    leg = fig.legend(handles=handles, loc="center right", bbox_to_anchor=((FW - RM) / FW, key_y / FH),
                     ncol=2, frameon=False, handlelength=0.9, handleheight=0.9, handletextpad=0.45,
                     columnspacing=1.6, borderaxespad=0.0, borderpad=0.0)

    # Colour key for the heatmaps, on the legend's baseline: each panel is scaled from 0 (white) to its own
    # maximum (darkest), so darker = more dissimilar. The gradient strip has the height of the legend swatches.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    lt = leg.get_texts()[0]
    base = lt.get_transform().transform(lt.get_position())[1] / fig.dpi if lt.get_va() == "baseline" else None
    sw = leg.get_patches()[0].get_window_extent(rend)
    k_lo, k_hi = sw.y0 / fig.dpi, sw.y1 / fig.dpi
    if base is None:   # fallback: centre the texts on the swatch
        base = 0.5 * (k_lo + k_hi) - 0.035
    kx = LM
    k_lab = text_in(kx, base, r"$D_{ij}$:", ha="left", va="baseline", fontsize=7, color=INK)
    kx += k_lab.get_window_extent(rend).width / fig.dpi + 0.06
    k0 = text_in(kx, base, "0", ha="left", va="baseline", fontsize=7, color=INK)
    kx += k0.get_window_extent(rend).width / fig.dpi + 0.04
    KW = 0.45
    kax = axes_in(kx, k_lo, KW, k_hi - k_lo)
    kax.imshow(np.linspace(0.0, 1.0, 256)[None, :], cmap=SEQ, vmin=0.0, vmax=1.0, aspect="auto",
               interpolation="bilinear", extent=(0, 1, 0, 1))
    kax.set_xticks([])
    kax.set_yticks([])
    for sp in kax.spines.values():
        sp.set_visible(True)
        sp.set_color(SPINE)
        sp.set_linewidth(0.6)
    text_in(kx + KW + 0.04, base, "panel max", ha="left", va="baseline", fontsize=7, color=INK)
    fig.savefig(args.out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out + ".png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
