import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#222222"            # text that carries information
UNIFORM_LIGHT = "#AAAAAA"  # uniform pairs when drawn in the same panel as the (blue) balanced design

DESIGN_STYLE = {
    # name: (label, color, linestyle, marker, open_marker)
    "round_robin":        ("Balanced", "#0072B2", "-", "o", False),
    "shuffled_circulant": ("Balanced, permuted circulant", "#56B4E9", "-", "D", False),
    "expander_fixed":     ("Random expander (fixed)", "#56B4E9", "-", "v", False),
    "uniform":            ("Uniform pairs", "#7F7F7F", "-", "s", False),
    "bernoulli_ht":       ("Bernoulli (HT)", "#999999", "--", "^", False),
    "bernoulli_ratio":    ("Bernoulli (ratio)", "#999999", ":", "<", False),
    "agent_subsample":    ("Agent subsampling", "#CC79A7", "-", "P", False),
    "landmark":           ("Landmarks", "#882255", "-", "X", False),
    "local_knn":          ("Nearest neighbors (fixed)", "#D55E00", "-", "o", True),
    "ring_fixed":         ("Fixed ring", "#CC6677", "-.", "s", True),
}
FAMILY_STYLE = {
    "near_simplex":  ("Near-simplex", "#000000"),
    "one_outlier":   ("One outlier", "#D55E00"),
    "hetero_scale":  ("Heterogeneous scale", "#E69F00"),
    "two_roles":     ("Two roles", "#009E73"),
    "unequal_roles": ("Unequal roles", "#CC79A7"),
    "ring":          ("Continuum", "#882255"),
    "vmas":          ("VMAS teams (incl. reset variants)", "#0072B2"),
    "zoo":           ("Model population", "#7F7F7F"),
}
ANATOMY_COLORS = {"level": "#BBBBBB", "distinct": "#E69F00", "relational": "#009E73"}


def setup():
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
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
    })


def save(fig, path_noext):
    fig.savefig(path_noext + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path_noext + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plain_log_ticks(axis, ticks, minor=True):
    """Fixed major ticks on a log axis labelled as plain numbers (0.1, 0.3, 1, 3, ...);
    minor ticks are kept unlabelled (minor=True) or removed (minor=False)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    if not minor:
        axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def panel_label(ax, s):
    ax.text(-0.02, 1.04, f"({s})", transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="bottom", ha="right")
