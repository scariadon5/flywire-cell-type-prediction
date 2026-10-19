"""Shared matplotlib style: thin marks, recessive axes, a fixed categorical order."""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots in fixed order (CVD-validated as an adjacent sequence). Never cycle past these.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "font.size": 10,
        "figure.dpi": 110,
    })


def hbar(ax, labels, values, color=SERIES[0], fmt="{:,.0f}"):
    """Horizontal bars, largest on top, each labelled with its value."""
    ax.barh(labels[::-1], values[::-1], color=color, height=0.7)
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(values[::-1]):
        ax.text(v, y, " " + fmt.format(v), va="center", color=TEXT_SECONDARY, fontsize=9)
    ax.margins(x=0.12)
    return ax


def finish(fig, path=None):
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.show()
