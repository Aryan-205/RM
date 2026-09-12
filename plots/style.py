"""
One visual language for every figure in the study.

A reader should be able to glance at any figure in the report and know which
model a line belongs to without reading the legend, because Random Forest is
green everywhere, Linear Regression is blue everywhere, and the classical /
numerical ground truth is always a solid black line.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

# Non-interactive backend: every script here writes PNGs, none opens a window.
mpl.use("Agg")

FIG_DPI = 150

MODEL_STYLE = {
    "linear":  {"color": "tab:blue",   "marker": "o", "label": "Linear Regression"},
    "poly2":   {"color": "tab:orange", "marker": "^", "label": "Polynomial Ridge (deg 2)"},
    "forest":  {"color": "tab:green",  "marker": "s", "label": "Random Forest"},
    "mlp":     {"color": "tab:red",    "marker": "D", "label": "Neural Network (MLP)"},
    "physics": {"color": "tab:purple", "marker": "*", "label": "Physics-Informed Features"},
    "pinn":    {"color": "tab:brown",  "marker": "P", "label": "PINN (physics-informed NN)"},
}

TRUTH_STYLE = {"color": "black", "linewidth": 2.2, "linestyle": "-"}

# Integrators get their own palette (Stage 10) so they are never confused
# with models.
INTEGRATOR_STYLE = {
    "euler":        {"color": "tab:red",    "label": "Explicit Euler"},
    "euler_cromer": {"color": "tab:orange", "label": "Euler-Cromer (semi-implicit)"},
    "verlet":       {"color": "tab:blue",   "label": "Velocity Verlet"},
    "rk4":          {"color": "tab:green",  "label": "Runge-Kutta 4"},
}


def apply_style():
    """Global matplotlib defaults. Call once at the top of a script."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "figure.titlesize": 13,
    })


def style_for(name):
    """Style dict for a model key, falling back to something sane."""
    return MODEL_STYLE.get(name, {"color": "grey", "marker": "x", "label": name})


# matplotlib's ScalarFormatter rounds to a fixed number of decimals, which
# silently renders 0.5 and 0.3 as "0" on a log axis whose ticks fall below 1.
# "%g" keeps significant digits instead of decimal places, so 0.5 stays 0.5
# and 20000 stays 20000.
_PLAIN_NUMBER = ticker.FuncFormatter(lambda value, _: f"{value:g}")


def readable_log_axes(ax, axis="both"):
    """
    Log axes label only powers of ten by default ("10^1"), which is hard to
    read when the data spans barely a decade. Show plain numbers instead, and
    label a few minor ticks when the range is narrow enough for them to fit.
    """
    if axis in ("x", "both") and ax.get_xscale() == "log":
        ax.xaxis.set_major_formatter(_PLAIN_NUMBER)
    if axis in ("y", "both") and ax.get_yscale() == "log":
        low, high = ax.get_ylim()
        ax.yaxis.set_major_formatter(_PLAIN_NUMBER)
        if low > 0 and np.log10(high / low) < 2.5:
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(2, 3, 5)))
            ax.yaxis.set_minor_formatter(_PLAIN_NUMBER)
            ax.tick_params(axis="y", which="minor", labelsize=8)


def save(fig, path, caption=None):
    """Save and report, so every script's stdout doubles as a figure index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    note = f"  -> {path.parent.name}/{path.name}"
    if caption:
        note += f"   [{caption}]"
    print(note)
