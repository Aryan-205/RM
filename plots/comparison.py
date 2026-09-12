"""Figures for Stage 12 -- projectile vs planetary, head to head."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from plots.style import readable_log_axes, save, style_for

SYSTEM_STYLE = {
    "projectile": {"color": "tab:blue", "marker": "o", "label": "Projectile motion"},
    "planetary": {"color": "tab:red", "marker": "s", "label": "Planetary motion"},
}


def plot_flow_map_comparison(curves, save_path):
    """
    The controlled comparison the whole of Stage 12 is built around.

    IDENTICAL formulation (learn s_t -> s_{t+dt}, then feed the output back
    in), IDENTICAL model families, IDENTICAL training procedure. Sampling intervals and state distributions differ between systems.

    Errors are normalised by each system's own characteristic length -- the
    horizontal range of the flight, and the semi-major axis of the orbit --
    because "3 metres" and "0.01 AU" are not comparable numbers, but "0.3% of
    the system size" and "1% of the system size" are.

    Time is normalised the same way, in units of the natural period of each
    system (the flight time, and the orbital period), so that one unit on the
    x axis means the same physical thing in both panels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)

    for ax, system in zip(axes, ["projectile", "planetary"]):
        subset = curves[curves["system"] == system]
        for model in subset["model"].unique():
            rows = subset[subset["model"] == model].sort_values("t_normalised")
            style = style_for(model)
            ax.plot(rows["t_normalised"], rows["relative_error"],
                    color=style["color"], linewidth=1.7, label=style["label"])

        ax.set_yscale("log")
        ax.set_xlabel("Time, in units of the system's natural period\n"
                      "(flight time / orbital period)")
        ax.set_title(SYSTEM_STYLE[system]["label"], fontsize=11)
        ax.legend(fontsize=8.5, loc="lower right")

    axes[0].set_ylabel("Position error / system size  (dimensionless, log scale)")

    fig.suptitle(
        "Stage 12: the same learned flow map, applied to two different physical systems\n"
        "Projectile: the exact flow map is AFFINE, so a linear model represents it exactly.  "
        "Orbit: nonlinear acceleration; fitted flow maps remain approximate."
    )
    fig.tight_layout()
    save(fig, save_path, "the controlled system-vs-system comparison")


def plot_learning_curve_comparison(projectile, planetary, save_path):
    """
    Training-set size against relative accuracy, for both systems on one axis.

    Reading the vertical gap between the two families of curves is the answer
    to "how much more data does the harder system need?". Reading their SLOPES
    is the more useful answer: if the planetary curve is not merely lower but
    flatter, then more data is not simply expensive for the orbital problem,
    it is ineffective.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    ax = axes[0]
    for frame, system in [(projectile, "projectile"), (planetary, "planetary")]:
        for model in frame["model"].unique():
            rows = frame[frame["model"] == model].sort_values("n_train")
            style = style_for(model)
            ax.plot(rows["n_train"], rows["relative_error"],
                    marker=SYSTEM_STYLE[system]["marker"],
                    linestyle="-" if system == "projectile" else "--",
                    color=style["color"], linewidth=1.5, markersize=5,
                    label=f"{style['label']} ({system})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training rows (log scale)")
    ax.set_ylabel("Relative one-step error on training pool (log scale)")
    ax.set_title("Training-pool fit curves (overlapping evaluation)\n(solid = projectile, dashed = planetary)")
    ax.legend(fontsize=7, ncol=2)
    readable_log_axes(ax)

    ax = axes[1]
    slopes, labels, colours = [], [], []
    for frame, system in [(projectile, "projectile"), (planetary, "planetary")]:
        for model in frame["model"].unique():
            rows = frame[frame["model"] == model].sort_values("n_train")
            usable = rows[rows["relative_error"] > 1e-12]
            if len(usable) < 3:
                continue
            slope = np.polyfit(np.log10(usable["n_train"]),
                               np.log10(usable["relative_error"]), 1)[0]
            slopes.append(slope)
            labels.append(f"{style_for(model)['label']}\n({system})")
            colours.append(style_for(model)["color"])

    bars = ax.bar(range(len(slopes)), slopes, color=colours)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    ax.set_ylabel("Fitted slope  d log(error) / d log(n)")
    ax.set_title("How much does each extra decade of data buy?\n"
                 "(more negative = data helps more)")

    fig.suptitle("Stage 12: the cost of data, in both systems")
    fig.tight_layout()
    save(fig, save_path, "learning curves compared")


def plot_sensitivity_comparison(results, save_path):
    """
    How fast do two nearby TRUE trajectories separate in each system?

    This has nothing to do with machine learning -- it is a property of the
    physics, computed from exact solutions. It matters because it sets a floor
    on what any predictor can achieve: if the system itself amplifies a tiny
    input error, then no model, however good, can forecast far ahead.

    Both systems turn out to amplify only POLYNOMIALLY (neither is chaotic),
    which is the finding that forbids blaming Stage 11's rollout divergence on
    chaos.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    for ax, system in zip(axes, ["projectile", "planetary"]):
        subset = results[results["system"] == system]
        colours = plt.cm.viridis(np.linspace(0.15, 0.8, subset["offset"].nunique()))
        for (offset, group), colour in zip(subset.groupby("offset"), colours):
            ax.plot(group["t_normalised"], group["relative_separation"],
                    color=colour, linewidth=1.5, label=f"offset {offset:.0e}")

        ax.set_yscale("log")
        ax.set_xlabel("Time, in units of the system's natural period")
        ax.set_ylabel("Separation / system size")
        amplification = subset.groupby("offset")["amplification"].last().median()
        ax.set_title(f"{SYSTEM_STYLE[system]['label']}\n"
                     f"median amplification over one period: {amplification:.1f}x")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Stage 12: sensitivity of the EXACT dynamics to the initial conditions\n"
        "(no machine learning involved -- this is the floor any predictor must live above)"
    )
    fig.tight_layout()
    save(fig, save_path, "physical sensitivity, both systems")


def plot_cost_comparison(table, save_path):
    """
    Three costs that get conflated in casual discussion of "ML speed":

      training  -- paid once, and it is enormous compared to writing down the
                   classical solution, which costs nothing
      inference -- paid per prediction, and this is where a learned surrogate
                   can genuinely win
      classical -- the cost of just computing the answer properly

    The honest summary for THIS project is that learned inference is indeed
    faster per query for the orbital problem, and that this does not matter,
    because the classical computation was already fast enough and is ten
    orders of magnitude more accurate. The trade only becomes interesting when
    the classical computation is the expensive part -- which is the situation
    in the published work this project reviews, and not the situation here.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    systems = table["system"].unique()
    models = table["model"].unique()
    width = 0.8 / len(models)
    positions = np.arange(len(systems))
    for i, model in enumerate(models):
        values = [
            table[(table["system"] == s) & (table["model"] == model)]
            ["train_seconds"].iloc[0] for s in systems
        ]
        bars = ax.bar(positions + i * width - 0.4 + width / 2, values, width * 0.9,
                      color=style_for(model)["color"], label=style_for(model)["label"])
        ax.bar_label(bars, fmt="%.2g", fontsize=7, rotation=90, padding=2)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([SYSTEM_STYLE[s]["label"] for s in systems])
    ax.set_ylabel("Training time (seconds, log scale)")
    ax.set_title("One-off cost: training")
    ax.legend(fontsize=8)
    ax.margins(y=0.3)

    ax = axes[1]
    for i, model in enumerate(models):
        values = [
            table[(table["system"] == s) & (table["model"] == model)]
            ["microseconds_per_prediction"].iloc[0] for s in systems
        ]
        bars = ax.bar(positions + i * width - 0.4 + width / 2, values, width * 0.9,
                      color=style_for(model)["color"], label=style_for(model)["label"])
        ax.bar_label(bars, fmt="%.2g", fontsize=7, rotation=90, padding=2)

    for i, system in enumerate(systems):
        classical = table[table["system"] == system]["classical_microseconds"].iloc[0]
        ax.plot([i - 0.45, i + 0.45], [classical, classical], "k-", linewidth=2.2)
        ax.annotate("classical physics", xy=(i, classical * 1.25), fontsize=8,
                    ha="center")

    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([SYSTEM_STYLE[s]["label"] for s in systems])
    ax.set_ylabel("Microseconds per prediction (log scale)")
    ax.set_title("Recurring cost: one prediction\n(black line = computing it classically)")
    ax.legend(fontsize=8)
    ax.margins(y=0.3)

    fig.suptitle("Stage 12: computational cost, and why speed is the wrong headline here")
    fig.tight_layout()
    save(fig, save_path, "training vs inference vs classical cost")


def plot_scorecard(scorecard, save_path):
    """
    The whole study on one axis.

    Each row is a criterion; the bars show the projectile and planetary
    results side by side after normalising each criterion to its own scale.
    The purpose is not precision -- the numbers are all elsewhere -- but to
    make the SHAPE of the answer visible: the two systems are comparable on
    the criteria that depend on fitting, and separated by orders of magnitude
    on the criteria that depend on time.
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))

    criteria = list(scorecard["criterion"])
    y = np.arange(len(criteria))
    height = 0.38

    ax.barh(y - height / 2, scorecard["projectile_score"], height,
            color=SYSTEM_STYLE["projectile"]["color"], label="Projectile motion")
    ax.barh(y + height / 2, scorecard["planetary_score"], height,
            color=SYSTEM_STYLE["planetary"]["color"], label="Planetary motion")

    for i, row in scorecard.reset_index().iterrows():
        ax.text(row["projectile_score"] + 0.02, i - height / 2, row["projectile_note"],
                va="center", fontsize=7.5)
        ax.text(row["planetary_score"] + 0.02, i + height / 2, row["planetary_note"],
                va="center", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(criteria, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.55)
    ax.set_xlabel("Normalised score  (1.0 = the better of the two systems on that row)")
    ax.set_title(
        "Stage 12: is machine learning equally effective for a simple and a\n"
        "dynamically complex physical system?",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, save_path, "the summary scorecard")
