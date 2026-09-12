"""Figures for Stage 13 -- physics-informed machine learning."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from plots.style import TRUTH_STYLE, readable_log_axes, save, style_for

PINN_STYLE = {"color": "tab:brown", "marker": "P", "label": "PINN (data + physics)"}
PLAIN_STYLE = {"color": "tab:red", "marker": "D", "label": "Plain NN (data only)"}


def plot_data_efficiency(summary, save_path):
    """
    The central claim of Stage 13, stated as a curve.

    Both networks have identical architecture, identical initialisation
    scheme, identical optimiser and identical training length. The ONLY
    difference is that one of them also has the equations of motion in its
    loss. Any gap between the curves is therefore attributable to the physics
    term and to nothing else -- this is a controlled comparison, not a
    demonstration that a bigger model does better.

    Two features to read off:
      * the PINN's point at zero labels. It was trained on no measurements at
        all, only on the requirement that its output satisfy d2x/dt2 = 0 and
        d2y/dt2 = -g with the right initial conditions. That it lands anywhere
        near the truth is the whole idea.
      * the curves converging on the right. With enough data the physics term
        stops mattering. Physics-informed learning is a low-data technique,
        and a report that claims otherwise is overselling it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    ax = axes[0]
    for key, style in [("pinn", PINN_STYLE), ("plain", PLAIN_STYLE)]:
        rows = summary[summary["model"] == key].sort_values("n_labels")
        rows = rows[rows["n_labels"] > 0]
        ax.errorbar(rows["n_labels"], rows["mean_error"], yerr=rows["std_error"],
                    marker=style["marker"], color=style["color"], capsize=3,
                    linewidth=1.7, markersize=6, label=style["label"])

    zero = summary[(summary["model"] == "pinn") & (summary["n_labels"] == 0)]
    if not zero.empty:
        level = zero["mean_error"].iloc[0]
        ax.axhline(level, color=PINN_STYLE["color"], linestyle="--", linewidth=1.3)
        ax.annotate(
            f"PINN trained on ZERO labelled points: {level:.2f} m",
            xy=(summary["n_labels"].max() * 0.06, level * 1.12),
            fontsize=9, color=PINN_STYLE["color"],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of labelled training points (log scale)")
    ax.set_ylabel("Mean radial test error (m, log scale)")
    ax.set_title("Same network, same training -- with and without the physics loss")
    ax.legend(fontsize=9)
    readable_log_axes(ax)

    ax = axes[1]
    pinn = summary[summary["model"] == "pinn"].set_index("n_labels")["mean_error"]
    plain = summary[summary["model"] == "plain"].set_index("n_labels")["mean_error"]
    shared = sorted(set(pinn.index) & set(plain.index) - {0})
    ax.bar([str(n) for n in shared], [plain[n] / pinn[n] for n in shared],
           color=PINN_STYLE["color"])
    ax.axhline(1.0, color="black", linewidth=1.2)
    ax.set_xlabel("Number of labelled training points")
    ax.set_ylabel("Error ratio  (plain NN / PINN)")
    ax.set_title("How many times better is the PINN?\n(1.0 = the physics term bought nothing)")

    fig.suptitle("Stage 13: physics-informed learning is a data-efficiency technique")
    fig.tight_layout()
    save(fig, save_path, "data efficiency of the PINN")


def plot_physics_residual(table, save_path):
    """
    A second, independent question: is the model's output PHYSICALLY POSSIBLE?

    Position error asks "is it in the right place". The residual asks "could
    any object obeying Newton's laws move like this at all". They are not the
    same question, and a model can score well on the first while failing the
    second -- a Random Forest's prediction is piecewise constant in t, so its
    numerical second derivative is wild even where its positions are good.

    We plot the mean of |d2x/dt2| (should be 0) and |d2y/dt2 + g| (should be
    0), both in m/s^2, evaluated by the same finite difference for every model
    so the comparison is fair. For scale, g itself is 9.81 m/s^2: a residual
    near that value means the model's implied acceleration is essentially
    unrelated to gravity.
    """
    fig, ax = plt.subplots(figsize=(10, 5.6))

    models = list(table["model"])
    positions = np.arange(len(models))
    width = 0.38

    bars_x = ax.bar(positions - width / 2, table["residual_x"], width * 0.92,
                    label="$|d^2x/dt^2 - 0|$", color="tab:blue")
    bars_y = ax.bar(positions + width / 2, table["residual_y"], width * 0.92,
                    label="$|d^2y/dt^2 + g|$", color="tab:orange")
    ax.bar_label(bars_x, fmt="%.2g", fontsize=7.5, rotation=90, padding=2)
    ax.bar_label(bars_y, fmt="%.2g", fontsize=7.5, rotation=90, padding=2)

    ax.axhline(9.81, color="red", linestyle="--", linewidth=1.2)
    ax.annotate("g = 9.81 m/s$^2$ (a residual this large means the implied\n"
                "acceleration has nothing to do with gravity)",
                xy=(len(models) - 0.5, 11), fontsize=8, color="red", ha="right")

    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(table["label"], fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel("Mean violation of the equation of motion (m/s$^2$, log scale)")
    ax.set_title(
        "Stage 13: does the model's output obey Newton's second law?\n"
        "(measured by finite differences on each model's own predictions)"
    )
    ax.legend(fontsize=9)
    ax.margins(y=0.3)
    save(fig, save_path, "physical consistency of each model")


def plot_pinn_extrapolation(panels, t_data_max_fraction, save_path):
    """
    The most persuasive demonstration of what a physics loss actually does.

    Both networks receive labelled points only from the first
    `t_data_max_fraction` of each flight. The PINN additionally receives
    COLLOCATION points -- inputs with no labels attached -- covering the whole
    flight, where the only thing it is told is that the equations of motion
    must hold.

    In the shaded region both models have data and both do fine. Outside it
    the plain network is extrapolating in exactly the sense of Stage 8, and it
    wanders. The PINN has no data there either, but it is not unconstrained:
    it still has to satisfy d2y/dt2 = -g, and that single requirement is
    almost enough to pin the trajectory down. This is the practical content of
    "physics-informed": a physical law is a statement about everywhere, and it
    can constrain a model in regions where no measurement exists.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(6.3 * len(panels), 5.6))
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels):
        ax.plot(panel["x_true"], panel["y_true"], label="Classical physics", **TRUTH_STYLE)
        ax.plot(panel["x_plain"], panel["y_plain"], "--", color=PLAIN_STYLE["color"],
                linewidth=1.7, label=PLAIN_STYLE["label"])
        ax.plot(panel["x_pinn"], panel["y_pinn"], "--", color=PINN_STYLE["color"],
                linewidth=1.7, label=PINN_STYLE["label"])

        cut = panel["cut_index"]
        ax.plot(panel["x_true"][:cut], panel["y_true"][:cut], color="grey",
                linewidth=9, alpha=0.25, zorder=0,
                label=f"labelled data covers this part only")

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(panel["title"], fontsize=10)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Stage 13: a physical law constrains the model where the data does not\n"
        f"(both networks saw labels only from the first {t_data_max_fraction:.0%} "
        "of each flight)"
    )
    fig.tight_layout()
    save(fig, save_path, "physics fills in where data runs out")


def plot_training_history(histories, save_path):
    """
    The three loss terms during training, so the optimisation can be audited
    rather than trusted.

    What to look for: all three curves should fall together. If the physics
    loss falls while the data loss rises, the weights are unbalanced and the
    network is satisfying the equations with the wrong trajectory -- which it
    can do, because infinitely many parabolas have the right curvature and
    only the initial-condition term rules them out.
    """
    fig, axes = plt.subplots(1, len(histories), figsize=(6.3 * len(histories), 5),
                             squeeze=False)

    for ax, (title, history) in zip(axes[0], histories.items()):
        epochs = [h["epoch"] for h in history]
        for key, colour, label in [
            ("data", "tab:blue", "data loss  (fit the measurements)"),
            ("physics", "tab:green", "physics loss  (obey the ODE)"),
            ("ic", "tab:orange", "initial-condition loss  (start correctly)"),
        ]:
            values = np.array([h[key] for h in history], dtype=float)
            if np.all(values == 0):
                continue
            ax.plot(epochs, np.maximum(values, 1e-16), color=colour, linewidth=1.5,
                    label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Loss term (dimensionless)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Stage 13: the three loss terms during training")
    fig.tight_layout()
    save(fig, save_path, "training diagnostics")
