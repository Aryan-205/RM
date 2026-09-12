"""Figures for Stage 8 -- interpolation vs extrapolation."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from plots.style import TRUTH_STYLE, readable_log_axes, save, style_for


def plot_regime_bars(table, regimes, models, save_path):
    """
    Mean radial error per (test regime, model), as grouped bars on a log axis.

    A log axis is essential here: the interpolation errors are ~1 m and the
    extrapolation errors are ~100 m, so on a linear axis the interpolation
    bars would be invisible and the figure would say nothing.
    """
    fig, ax = plt.subplots(figsize=(13, 5.5))

    width = 0.8 / len(models)
    positions = np.arange(len(regimes))

    for i, model in enumerate(models):
        values = [
            table[(table["regime"] == r) & (table["model"] == model)]["mean_radial"].iloc[0]
            for r in regimes
        ]
        style = style_for(model)
        bars = ax.bar(
            positions + i * width - 0.4 + width / 2, values, width * 0.92,
            color=style["color"], label=style["label"],
        )
        ax.bar_label(bars, fmt="%.2g", fontsize=6.5, rotation=90, padding=2)

    ax.set_yscale("log")
    # The physics-feature model's error is ~1e-13 m, i.e. floating-point
    # round-off. Letting the axis reach down there would squeeze the entire
    # 0.1-70 m range -- the part that actually differs between models -- into
    # a thin strip at the top. We floor the axis and keep the true value as
    # the printed bar label instead.
    ax.set_ylim(bottom=1e-2)
    ax.set_xticks(positions)
    ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=8.5)
    ax.set_ylabel("Mean radial error |predicted - true| (m, log scale)")
    ax.set_title(
        "Stage 8: the same trained models, evaluated inside and outside their training region\n"
        "(leftmost group is the control: test data drawn from the SAME region as training;\n"
        "bars running off the bottom are exact to floating-point round-off -- see printed labels)"
    )
    ax.legend(ncol=5, fontsize=8.5, loc="upper left")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.margins(y=0.25)
    save(fig, save_path, "regime comparison")


def plot_extrapolation_distance(curve, models, v0_train_high, save_path):
    """
    Error as a smooth function of how far outside the training box we ask.

    The bar chart says extrapolation is bad; this says *how* it goes bad.
    A model that degrades gracefully has a gently rising line; a model that
    has no notion of "outside" at all has a line that rises without bound.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model in models:
        rows = curve[curve["model"] == model].sort_values("v0_centre")
        style = style_for(model)
        ax.plot(
            rows["v0_centre"], rows["mean_radial"],
            marker=style["marker"], color=style["color"], label=style["label"],
            linewidth=1.6, markersize=5,
        )

    ax.axvline(v0_train_high, color="black", linestyle="--", linewidth=1.2)
    ax.axvspan(
        curve["v0_centre"].min(), v0_train_high, color="grey", alpha=0.12,
    )
    ax.annotate(
        "edge of training data",
        xy=(v0_train_high, ax.get_ylim()[1]), xytext=(4, -12),
        textcoords="offset points", fontsize=9, rotation=90, va="top",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Initial speed $v_0$ of the test samples (m/s)")
    ax.set_ylabel("Mean radial error (m, log scale)")
    ax.set_title(
        "Stage 8: prediction error as a function of distance beyond the training range\n"
        "(all models trained on $v_0 \\in$ [%g, %g] m/s only)"
        % (curve.attrs.get("v0_train_low", np.nan), v0_train_high)
    )
    ax.legend()
    readable_log_axes(ax, axis="y")
    save(fig, save_path, "graceful vs catastrophic failure")


def plot_generalization_trajectories(panels, save_path):
    """
    What extrapolation failure looks like as physics rather than as a number.

    panels : list of dicts with keys
             title, x_true, y_true, predictions {model: (x, y)}
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.4))
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels):
        # Ground truth goes down FIRST so that a model which is exactly right
        # (the physics-feature one) is drawn on top of it and stays visible;
        # otherwise it would be hidden underneath the black line and a reader
        # would think it had not been plotted.
        ax.plot(panel["x_true"], panel["y_true"], label="Classical physics", **TRUTH_STYLE)
        for model, (px, py) in panel["predictions"].items():
            style = style_for(model)
            ax.plot(px, py, "--", color=style["color"], linewidth=1.6, label=style["label"])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(panel["title"], fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Stage 8: one trajectory the models were trained for, and one they were not"
    )
    fig.tight_layout()
    save(fig, save_path, "trajectory-level view of extrapolation")


def plot_time_extrapolation(curve, models, t_train_high, save_path):
    """
    'How far into the future can the model predict?'

    Models are trained using only times in the first `t_train_high` fraction
    of each flight, then asked about the whole flight. Everything to the right
    of the dashed line is a genuine forward-in-time extrapolation.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model in models:
        rows = curve[curve["model"] == model].sort_values("t_fraction_centre")
        style = style_for(model)
        ax.plot(
            rows["t_fraction_centre"], rows["mean_radial"],
            marker=style["marker"], color=style["color"], label=style["label"],
            linewidth=1.6, markersize=5,
        )

    ax.axvline(t_train_high, color="black", linestyle="--", linewidth=1.2)
    ax.axvspan(0, t_train_high, color="grey", alpha=0.12)
    ax.text(
        t_train_high / 2, ax.get_ylim()[1], " trained on this part of the flight ",
        ha="center", va="top", fontsize=9,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Position within the flight,  $t / t_{\\mathrm{flight}}$")
    ax.set_ylabel("Mean radial error (m, log scale)")
    ax.set_title(
        "Stage 8: predicting later into the flight than the model was ever shown\n"
        f"(training times restricted to the first {t_train_high:.0%} of every trajectory)"
    )
    ax.legend()
    readable_log_axes(ax, axis="y")
    save(fig, save_path, "forward-in-time extrapolation")


def plot_response_slice(slice_data, models, v0_train, save_path):
    """
    The mechanism behind the numbers: a 1-D slice through the input space.

    We hold theta, y0 and t fixed and sweep v0 across a range far wider than
    the training range, then plot each model's predicted x. Because x is
    exactly linear in v0 here, the true response is a straight line, and any
    departure from it is purely the model's inductive bias made visible.

    A Random Forest averages training targets inside boxes; beyond the last
    box boundary there are no new boxes, so its prediction becomes CONSTANT.
    That flat tail is not a bug -- it is the defining behaviour of the model
    family, and it is why tree ensembles must never be trusted outside their
    training envelope.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(slice_data["v0"], slice_data["x_true"], label="True $x = v_0\\cos\\theta\\,t$",
            **TRUTH_STYLE)
    for model in models:
        style = style_for(model)
        ax.plot(
            slice_data["v0"], slice_data[model], "--",
            color=style["color"], linewidth=1.6, label=style["label"],
        )

    ax.axvspan(v0_train[0], v0_train[1], color="grey", alpha=0.15)
    ax.text(
        np.mean(v0_train), ax.get_ylim()[1], " training range ",
        ha="center", va="top", fontsize=9,
    )
    ax.set_xlabel("Initial speed $v_0$ (m/s)")
    ax.set_ylabel("Predicted horizontal position $x$ (m)")
    ax.set_title(
        "Stage 8: model response along a 1-D slice "
        f"($\\theta$ = {np.rad2deg(slice_data.attrs['theta']):.0f}$^\\circ$, "
        f"$y_0$ = {slice_data.attrs['y0']:.0f} m, t = {slice_data.attrs['t']:.2f} s)\n"
        "The true response is a straight line, so every deviation is model bias"
    )
    ax.legend()
    save(fig, save_path, "why the forest cannot extrapolate")
