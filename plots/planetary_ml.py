"""Figures for Stage 11 -- machine learning applied to orbital motion."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from plots.style import TRUTH_STYLE, readable_log_axes, save, style_for

APPROACH_STYLE = {
    "A_flow_map": {"color": "tab:green", "marker": "s",
                   "label": "A: learned flow map (autoregressive)"},
    "B_direct":   {"color": "tab:purple", "marker": "*",
                   "label": "B: learned closed form (a, e, t) -> (x, y)"},
    "C_history":  {"color": "tab:orange", "marker": "^",
                   "label": "C: learned multistep map (3-state window)"},
}


def plot_one_step_accuracy(table, models, save_path):
    """
    Mean one-step position error, absolute targets vs delta targets.

    Two things to read off. First, every model predicts a single step very
    accurately -- which is exactly why one-step accuracy is a misleading way
    to report a learned simulator. Second, delta targets help the flexible
    models by one to two orders of magnitude, and do literally nothing for the
    linear and polynomial models. That second fact is not noise: a linear
    model can already represent the identity map exactly, so predicting
    s + f(s) instead of g(s) is a reparametrisation it can absorb without
    changing its fit at all.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    width = 0.38
    positions = np.arange(len(models))
    for offset, mode, hatch in [(-width / 2, "absolute", "//"), (width / 2, "delta", "")]:
        values = [
            table[(table["model"] == m) & (table["mode"] == mode)]["mean_position_error"].iloc[0]
            for m in models
        ]
        bars = ax.bar(positions + offset, values, width * 0.92, hatch=hatch,
                      label=f"{mode} targets",
                      color=[style_for(m)["color"] for m in models],
                      edgecolor="black", linewidth=0.6,
                      alpha=1.0 if mode == "delta" else 0.55)
        ax.bar_label(bars, fmt="%.1e", fontsize=7, rotation=90, padding=2)

    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([style_for(m)["label"] for m in models], fontsize=8, rotation=12)
    ax.set_ylabel("Mean one-step position error (AU, log scale)")
    ax.set_title("Accuracy of a SINGLE predicted step")
    ax.legend(fontsize=9)
    ax.margins(y=0.3)

    ax = axes[1]
    for m in models:
        rows = table[table["model"] == m]
        absolute = rows[rows["mode"] == "absolute"]["mean_position_error"].iloc[0]
        delta = rows[rows["mode"] == "delta"]["mean_position_error"].iloc[0]
        ax.barh(style_for(m)["label"], absolute / delta, color=style_for(m)["color"])
    ax.axvline(1.0, color="black", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Error reduction from using delta targets  (absolute / delta)")
    ax.set_title("How much does predicting the CHANGE help?\n"
                 "(1.0 = no difference)")
    ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Stage 11: one-step accuracy of the learned flow map (Approach A)")
    fig.tight_layout()
    save(fig, save_path, "one-step accuracy and the delta trick")


def plot_rollout_orbits(panels, duration_years, save_path):
    """
    Each panel: one unseen orbit, the true trajectory in black, and the
    model's autoregressive rollout in colour, over the same time span.

    The failure modes are visually distinct and physically meaningful. A
    rollout that spirals inwards has lost energy; one that spirals outwards
    has gained it; one that stays on the right ellipse but drifts around it
    has a PHASE error -- right orbit, wrong time -- which is the mildest and
    most common failure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))

    for ax, panel in zip(axes.ravel(), panels):
        style = style_for(panel["model"])
        ax.plot(panel["truth"][:, 0], panel["truth"][:, 1],
                label="RK4 reference (truth)", **TRUTH_STYLE)
        ax.plot(panel["predicted"][:, 0], panel["predicted"][:, 1],
                color=style["color"], linewidth=0.9, label="ML rollout")
        ax.plot(panel["truth"][0, 0], panel["truth"][0, 1], "ko", markersize=6,
                label="start")
        ax.plot(0, 0, marker="*", markersize=15, color="goldenrod",
                markeredgecolor="black", linestyle="none", label="Central body")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("y (AU)")
        if np.isfinite(panel["final_error"]):
            subtitle = (f"final position error = {panel['final_error']:.3f} AU,  "
                        f"|$\\Delta E/E$| = {panel['final_energy_error']:.2e}")
        else:
            # NaN here is not missing data: the rollout left the system (see
            # OneStepModel.rollout). Saying so is more informative than "nan".
            subtitle = (f"ESCAPED the system after {panel.get('escape_time', float('nan')):.2f} yr "
                        "-- rollout unstable")
        ax.set_title(f"{style['label']}\n{subtitle}")
        ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        f"Stage 11: {duration_years:.1f}-year autoregressive rollout on an unseen orbit\n"
        f"(a = {panels[0]['a']:.3f} AU, e = {panels[0]['e']:.3f}; "
        "each model's own output is fed back in at every step)"
    )
    fig.tight_layout()
    save(fig, save_path, "what a diverging rollout looks like")


def plot_error_growth(curves, models, numerical_error, thresholds, save_path):
    """
    THE central figure of Part II.

    Mean position error against time, averaged over many unseen orbits, with
    the shaded band covering the middle 50% of them. The y axis is
    logarithmic, so a STRAIGHT rising line means EXPONENTIAL growth in time
    and a bending line means something slower (typically linear or quadratic
    accumulation of a biased step error).

    The RK4 reference error is stated in an annotation below the learned
    curves. The log-axis floor is kept at 1e-5 AU for readability.

    The right panel converts the same data into the number a user would
    actually ask for: for how long does this model stay within a given error
    tolerance?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    ax = axes[0]
    for model in models:
        # t = 0 is dropped: the error there is exactly zero (the rollout starts
        # from the true state), and a zero on a log axis draws a vertical line
        # down to the floor that carries no information and wastes the range.
        curve = curves[(curves["model"] == model) & (curves["t"] > 0)]
        style = style_for(model)
        ax.fill_between(curve["t"], curve["q25"], curve["q75"],
                        color=style["color"], alpha=0.15)
        ax.plot(curve["t"], curve["mean"], color=style["color"],
                linewidth=1.6, label=style["label"])

    # Letting the axis reach the RK4 reference at ~1e-11 AU would compress the
    # entire ML story -- which happens between 1e-4 and 1e2 -- into the top
    # fifth of the panel. The reference is stated in words instead, which makes
    # the size of the gap clearer than a line at the bottom edge would.
    ax.set_ylim(bottom=1e-5)
    ax.annotate(
        f"RK4 reference holds {numerical_error:.0e} AU across the whole horizon\n"
        "-- six orders of magnitude below the bottom of this axis",
        xy=(curves["t"].max() * 0.03, 1.8e-5), fontsize=8, color="black",
    )
    for level, name in thresholds:
        ax.axhline(level, color="grey", linestyle="--", linewidth=0.8)
        ax.annotate(name, xy=(curves["t"].max() * 0.72, level * 1.25),
                    fontsize=8, color="grey")

    ax.set_yscale("log")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Position error (AU, log scale)")
    ax.set_title("Error accumulation during autoregressive rollout\n"
                 "(mean over unseen orbits; band = middle 50%)")
    ax.legend(fontsize=8.5, loc="lower right")

    ax = axes[1]
    horizon = curves.attrs["horizon_table"]
    levels = [level for level, _ in thresholds]
    width = 0.8 / len(models)
    positions = np.arange(len(levels))
    for i, model in enumerate(models):
        values = [
            horizon[(horizon["model"] == model) & (horizon["threshold"] == level)]
            ["time_to_threshold"].iloc[0]
            for level in levels
        ]
        style = style_for(model)
        bars = ax.bar(positions + i * width - 0.4 + width / 2, values, width * 0.9,
                      color=style["color"], label=style["label"])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{name}\n({level:g} AU)" for level, name in thresholds],
                       fontsize=8.5)
    ax.set_ylabel("Time before the error exceeds the tolerance (years)")
    ax.set_title("Usable prediction horizon\n(1 orbit = 1 year for a = 1 AU)")
    ax.legend(fontsize=8.5)
    ax.margins(y=0.2)

    fig.suptitle("Stage 11: how long can a learned simulator be trusted?")
    fig.tight_layout()
    save(fig, save_path, "error growth and usable horizon")


def plot_ml_conservation(panels, models, save_path):
    """
    Does the learned model respect the conservation laws?

    Nothing in the training objective mentions energy or angular momentum;
    the loss only penalises being in the wrong place. So there is no reason
    for a learned rollout to conserve anything, and it does not. The top row
    shows the relative energy error climbing by many orders of magnitude,
    while the RK4 reference stays flat at ~1e-11.

    The bottom row plots the orbital radius against time. The true radius
    oscillates between periapsis and apoapsis for ever with a fixed envelope.
    A rollout whose envelope creeps upwards or downwards is not just
    inaccurate -- it is describing a different physical system.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax = axes[0]
    for model in models:
        panel = panels[model]
        style = style_for(model)
        ax.plot(panel["t"], panel["energy_rel_error"], color=style["color"],
                linewidth=1.3, label=style["label"])
    ax.axhline(panels["numerical_energy_error"], color="black", linestyle=":",
               linewidth=1.4, label="RK4 reference")
    ax.set_yscale("log")
    ax.set_ylabel("|$\\Delta E / E_0$|")
    ax.set_title("Energy conservation during the rollout\n"
                 "(the loss function never mentioned energy)")
    ax.legend(fontsize=8.5, ncol=2)

    ax = axes[1]
    for model in models:
        panel = panels[model]
        style = style_for(model)
        ax.plot(panel["t"], panel["radius"], color=style["color"],
                linewidth=1.1, label=style["label"])
    ax.plot(panels[models[0]]["t"], panels[models[0]]["radius_true"],
            label="RK4 reference (truth)", **TRUTH_STYLE)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Orbital radius |r| (AU)")
    ax.set_title("Orbital radius over time\n"
                 "(a constant envelope means the orbit is still the right orbit)")
    ax.legend(fontsize=8.5, ncol=2)

    fig.suptitle("Stage 11: conserved quantities are not conserved by a learned model")
    fig.tight_layout()
    save(fig, save_path, "ML rollouts violate conservation laws")


def plot_approach_comparison(curves, save_path):
    """
    The three formulations on one axis, same data, same model family.

    Approach B is flat because it has no feedback loop -- ask it for t = 4 yr
    and it answers directly, with the same accuracy it has at t = 0.1 yr.
    Approaches A and C climb because each of their predictions becomes the
    next input.

    The tempting conclusion -- "so B is the better approach" -- is wrong, and
    the report should say why. B works only because a two-body orbit is
    integrable and can be labelled by (a, e). It is not a general method for
    dynamics; it is a curve fit to a solution we already possessed. A is the
    formulation that would still be available for a system with no analytic
    solution, which is the case where machine learning would actually be
    needed.
    """
    fig, ax = plt.subplots(figsize=(9.5, 6))

    for approach, style in APPROACH_STYLE.items():
        rows = curves[curves["approach"] == approach]
        if rows.empty:
            continue
        ax.fill_between(rows["t"], rows["q25"], rows["q75"],
                        color=style["color"], alpha=0.15)
        ax.plot(rows["t"], rows["mean"], color=style["color"], linewidth=1.8,
                label=style["label"])

    ax.set_yscale("log")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Position error (AU, log scale)")
    ax.set_title(
        "Stage 11: three ways to pose the same prediction problem\n"
        "(identical training trajectories and identical model family in all three)"
    )
    ax.legend(fontsize=9, loc="lower right")
    readable_log_axes(ax, axis="y")
    save(fig, save_path, "formulation matters more than model choice")


def plot_direct_extrapolation(curve, t_train_max, save_path):
    """
    Approach B does not accumulate error, but that does not make it safe.

    Because time enters as a raw input, asking for t beyond the training
    window is an extrapolation in exactly the sense of Stage 8 -- and the
    models fail there in exactly the same way. The flat region on the left and
    the collapse on the right are the same phenomenon as the Random Forest's
    flat tail in Figure stage8_response_slice.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    for model in curve["model"].unique():
        rows = curve[curve["model"] == model].sort_values("t")
        style = style_for(model)
        ax.plot(rows["t"], rows["mean_error"], color=style["color"],
                linewidth=1.6, label=style["label"])

    ax.axvline(t_train_max, color="black", linestyle="--", linewidth=1.2)
    ax.axvspan(0, t_train_max, color="grey", alpha=0.12)
    ax.text(t_train_max / 2, ax.get_ylim()[1], " trained on this time range ",
            ha="center", va="top", fontsize=9)

    ax.set_yscale("log")
    ax.set_xlabel("Time requested (years)")
    ax.set_ylabel("Position error (AU, log scale)")
    ax.set_title(
        "Stage 11: Approach B avoids error ACCUMULATION but not extrapolation\n"
        "(time is a raw input, so beyond the training window it is unconstrained)"
    )
    ax.legend(fontsize=9)
    save(fig, save_path, "Approach B fails outside its training time window")


def plot_planetary_learning_curve(summary, models, save_path):
    """
    Left : one-step error against training-set size. Falls steadily -- more
           data really does make each individual step more accurate.
    Right: the usable rollout horizon against the same training-set size.
           This is the figure that matters, and it is far less encouraging.
           Horizon grows roughly like the logarithm of the data, because the
           error at time t is (per-step error) amplified by the dynamics, so
           reducing the per-step error by a factor of ten buys only a fixed
           additive extension of the horizon -- not a tenfold one.

    Together these two panels are the quantitative answer to "can we fix this
    with more data?": you can, but the exchange rate is brutal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    ax = axes[0]
    for model in models:
        rows = summary[summary["model"] == model].sort_values("n_train")
        style = style_for(model)
        ax.plot(rows["n_train"], rows["one_step_error"], marker=style["marker"],
                color=style["color"], linewidth=1.6, markersize=5,
                label=style["label"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training pairs (log scale)")
    ax.set_ylabel("Mean one-step position error (AU, log scale)")
    ax.set_title("More data makes each step more accurate")
    ax.legend(fontsize=8.5)
    readable_log_axes(ax)

    ax = axes[1]
    for model in models:
        rows = summary[summary["model"] == model].sort_values("n_train")
        style = style_for(model)
        ax.plot(rows["n_train"], rows["horizon_0.01AU"], marker=style["marker"],
                color=style["color"], linewidth=1.6, markersize=5,
                label=style["label"])
    ax.set_xscale("log")
    ax.set_xlabel("Training pairs (log scale)")
    ax.set_ylabel("Time before error exceeds 0.01 AU (years)")
    ax.set_title("...but the usable horizon barely moves")
    ax.legend(fontsize=8.5)
    readable_log_axes(ax, axis="x")

    fig.suptitle("Stage 11: the exchange rate between training data and forecast horizon")
    fig.tight_layout()
    save(fig, save_path, "data buys accuracy, not horizon")


def plot_sensitivity(true_growth, ml_growth, save_path):
    """
    Is the error growth chaos, or is it the model?

    A two-body Kepler orbit is INTEGRABLE, not chaotic -- it has as many
    conserved quantities as degrees of freedom, and nearby trajectories
    separate at worst linearly in time, never exponentially. The left panel
    demonstrates this directly: we perturb the true initial condition by
    1e-10, 1e-8, 1e-6 AU and integrate exactly. The separation grows like t,
    driven by the slightly different orbital periods, and is perfectly
    predictable.

    The right panel overlays the learned model's error growth on the same
    axes. It is far steeper. That comparison is the point of the figure: the
    divergence of the ML rollout is NOT an unavoidable consequence of chaotic
    dynamics -- there is no chaos here to blame. It is the compounding of a
    small, systematic, model-made error, and it is therefore a property of the
    method, not of the physics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    ax = axes[0]
    colours = plt.cm.viridis(np.linspace(0.15, 0.8, len(true_growth)))
    for (delta, times, separation), colour in zip(true_growth, colours):
        ax.plot(times, separation, color=colour, linewidth=1.5,
                label=f"initial offset {delta:.0e} AU")
    # A pure t^1 guide line anchored to the last curve
    delta, times, separation = true_growth[-1]
    guide = separation[len(separation) // 10] * (times / times[len(times) // 10])
    ax.plot(times[1:], guide[1:], "k:", linewidth=1.2, label="linear growth $\\propto t$")
    ax.set_yscale("log")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Separation between the two exact trajectories (AU)")
    ax.set_title("Exact dynamics: how fast do nearby orbits separate?\n"
                 "(linear, not exponential -- the two-body problem is integrable)")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    for delta, times, separation in true_growth:
        ax.plot(times, separation, color="grey", linewidth=1.1, alpha=0.7)
    ax.annotate("exact dynamics,\nperturbed initial conditions",
                xy=(times[-1] * 0.55, separation[-1] * 0.25), fontsize=8.5, color="grey")
    for model, (times_ml, error_ml) in ml_growth.items():
        style = style_for(model)
        ax.plot(times_ml, error_ml, color=style["color"], linewidth=1.7,
                label=style["label"])
    ax.set_yscale("log")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Position error / separation (AU, log scale)")
    ax.set_title("ML rollout error on the same axes\n"
                 "(much steeper: this is model error compounding, not chaos)")
    ax.legend(fontsize=8.5, loc="lower right")

    fig.suptitle("Stage 11: separating physical sensitivity from model error")
    fig.tight_layout()
    save(fig, save_path, "the divergence is the model's fault, not chaos")
