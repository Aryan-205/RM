import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from plots.style import MODEL_STYLE as _ALL_MODEL_STYLES
from plots.style import readable_log_axes as _readable_log_axes

# Stage 7 only compares these two models; the colours come from the shared
# palette in plots/style.py so a green line means "Random Forest" in every
# figure of the study, not just in this one.
MODEL_STYLE = {name: _ALL_MODEL_STYLES[name] for name in ["linear", "forest"]}


def plot_learning_curve(summary, save_path):
    """
    Training-set size vs test RMSE, one panel per target (x and y).

    Point   = mean RMSE over the repeated random training draws.
    Errorbar = +/- 1 standard deviation across those draws.

    Falling curve  = more data genuinely helps.
    Flat curve     = the model is limited by its own assumptions, not by data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, target in zip(axes, ["x", "y"]):
        for model, style in MODEL_STYLE.items():
            rows = summary[summary["model"] == model].sort_values("n_train")
            ax.errorbar(
                rows["n_train"],
                rows[f"rmse_{target}_mean"],
                yerr=rows[f"rmse_{target}_std"],
                capsize=4, linewidth=1.5, **style
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training set size (rows, log scale)")
        ax.set_ylabel(f"Test RMSE for {target} (m, log scale)")
        ax.set_title(f"Learning curve: {target} position")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        _readable_log_axes(ax)

    fig.suptitle(
        "Stage 7: effect of training-set size on prediction error\n"
        "(every point evaluated on the same held-out test set)"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_stability(summary, save_path):
    """
    Standard deviation of test RMSE across repeated random training draws.

    This answers a different question from the learning curve: not "how
    accurate is the model?" but "how much does the answer depend on WHICH
    rows we happened to train on?". A shrinking spread means the result has
    become reproducible; a large spread at small n means a single run at
    that size cannot be trusted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, target in zip(axes, ["x", "y"]):
        for model, style in MODEL_STYLE.items():
            rows = summary[summary["model"] == model].sort_values("n_train")
            ax.plot(
                rows["n_train"], rows[f"rmse_{target}_std"],
                linewidth=1.5, **style
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training set size (rows, log scale)")
        ax.set_ylabel(f"Std. dev. of test RMSE for {target} (m, log scale)")
        ax.set_title(f"Run-to-run variability: {target} position")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        _readable_log_axes(ax)

    fig.suptitle(
        "Stage 7: how reproducible is the result at each training-set size?\n"
        "(spread over repeated random training draws)"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_trajectory_by_datasize(
    x_classical, y_classical, sweeps, v0, theta, y0, error_curve, save_path
):
    """
    The physical version of the learning curve.

    Left  : one representative launch condition from the test set. The black
            parabola is the true physics; each dashed line is the Random
            Forest's predicted trajectory after training on n rows.
    Right : mean radial error |predicted - true| along the trajectory,
            averaged over many held-out conditions, with the shaded band
            covering the middle 50% of conditions.

    The left panel shows WHAT the error looks like (jagged, because a forest
    predicts in piecewise-constant steps); the right panel is the honest
    quantitative claim, because a single trajectory is far too noisy to judge
    convergence by eye.

    sweeps      : list of (n_train, predictions) with predictions columns [x, y]
    error_curve : DataFrame with n_train, mean_error, q25_error, q75_error
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(sweeps)))
    for (n_train, pred), colour in zip(sweeps, colours):
        ax.plot(
            pred[:, 0], pred[:, 1], "--", color=colour, linewidth=1.5,
            label=f"Random Forest, n = {n_train:,}"
        )
    ax.plot(
        x_classical, y_classical, "k-", linewidth=2.5,
        label="Classical physics (ground truth)"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        "One representative test condition\n"
        f"(v0 = {v0:.1f} m/s, theta = {np.rad2deg(theta):.1f} deg, y0 = {y0:.1f} m)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(
        error_curve["n_train"], error_curve["q25_error"], error_curve["q75_error"],
        color="tab:green", alpha=0.2, label="middle 50% of conditions"
    )
    ax.plot(
        error_curve["n_train"], error_curve["mean_error"],
        color="tab:green", marker="s", linewidth=1.5,
        label="mean over held-out conditions"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training set size (rows, log scale)")
    ax.set_ylabel("Mean radial trajectory error (m, log scale)")
    ax.set_title("Whole-trajectory error vs training-set size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    _readable_log_axes(ax)

    fig.suptitle(
        "Stage 7: reconstructing a full trajectory the model was never trained on"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
