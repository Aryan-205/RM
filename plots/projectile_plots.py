import numpy as np
import matplotlib.pyplot as plt


def plot_actual_vs_predicted(y_test_arr, forest_predictions, save_path):
    """
    Scatter: actual vs predicted x and y (Random Forest).
    Points on the red dashed line = perfect prediction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].scatter(y_test_arr[:, 0], forest_predictions[:, 0], alpha=0.4, s=10)
    axes[0].plot(
        [y_test_arr[:, 0].min(), y_test_arr[:, 0].max()],
        [y_test_arr[:, 0].min(), y_test_arr[:, 0].max()],
        "r--", label="perfect prediction"
    )
    axes[0].set_xlabel("Actual x (m)")
    axes[0].set_ylabel("Predicted x (m)")
    axes[0].set_title("Random Forest: Actual vs Predicted x")
    axes[0].legend()

    axes[1].scatter(y_test_arr[:, 1], forest_predictions[:, 1], alpha=0.4, s=10, color="green")
    axes[1].plot(
        [y_test_arr[:, 1].min(), y_test_arr[:, 1].max()],
        [y_test_arr[:, 1].min(), y_test_arr[:, 1].max()],
        "r--", label="perfect prediction"
    )
    axes[1].set_xlabel("Actual y (m)")
    axes[1].set_ylabel("Predicted y (m)")
    axes[1].set_title("Random Forest: Actual vs Predicted y")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_residuals(actual_x, residuals_linear, residuals_forest, save_path):
    """
    Residual (actual - predicted) vs actual x, for both models.
    Flat scatter around 0 = good. Fanning out = systematic bias.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(actual_x, residuals_linear, alpha=0.4, s=10, label="Linear Regression")
    plt.scatter(actual_x, residuals_forest, alpha=0.4, s=10, label="Random Forest")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Actual x (m)")
    plt.ylabel("Residual: actual x - predicted x (m)")
    plt.title("Residuals vs Actual x")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_trajectory_comparison(
    x_classical, y_classical,
    forest_sweep_pred, linear_sweep_pred,
    v0, theta, y0,
    save_path
):
    """
    One fixed launch condition: classical parabola vs each model's
    predicted trajectory swept over t.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(x_classical, y_classical, "k-", linewidth=2, label="Classical physics")
    plt.plot(forest_sweep_pred[:, 0], forest_sweep_pred[:, 1], "g--", label="Random Forest")
    plt.plot(linear_sweep_pred[:, 0], linear_sweep_pred[:, 1], "b:", label="Linear Regression")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"Trajectory comparison (v0={v0:.1f}, theta={theta:.2f} rad, y0={y0:.1f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
