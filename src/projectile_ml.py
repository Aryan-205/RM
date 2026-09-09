import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# allow "from plots.projectile_plots import ..." when run from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from plots.projectile_plots import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_trajectory_comparison,
)

g = 9.81
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# -------------------- LOAD DATA --------------------

dataset = pd.read_csv("data/projectile_dataset.csv")

features = dataset[["v0", "theta", "y0", "t"]]
targets = dataset[["x", "y"]]

X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=42
)


# -------------------- TRAIN MODELS --------------------

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)


# -------------------- STAGE 5: METRICS --------------------

def report_metrics(model_name, y_true, y_pred):
    """
    Print MAE, RMSE, R^2 for x and y separately.
    y_true, y_pred: arrays/dataframes with columns [x, y] in that order.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    print(f"\n--- {model_name} ---")
    for i, label in enumerate(["x", "y"]):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"{label}: MAE={mae:.3f} m, RMSE={rmse:.3f} m, R2={r2:.4f}")


print("=" * 50)
print("STAGE 5 -- EVALUATION METRICS (on 400 held-out test rows)")
print("=" * 50)

report_metrics("Linear Regression", y_test, linear_predictions)
report_metrics("Random Forest", y_test, forest_predictions)


# -------------------- STAGE 6: VISUAL COMPARISON --------------------

y_test_arr = np.asarray(y_test)

plot_actual_vs_predicted(
    y_test_arr, forest_predictions,
    save_path=RESULTS_DIR / "actual_vs_predicted.png"
)
print(f"\nSaved: {RESULTS_DIR / 'actual_vs_predicted.png'}")

residuals_linear = y_test_arr[:, 0] - linear_predictions[:, 0]
residuals_forest = y_test_arr[:, 0] - forest_predictions[:, 0]

plot_residuals(
    y_test_arr[:, 0], residuals_linear, residuals_forest,
    save_path=RESULTS_DIR / "residuals.png"
)
print(f"Saved: {RESULTS_DIR / 'residuals.png'}")


def classical_trajectory(v0, theta, y0, t_array):
    x = v0 * np.cos(theta) * t_array
    y = y0 + v0 * np.sin(theta) * t_array - 0.5 * g * t_array**2
    return x, y


# pick one condition from the test set to compare against
sample = X_test.iloc[0]
v0_s, theta_s, y0_s = sample["v0"], sample["theta"], sample["y0"]

time_of_flight = (
    v0_s * np.sin(theta_s)
    + np.sqrt((v0_s * np.sin(theta_s)) ** 2 + 2 * g * y0_s)
) / g
t_sweep = np.linspace(0, time_of_flight, 50)

x_classical, y_classical = classical_trajectory(v0_s, theta_s, y0_s, t_sweep)

sweep_features = pd.DataFrame({
    "v0": v0_s, "theta": theta_s, "y0": y0_s, "t": t_sweep
})
forest_sweep_pred = forest_model.predict(sweep_features)
linear_sweep_pred = linear_model.predict(sweep_features)

plot_trajectory_comparison(
    x_classical, y_classical,
    forest_sweep_pred, linear_sweep_pred,
    v0_s, theta_s, y0_s,
    save_path=RESULTS_DIR / "trajectory_comparison.png"
)
print(f"Saved: {RESULTS_DIR / 'trajectory_comparison.png'}")
