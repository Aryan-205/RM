import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

g = 9.81


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

# 1) Predicted vs Actual scatter (x and y), Random Forest
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

y_test_arr = np.asarray(y_test)

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
plt.savefig("results_actual_vs_predicted.png")
print("\nSaved: results_actual_vs_predicted.png")


# 2) Residual plot: error vs actual x, both models
residuals_linear = y_test_arr[:, 0] - linear_predictions[:, 0]
residuals_forest = y_test_arr[:, 0] - forest_predictions[:, 0]

plt.figure(figsize=(8, 5))
plt.scatter(y_test_arr[:, 0], residuals_linear, alpha=0.4, s=10, label="Linear Regression")
plt.scatter(y_test_arr[:, 0], residuals_forest, alpha=0.4, s=10, label="Random Forest")
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Actual x (m)")
plt.ylabel("Residual: actual x - predicted x (m)")
plt.title("Residuals vs Actual x")
plt.legend()
plt.tight_layout()
plt.savefig("results_residuals.png")
print("Saved: results_residuals.png")


# 3) Classical trajectory vs ML-predicted trajectory, one fixed launch condition
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

plt.figure(figsize=(8, 5))
plt.plot(x_classical, y_classical, "k-", linewidth=2, label="Classical physics")
plt.plot(forest_sweep_pred[:, 0], forest_sweep_pred[:, 1], "g--", label="Random Forest")
plt.plot(linear_sweep_pred[:, 0], linear_sweep_pred[:, 1], "b:", label="Linear Regression")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title(f"Trajectory comparison (v0={v0_s:.1f}, theta={theta_s:.2f} rad, y0={y0_s:.1f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results_trajectory_comparison.png")
print("Saved: results_trajectory_comparison.png")
