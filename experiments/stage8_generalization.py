"""
Stage 8 -- Test Generalization (interpolation vs extrapolation)

Research sub-question:
    Does a model that is accurate on held-out data from the training
    distribution remain accurate on physical conditions it has never seen?

Why this is the single most important experiment in Part I
----------------------------------------------------------
Stage 5 reported R^2 = 0.99 for the Random Forest. Read carelessly, that is
"the model learned projectile motion". It did not. R^2 was computed on test
rows drawn from *the same distribution* as the training rows: same speed
range, same angle range, same height range. That measures INTERPOLATION.

A physical law is supposed to hold everywhere. If the model had genuinely
induced x = v0 cos(theta) t, it would work at v0 = 60 m/s even though it only
ever saw v0 <= 40 m/s. Stage 8 tests exactly that, and the answer separates
"a model that fits data" from "a model that learned physics".

Definitions used throughout
---------------------------
Interpolation : the query point lies inside the convex hull of the training
                inputs. The model is filling a gap between things it has seen.
Extrapolation : the query point lies outside it. The model is being asked to
                invent behaviour, and its answer is decided entirely by its
                built-in assumptions (its inductive bias), not by data.

Experimental design
-------------------
  Independent variable : which region of parameter space the TEST data comes
                         from (the training set is held fixed throughout)
  Dependent variable   : mean radial error, RMSE_x, RMSE_y, R^2
  Controlled           : the training set (one fixed set of 8,000 rows), the
                         model hyperparameters, the size of every test set

Run:  python experiments/stage8_generalization.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from data.generation.projectileDataGeneration import (
    generateProjectileDatasetInRange,
    timeOfFlight,
)
from plots.generalization import (
    plot_extrapolation_distance,
    plot_generalization_trajectories,
    plot_regime_bars,
    plot_response_slice,
    plot_time_extrapolation,
)
from plots.style import apply_style
from src.evaluation import evaluate_multioutput, radial_error, summarise_radial
from src.models import MODEL_LABELS, ProjectilePhysicsModel, make_model

g = 9.81
FEATURES = ["v0", "theta", "y0", "t"]
TARGETS = ["x", "y"]

RESULTS = ROOT / "results"

# ---- the training envelope: deliberately a strict SUBSET of what is physical
TRAIN_V0 = (15.0, 40.0)          # m/s
TRAIN_THETA_DEG = (20.0, 70.0)   # degrees
TRAIN_Y0 = (0.0, 15.0)           # m
TRAIN_ROWS = 8000

TEST_ROWS = 2000
MODELS = ["linear", "poly2", "forest", "mlp", "physics"]


def build_model(name):
    """The physics-feature model is ours, the rest come from sklearn."""
    return ProjectilePhysicsModel() if name == "physics" else make_model(name)


# -------------------- TEST REGIMES --------------------
# Each regime is one sampling box. `interpolation` uses exactly the training
# box (but a different seed, so the rows are new); every other regime pushes
# one variable -- or all of them -- outside it.

REGIMES = {
    "interpolation": dict(
        v0_range=TRAIN_V0, theta_deg_range=TRAIN_THETA_DEG, y0_range=TRAIN_Y0,
        note="control: same region as training, unseen rows",
    ),
    "extrap_v0_high": dict(
        v0_range=(40.0, 55.0), theta_deg_range=TRAIN_THETA_DEG, y0_range=TRAIN_Y0,
        note="faster launches than any seen",
    ),
    "extrap_v0_low": dict(
        v0_range=(5.0, 15.0), theta_deg_range=TRAIN_THETA_DEG, y0_range=TRAIN_Y0,
        note="slower launches than any seen",
    ),
    "extrap_theta_high": dict(
        v0_range=TRAIN_V0, theta_deg_range=(70.0, 85.0), y0_range=TRAIN_Y0,
        note="steeper, near-vertical launches",
    ),
    "extrap_theta_low": dict(
        v0_range=TRAIN_V0, theta_deg_range=(5.0, 20.0), y0_range=TRAIN_Y0,
        note="flatter, near-horizontal launches",
    ),
    "extrap_y0_high": dict(
        v0_range=TRAIN_V0, theta_deg_range=TRAIN_THETA_DEG, y0_range=(15.0, 30.0),
        note="launched from higher ground than any seen",
    ),
    "extrap_all": dict(
        v0_range=(40.0, 55.0), theta_deg_range=(70.0, 85.0), y0_range=(15.0, 30.0),
        note="every variable outside its training range at once",
    ),
}


def evaluate_everywhere(models, seed0=500):
    """Evaluate every trained model on every regime. Returns a tidy table."""
    rows = []
    for i, (regime, spec) in enumerate(REGIMES.items()):
        test = generateProjectileDatasetInRange(
            TEST_ROWS,
            v0_range=spec["v0_range"],
            theta_deg_range=spec["theta_deg_range"],
            y0_range=spec["y0_range"],
            seed=seed0 + i,
        )
        X_test, y_test = test[FEATURES], test[TARGETS]

        for name, model in models.items():
            pred = model.predict(X_test)
            per_target = evaluate_multioutput(y_test, pred, TARGETS)
            row = {
                "regime": regime,
                "note": spec["note"],
                "model": name,
                "model_label": MODEL_LABELS[name],
                "rmse_x": per_target.loc[0, "rmse"],
                "rmse_y": per_target.loc[1, "rmse"],
                "mae_x": per_target.loc[0, "mae"],
                "mae_y": per_target.loc[1, "mae"],
                "r2_x": per_target.loc[0, "r2"],
                "r2_y": per_target.loc[1, "r2"],
            }
            row.update(summarise_radial(y_test, pred))
            rows.append(row)
    return pd.DataFrame(rows)


def extrapolation_distance_curve(models, bin_edges, seed0=700):
    """
    Sweep the test v0 across narrow bins, from inside the training range to
    far outside it, and record the error in each bin. This turns the coarse
    in/out distinction of the bar chart into a continuous curve.
    """
    rows = []
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        test = generateProjectileDatasetInRange(
            1500, v0_range=(low, high),
            theta_deg_range=TRAIN_THETA_DEG, y0_range=TRAIN_Y0,
            seed=seed0 + i,
        )
        X_test, y_test = test[FEATURES], test[TARGETS]
        for name, model in models.items():
            err = radial_error(y_test, model.predict(X_test))
            rows.append({
                "model": name, "v0_low": low, "v0_high": high,
                "v0_centre": 0.5 * (low + high),
                "mean_radial": float(err.mean()),
                "median_radial": float(np.median(err)),
            })
    curve = pd.DataFrame(rows)
    curve.attrs["v0_train_low"] = TRAIN_V0[0]
    return curve


def time_extrapolation(bins=10, seed0=900):
    """
    A second, physically different kind of extrapolation: forward in time.

    Models are retrained on trajectories truncated to the first 60% of their
    flight, then asked to predict the remaining 40%. This is the closest thing
    in Part I to the question that dominates Part II -- how far into the
    future can a learned model be trusted?
    """
    t_train_high = 0.6

    train = generateProjectileDatasetInRange(
        TRAIN_ROWS, v0_range=TRAIN_V0, theta_deg_range=TRAIN_THETA_DEG,
        y0_range=TRAIN_Y0, t_fraction_range=(0.0, t_train_high), seed=seed0,
    )
    models = {}
    for name in MODELS:
        models[name] = build_model(name).fit(train[FEATURES], train[TARGETS])

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        test = generateProjectileDatasetInRange(
            1500, v0_range=TRAIN_V0, theta_deg_range=TRAIN_THETA_DEG,
            y0_range=TRAIN_Y0, t_fraction_range=(edges[i], edges[i + 1]),
            seed=seed0 + 50 + i,
        )
        for name, model in models.items():
            err = radial_error(test[TARGETS], model.predict(test[FEATURES]))
            rows.append({
                "model": name,
                "t_fraction_low": edges[i], "t_fraction_high": edges[i + 1],
                "t_fraction_centre": 0.5 * (edges[i] + edges[i + 1]),
                "mean_radial": float(err.mean()),
                "extrapolating": edges[i] >= t_train_high,
            })
    return pd.DataFrame(rows), t_train_high


def response_slice(models, theta_deg=45.0, y0=5.0, t=1.5, v0_lo=0.0, v0_hi=80.0):
    """One-dimensional cut through input space (see plot for interpretation)."""
    theta = np.deg2rad(theta_deg)
    v0 = np.linspace(v0_lo, v0_hi, 400)
    frame = pd.DataFrame({"v0": v0, "theta": theta, "y0": y0, "t": t})

    out = pd.DataFrame({"v0": v0, "x_true": v0 * np.cos(theta) * t})
    for name, model in models.items():
        out[name] = model.predict(frame)[:, 0]
    out.attrs.update({"theta": theta, "y0": y0, "t": t})
    return out


def trajectory_panels(models):
    """Full trajectories for one in-range and one out-of-range condition."""
    conditions = [
        dict(v0=30.0, theta_deg=45.0, y0=8.0,
             title="INSIDE the training region\n$v_0$=30 m/s, $\\theta$=45$^\\circ$, $y_0$=8 m"),
        dict(v0=52.0, theta_deg=78.0, y0=25.0,
             title="OUTSIDE the training region\n$v_0$=52 m/s, $\\theta$=78$^\\circ$, $y_0$=25 m"),
    ]

    panels = []
    for c in conditions:
        theta = np.deg2rad(c["theta_deg"])
        flight = timeOfFlight(c["v0"], theta, c["y0"])
        t = np.linspace(0, flight, 120)
        frame = pd.DataFrame({"v0": c["v0"], "theta": theta, "y0": c["y0"], "t": t})

        predictions = {}
        for name, model in models.items():
            pred = model.predict(frame)
            predictions[name] = (pred[:, 0], pred[:, 1])

        panels.append({
            "title": c["title"],
            "x_true": c["v0"] * np.cos(theta) * t,
            "y_true": c["y0"] + c["v0"] * np.sin(theta) * t - 0.5 * g * t ** 2,
            "predictions": predictions,
        })
    return panels


# -------------------- MAIN --------------------

if __name__ == "__main__":
    apply_style()
    RESULTS.mkdir(exist_ok=True)

    print("=" * 72)
    print("STAGE 8 -- GENERALIZATION: INTERPOLATION vs EXTRAPOLATION")
    print("=" * 72)
    print(f"Training envelope : v0 {TRAIN_V0} m/s, theta {TRAIN_THETA_DEG} deg, "
          f"y0 {TRAIN_Y0} m")
    print(f"Training rows     : {TRAIN_ROWS:,}")
    print(f"Test regimes      : {len(REGIMES)} x {TEST_ROWS:,} rows each\n")

    train = generateProjectileDatasetInRange(
        TRAIN_ROWS, v0_range=TRAIN_V0, theta_deg_range=TRAIN_THETA_DEG,
        y0_range=TRAIN_Y0, seed=101,
    )
    X_train, y_train = train[FEATURES], train[TARGETS]

    models = {}
    for name in MODELS:
        start = time.perf_counter()
        models[name] = build_model(name).fit(X_train, y_train)
        print(f"  trained {MODEL_LABELS[name]:<28s} in {time.perf_counter() - start:6.2f} s")

    # ---------- experiment 1: the seven regimes ----------
    table = evaluate_everywhere(models)
    table.to_csv(RESULTS / "stage8_generalization.csv", index=False)

    print("\n--- mean radial error (m) by test regime ---")
    pivot = table.pivot(index="regime", columns="model", values="mean_radial")
    pivot = pivot.reindex(list(REGIMES), axis=0).reindex(MODELS, axis=1)
    # scientific notation, because the physics model's error sits at the
    # ~1e-14 m level -- floating-point round-off, i.e. exactly correct.
    print(pivot.to_string(float_format=lambda v: f"{v:10.3e}"))

    print("\n--- R^2 for y by test regime (negative = worse than predicting the mean) ---")
    pivot_r2 = table.pivot(index="regime", columns="model", values="r2_y")
    print(pivot_r2.reindex(list(REGIMES), axis=0).reindex(MODELS, axis=1).round(3).to_string())

    # ---------- experiment 2: distance beyond the edge ----------
    edges = np.concatenate([np.arange(15, 41, 5.0), np.arange(42.5, 81, 2.5)])
    curve = extrapolation_distance_curve(models, edges)
    curve.to_csv(RESULTS / "stage8_extrapolation_distance.csv", index=False)

    # ---------- experiment 3: forward in time ----------
    time_curve, t_train_high = time_extrapolation()
    time_curve.to_csv(RESULTS / "stage8_time_extrapolation.csv", index=False)

    print(f"\n--- mean radial error (m) vs position in flight "
          f"(trained on first {t_train_high:.0%}) ---")
    tp = time_curve.pivot(index="t_fraction_centre", columns="model", values="mean_radial")
    print(tp.reindex(MODELS, axis=1).round(3).to_string())

    # ---------- experiment 4: the 1-D response slice ----------
    slice_data = response_slice(models)
    slice_data.to_csv(RESULTS / "stage8_response_slice.csv", index=False)

    # ---------- figures ----------
    print("\nFigures:")
    plot_regime_bars(table, list(REGIMES), MODELS, RESULTS / "stage8_regime_bars.png")
    plot_extrapolation_distance(curve, MODELS, TRAIN_V0[1],
                                RESULTS / "stage8_extrapolation_distance.png")
    plot_time_extrapolation(time_curve, MODELS, t_train_high,
                            RESULTS / "stage8_time_extrapolation.png")
    plot_response_slice(slice_data, MODELS, TRAIN_V0,
                        RESULTS / "stage8_response_slice.png")
    plot_generalization_trajectories(trajectory_panels(models),
                                     RESULTS / "stage8_trajectories.png")

    # ---------- the headline number ----------
    control = table[table["regime"] == "interpolation"].set_index("model")["mean_radial"]
    worst = table[table["regime"] == "extrap_all"].set_index("model")["mean_radial"]
    print("\n--- degradation from interpolation to full extrapolation ---")
    for name in MODELS:
        factor = worst[name] / control[name]
        note = "  (both at machine precision)" if worst[name] < 1e-9 else ""
        print(f"  {MODEL_LABELS[name]:<28s} {control[name]:10.3e} m  ->  "
              f"{worst[name]:10.3e} m   ({factor:,.1f}x){note}")
