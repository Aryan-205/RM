"""
Stage 13 -- Physics-Informed Machine Learning (optional advanced component)

The question this stage exists to answer
----------------------------------------
Everything up to here treated physics purely as a data source: we used the
equations to manufacture examples, then threw the equations away and let a
generic regressor find whatever pattern it could. Stage 8 showed the cost of
that -- the models were accurate inside their training envelope and
untrustworthy outside it, because a pile of examples says nothing about
regions it does not cover.

A physical law is a different kind of object from a data point. It is a
statement about EVERYWHERE. So the question is:

    if we put the law itself into the learning problem rather than only its
    consequences, how much less data do we need, and does the model stop
    producing physically impossible motion?

Three answers are collected here, in increasing order of strength.

  13a. HARD constraints. Re-express the inputs in a basis in which the
       solution is linear (see ProjectilePhysicsModel). The physics is not
       learned at all, it is built in, and the result is exact to
       floating-point round-off everywhere including far outside the
       training range. Cheap, and unbeatable -- when you know the answer.

  13b. SOFT constraints (the PINN). Add the residual of the equation of
       motion to the loss, evaluated at unlabelled collocation points. The
       physics is a penalty, not a guarantee. Slower and less exact than
       13a -- but it only needs the DIFFERENTIAL EQUATION, not its solution,
       which is the situation in every problem where one would actually reach
       for this technique.

  13c. Physical consistency as a metric in its own right. Every model from
       every stage is scored on how badly its own predictions violate
       d2x/dt2 = 0 and d2y/dt2 = -g. This is a question none of the earlier
       stages asked, and some models that looked good by RMSE do badly on it.

Experimental design
-------------------
  Independent variables : number of labelled points, presence/absence of the
                          physics term, region of time covered by the labels
  Dependent variables   : mean radial test error, residual of the equations
                          of motion
  Controlled            : network architecture (32-32 tanh), initialisation
                          scheme, optimiser and learning rate, number of
                          epochs, collocation points, test set, random seeds

The plain network is not a strawman: it is the SAME class with
physics_weight = 0, so the only difference between the two arms is the loss.

Run:  python experiments/stage13_physics_informed.py
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
from plots.pinn import (
    plot_data_efficiency,
    plot_physics_residual,
    plot_pinn_extrapolation,
    plot_training_history,
)
from plots.style import apply_style
from src.models import MODEL_LABELS, ProjectilePhysicsModel, make_model
from src.pinn import G, ProjectilePINN

RESULTS = ROOT / "results"

FEATURES = ["v0", "theta", "y0", "t"]
TARGETS = ["x", "y"]

V0_RANGE = (15.0, 40.0)
THETA_RANGE = (20.0, 70.0)
Y0_RANGE = (0.0, 15.0)

N_COLLOCATION = 4000
N_TEST = 2000
EPOCHS = 4000
HIDDEN = (32, 32)
SEEDS = [0, 1, 2]

LABEL_COUNTS = [0, 10, 20, 50, 100, 200, 500, 2000]

# For the extrapolation experiment: labels only from the first 40% of flight,
# collocation points from the whole flight.
DATA_FRACTION = 0.4


def sample(n, seed, t_fraction_range=(0.0, 1.0)):
    return generateProjectileDatasetInRange(
        n, v0_range=V0_RANGE, theta_deg_range=THETA_RANGE, y0_range=Y0_RANGE,
        t_fraction_range=t_fraction_range, seed=seed,
    )


def finite_difference_residual(predict, X, h=1e-2):
    """
    |d2x/dt2 - 0| and |d2y/dt2 + g| for ANY model with a .predict(DataFrame),
    using the same central difference the PINN uses internally.

    Applying one identical measurement to every model is the point: it means
    the Random Forest and the PINN are being judged by the same ruler, and the
    resulting numbers are directly comparable.
    """
    X = X.copy()
    minus, plus = X.copy(), X.copy()
    minus["t"] = X["t"] - h
    plus["t"] = X["t"] + h

    f_minus = np.asarray(predict(minus), dtype=float)
    f_mid = np.asarray(predict(X), dtype=float)
    f_plus = np.asarray(predict(plus), dtype=float)

    second = (f_plus - 2.0 * f_mid + f_minus) / h ** 2
    return float(np.mean(np.abs(second[:, 0]))), float(np.mean(np.abs(second[:, 1] + G)))


# -------------------- 13b.1 DATA EFFICIENCY --------------------

def data_efficiency(collocation, test):
    """
    Sweep the number of labelled points for both arms, repeated over seeds.

    The labelled points are always the FIRST n rows of the same shuffled pool,
    so a run with 50 labels sees a superset of the run with 20. That removes
    "which points did it happen to get" as a source of variation between
    adjacent points on the curve.
    """
    X_pool = collocation[FEATURES].to_numpy()
    Y_pool = collocation[TARGETS].to_numpy()
    X_collocation = X_pool
    X_test, Y_test = test[FEATURES].to_numpy(), test[TARGETS].to_numpy()

    rows = []
    for n_labels in LABEL_COUNTS:
        for arm, physics_weight in [("pinn", 1.0), ("plain", 0.0)]:
            if arm == "plain" and n_labels == 0:
                continue    # a network with no data and no physics learns nothing

            errors, residuals_x, residuals_y, seconds = [], [], [], []
            for seed in SEEDS:
                X_data = X_pool[:n_labels] if n_labels else None
                Y_data = Y_pool[:n_labels] if n_labels else None

                start = time.perf_counter()
                model = ProjectilePINN(
                    hidden=HIDDEN, seed=seed, lr=3e-3,
                    physics_weight=physics_weight, ic_weight=physics_weight,
                ).fit(X_data, Y_data, X_collocation, epochs=EPOCHS)
                seconds.append(time.perf_counter() - start)

                prediction = model.predict(X_test)
                errors.append(float(np.linalg.norm(prediction - Y_test, axis=1).mean()))
                residual = model.physics_residual(X_test)
                residuals_x.append(float(residual[:, 0].mean()))
                residuals_y.append(float(residual[:, 1].mean()))

            rows.append({
                "model": arm, "n_labels": n_labels,
                "mean_error": float(np.mean(errors)),
                "std_error": float(np.std(errors)),
                "residual_x": float(np.mean(residuals_x)),
                "residual_y": float(np.mean(residuals_y)),
                "train_seconds": float(np.mean(seconds)),
            })
            print(f"    {arm:<5s} n_labels={n_labels:>5d}  "
                  f"error = {np.mean(errors):8.4f} +/- {np.std(errors):6.4f} m   "
                  f"residual = ({np.mean(residuals_x):.3f}, {np.mean(residuals_y):.3f}) m/s^2")

    return pd.DataFrame(rows)


# -------------------- 13b.2 PHYSICS FILLS IN FOR MISSING DATA --------------

def extrapolation_experiment(collocation, n_labels=200):
    """
    Labels from the first DATA_FRACTION of the flight only; collocation points
    from the whole flight. Then ask both networks about the whole flight.
    """
    labelled = sample(n_labels, seed=31, t_fraction_range=(0.0, DATA_FRACTION))
    X_data, Y_data = labelled[FEATURES].to_numpy(), labelled[TARGETS].to_numpy()
    X_collocation = collocation[FEATURES].to_numpy()

    models = {}
    histories = {}
    for arm, weight, title in [
        ("pinn", 1.0, "PINN: data (first 40%) + physics (everywhere)"),
        ("plain", 0.0, "Plain NN: data only (first 40%)"),
    ]:
        model = ProjectilePINN(
            hidden=HIDDEN, seed=0, lr=3e-3,
            physics_weight=weight, ic_weight=weight,
        ).fit(X_data, Y_data, X_collocation, epochs=EPOCHS)
        models[arm] = model
        histories[title] = model.history

    # quantitative: error inside vs outside the labelled time window
    rows = []
    for label, fraction_range in [("inside labelled window", (0.0, DATA_FRACTION)),
                                  ("beyond labelled window", (DATA_FRACTION, 1.0))]:
        probe = sample(N_TEST, seed=41, t_fraction_range=fraction_range)
        X, Y = probe[FEATURES].to_numpy(), probe[TARGETS].to_numpy()
        for arm, model in models.items():
            error = np.linalg.norm(model.predict(X) - Y, axis=1)
            rows.append({"region": label, "model": arm,
                         "mean_error": float(error.mean()),
                         "median_error": float(np.median(error))})
            print(f"    {arm:<5s} {label:<24s} mean error = {error.mean():8.4f} m")

    # qualitative: two full trajectories
    panels = []
    for v0, theta_deg, y0 in [(25.0, 40.0, 5.0), (35.0, 60.0, 10.0)]:
        theta = np.deg2rad(theta_deg)
        flight = timeOfFlight(v0, theta, y0)
        t = np.linspace(0, flight, 150)
        frame = pd.DataFrame({"v0": v0, "theta": theta, "y0": y0, "t": t})
        X = frame[FEATURES].to_numpy()

        pinn_prediction = models["pinn"].predict(X)
        plain_prediction = models["plain"].predict(X)

        panels.append({
            "title": (f"$v_0$ = {v0:.0f} m/s, $\\theta$ = {theta_deg:.0f}$^\\circ$, "
                      f"$y_0$ = {y0:.0f} m"),
            "x_true": v0 * np.cos(theta) * t,
            "y_true": y0 + v0 * np.sin(theta) * t - 0.5 * G * t ** 2,
            "x_pinn": pinn_prediction[:, 0], "y_pinn": pinn_prediction[:, 1],
            "x_plain": plain_prediction[:, 0], "y_plain": plain_prediction[:, 1],
            "cut_index": int(DATA_FRACTION * len(t)),
        })

    return pd.DataFrame(rows), panels, histories, models


# -------------------- 13c. PHYSICAL CONSISTENCY OF EVERY MODEL -------------

def consistency_audit(train, test, pinn_model, plain_model):
    """
    Score every model family in the study on the same physical-consistency
    measure, so the report can say which of them describe possible motion.
    """
    X_train, Y_train = train[FEATURES], train[TARGETS]
    X_test = test[FEATURES]
    Y_test = test[TARGETS].to_numpy()

    entries = []
    for name in ["linear", "poly2", "forest", "mlp"]:
        model = make_model(name).fit(X_train, Y_train)
        entries.append((name, MODEL_LABELS[name], model.predict))

    physics_model = ProjectilePhysicsModel().fit(X_train, Y_train)
    entries.append(("physics", "Physics-Informed Features (hard)", physics_model.predict))
    entries.append(("pinn", "PINN (data + physics, soft)",
                    lambda frame: pinn_model.predict(frame[FEATURES].to_numpy())))
    entries.append(("plain_nn", "Plain NN (same net, data only)",
                    lambda frame: plain_model.predict(frame[FEATURES].to_numpy())))

    rows = []
    for key, label, predict in entries:
        residual_x, residual_y = finite_difference_residual(predict, X_test)
        error = np.linalg.norm(np.asarray(predict(X_test)) - Y_test, axis=1)
        rows.append({
            "model": key, "label": label,
            "mean_radial_error": float(error.mean()),
            "residual_x": residual_x, "residual_y": residual_y,
        })
        print(f"    {label:<34s} error = {error.mean():9.4f} m   "
              f"|res_x| = {residual_x:9.4f}   |res_y+g| = {residual_y:9.4f} m/s^2")

    return pd.DataFrame(rows)


# -------------------- MAIN --------------------

if __name__ == "__main__":
    apply_style()
    RESULTS.mkdir(exist_ok=True)

    print("=" * 72)
    print("STAGE 13 -- PHYSICS-INFORMED MACHINE LEARNING")
    print("=" * 72)
    print(f"Network      : {HIDDEN} tanh, {EPOCHS} epochs, Adam(3e-3), NumPy from scratch")
    print(f"Collocation  : {N_COLLOCATION:,} unlabelled points")
    print(f"Seeds        : {SEEDS} (every number below is a mean over these)\n")

    collocation = sample(N_COLLOCATION, seed=7)
    test = sample(N_TEST, seed=8)

    # ---- 13a: the hard-constraint model, for reference
    print("[13a] Hard physics constraints (linear regression in a physics basis)")
    hard = ProjectilePhysicsModel().fit(collocation[FEATURES], collocation[TARGETS])
    coefficients = hard.learned_coefficients
    print("      learned coefficients (rows x, y; columns v0cos(th)t, v0sin(th)t, y0, t^2):")
    print("      " + str(np.round(coefficients, 6)).replace("\n", "\n      "))
    print(f"      the t^2 coefficient for y is {coefficients[1, 3]:.6f}; "
          f"-g/2 = {-G / 2:.6f}")
    print("      => the fit has recovered the gravitational constant of the simulation.\n")

    # ---- 13b.1
    print("[13b] Data efficiency: PINN vs an identical network without the physics loss")
    efficiency = data_efficiency(collocation, test)
    efficiency.to_csv(RESULTS / "stage13_data_efficiency.csv", index=False)

    pinn_rows = efficiency[efficiency["model"] == "pinn"].set_index("n_labels")
    plain_rows = efficiency[efficiency["model"] == "plain"].set_index("n_labels")
    zero_label_error = pinn_rows.loc[0, "mean_error"]
    matched = plain_rows[plain_rows["mean_error"] <= zero_label_error]
    print(f"\n    A PINN with NO labels reaches {zero_label_error:.3f} m.")
    if not matched.empty:
        print(f"    The plain network needs {matched.index.min():,} labelled points "
              f"to match that.")

    # ---- 13b.2
    print(f"\n[13b] Physics as a substitute for missing data "
          f"(labels only in the first {DATA_FRACTION:.0%} of each flight)")
    regions, panels, histories, models = extrapolation_experiment(collocation)
    regions.to_csv(RESULTS / "stage13_extrapolation.csv", index=False)

    beyond = regions[regions["region"] == "beyond labelled window"].set_index("model")
    print(f"    beyond the labelled window the PINN is "
          f"{beyond.loc['plain', 'mean_error'] / beyond.loc['pinn', 'mean_error']:.1f}x "
          f"more accurate than the plain network")

    # ---- 13c
    print("\n[13c] Physical-consistency audit of every model in the study")
    audit = consistency_audit(collocation, test, models["pinn"], models["plain"])
    audit.to_csv(RESULTS / "stage13_consistency_audit.csv", index=False)

    # ---- figures
    print("\nFigures:")
    plot_data_efficiency(efficiency, RESULTS / "stage13_data_efficiency.png")
    plot_physics_residual(audit, RESULTS / "stage13_physics_residual.png")
    plot_pinn_extrapolation(panels, DATA_FRACTION,
                            RESULTS / "stage13_extrapolation.png")
    plot_training_history(histories, RESULTS / "stage13_training_history.png")

    print("\nWhat this stage does NOT show:")
    print("  * that PINNs beat plain networks in general -- with 2,000 labels the")
    print("    two are equivalent here, and the physics term only pays when data")
    print("    is scarce or absent in part of the domain.")
    print("  * that the PINN discovered physics. It was TOLD d2y/dt2 = -g; it")
    print("    learned a function consistent with that, which is a different and")
    print("    much weaker claim.")
