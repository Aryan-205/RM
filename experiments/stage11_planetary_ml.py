"""
Stage 11 -- Machine Learning for Planetary Motion

Research sub-questions answered here:
    4. Can the same general ML approach be applied to planetary motion?
    5. Does the increased complexity of planetary dynamics lead to greater
       prediction error?
    7. How does prediction error accumulate over time?

Experimental design
-------------------
  Independent variables : problem formulation (Approach A / B / C), target
                          parametrisation (absolute vs delta), model family,
                          training-set size, prediction horizon
  Dependent variables   : one-step position error, rollout position error as
                          a function of time, relative energy drift, usable
                          horizon at a fixed error tolerance
  Controlled            : the training trajectories (one fixed set for every
                          model), the RK4 reference and its step size, the
                          test orbits, GM, the rollout length

The reference (ground truth) is RK4 with ten substeps per stored point, which
Stage 10 established is accurate to ~4e-11 AU per orbit. Every ML error
reported below is at least seven orders of magnitude larger than that, so the
reference can safely be treated as exact.

Run:  python experiments/stage11_planetary_ml.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from data.generation.planetaryDataGeneration import (
    DT_STORE,
    SUBSTEPS,
    generate_trajectory_set,
    history_feature_names,
    make_history_pairs,
    make_one_step_pairs,
)
from plots.planetary_ml import (
    plot_approach_comparison,
    plot_direct_extrapolation,
    plot_error_growth,
    plot_ml_conservation,
    plot_one_step_accuracy,
    plot_planetary_learning_curve,
    plot_rollout_orbits,
    plot_sensitivity,
)
from plots.style import apply_style
from src.planetary_physics import (
    GM,
    kepler_solution,
    simulate_reference,
    state_from_elements,
)
from src.planetary_ml import (
    DIVERGENCE_RADIUS,
    NEXT_COLUMNS,
    STATE_COLUMNS,
    DirectModel,
    HistoryModel,
    OneStepModel,
    error_doubling_time,
    make_state_model,
    rollout_diagnostics,
    time_to_error,
)

RESULTS = ROOT / "results"
DATA = ROOT / "data"

MODELS = ["linear", "poly2", "forest", "mlp"]
HISTORY = 3

ROLLOUT_YEARS = 5.0
ROLLOUT_STEPS = int(round(ROLLOUT_YEARS / DT_STORE))   # 1,250 model calls

N_TEST_ORBITS = 20
REPRESENTATIVE = 0        # which test orbit to draw in the orbit figures

# Error tolerances at which we quote a "usable horizon". 0.01 AU is roughly
# 1.5 million km -- about four times the Earth-Moon distance, and a very
# generous tolerance for an orbit of radius 1 AU.
THRESHOLDS = [(0.001, "0.1% of 1 AU"), (0.01, "1% of 1 AU"), (0.1, "10% of 1 AU")]

TRAIN_SIZES = [500, 2000, 8000, 20000, 44000]

# The direct-map (Approach B) extrapolation test: trained on the first
# TRAIN_YEARS of each orbit, queried out to DIRECT_TEST_YEARS.
DIRECT_TEST_YEARS = 8.0


# -------------------- DATA --------------------

def load_or_make(name, **kwargs):
    path = DATA / f"planetary_{name}_trajectories.csv"
    if not path.exists():
        print(f"  generating {path.name} ...")
        generate_trajectory_set(**kwargs).to_csv(path, index=False)
    return pd.read_csv(path)


def test_initial_conditions(n=N_TEST_ORBITS, seed=2024):
    """Unseen orbits, drawn from the same (a, e) box the models were trained on."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.85, 1.15, n)
    e = rng.uniform(0.05, 0.35, n)
    states = np.stack([state_from_elements(ai, ei) for ai, ei in zip(a, e)])
    return a, e, states


def reference_rollouts(states, n_steps=ROLLOUT_STEPS):
    """RK4 ground-truth trajectories for every test initial condition."""
    truths = []
    for state0 in states:
        times, traj = simulate_reference(state0, DT_STORE, n_steps, SUBSTEPS)
        truths.append(traj)
    return times, np.stack(truths, axis=1)      # (n_steps+1, N, 4)


# -------------------- EXPERIMENT 1: ONE-STEP ACCURACY --------------------

def one_step_accuracy(pairs, test_pairs):
    """Fit every model in both target parametrisations; report one-step error."""
    X, Y = pairs[STATE_COLUMNS].to_numpy(), pairs[NEXT_COLUMNS].to_numpy()
    Xt, Yt = test_pairs[STATE_COLUMNS].to_numpy(), test_pairs[NEXT_COLUMNS].to_numpy()

    rows, fitted = [], {}
    for name in MODELS:
        for mode in ["absolute", "delta"]:
            model = OneStepModel(make_state_model(name), mode=mode).fit(X, Y)
            predicted = model.step(Xt)
            position_error = np.linalg.norm(predicted[:, 0:2] - Yt[:, 0:2], axis=1)
            velocity_error = np.linalg.norm(predicted[:, 2:4] - Yt[:, 2:4], axis=1)

            rows.append({
                "model": name, "mode": mode,
                "mean_position_error": float(position_error.mean()),
                "median_position_error": float(np.median(position_error)),
                "p95_position_error": float(np.percentile(position_error, 95)),
                "mean_velocity_error": float(velocity_error.mean()),
                "train_seconds": model.train_seconds,
            })
            fitted[(name, mode)] = model
            print(f"    {name:<7s} {mode:<9s} one-step position error = "
                  f"{position_error.mean():.3e} AU   ({model.train_seconds:.1f} s to train)")

    return pd.DataFrame(rows), fitted


# -------------------- EXPERIMENT 2: ROLLOUT --------------------

def rollout_curves(fitted, times, truths, states0):
    """
    Roll every model out over the full horizon for every test orbit, and
    summarise the error as a function of time across orbits.
    """
    curves, per_orbit, panels = [], [], {}

    for name in MODELS:
        model = fitted[(name, "delta")]
        start = time.perf_counter()
        predicted = model.rollout(states0, ROLLOUT_STEPS)     # (steps+1, N, 4)
        elapsed = time.perf_counter() - start

        error = np.linalg.norm(predicted[:, :, 0:2] - truths[:, :, 0:2], axis=2)

        # A NaN means the rollout escaped the system (see OneStepModel.rollout).
        # Those points are not missing data -- the error really is at least the
        # divergence radius -- so we substitute that lower bound rather than
        # dropping them, which would silently flatter an unstable model.
        escaped = ~np.isfinite(error)
        error_filled = np.where(escaped, DIVERGENCE_RADIUS, error)
        escape_fraction = escaped.any(axis=0).mean()

        curves.append(pd.DataFrame({
            "model": name, "t": times,
            "mean": error_filled.mean(axis=1),
            "median": np.median(error_filled, axis=1),
            "q25": np.percentile(error_filled, 25, axis=1),
            "q75": np.percentile(error_filled, 75, axis=1),
            "escaped_fraction": escaped.mean(axis=1),
        }))

        for j in range(error.shape[1]):
            column = error_filled[:, j]
            growth = error_doubling_time(times, column)
            escape_index = np.where(escaped[:, j])[0]
            row = {"model": name, "orbit": j,
                   "final_error": float(column[-1]),
                   "escaped": bool(escaped[:, j].any()),
                   "escape_time": float(times[escape_index[0]]) if len(escape_index)
                                  else np.inf,
                   **growth}
            for level, _ in THRESHOLDS:
                row[f"horizon_{level}"] = time_to_error(times, column, level)
            per_orbit.append(row)

        panels[name] = {
            "predicted": predicted[:, REPRESENTATIVE, :],
            "truth": truths[:, REPRESENTATIVE, :],
            "diagnostics": rollout_diagnostics(
                times, predicted[:, REPRESENTATIVE, :], truths[:, REPRESENTATIVE, :]
            ),
            "rollout_seconds": elapsed,
        }
        note = (f"   [{escape_fraction:.0%} of rollouts left the system]"
                if escape_fraction else "")
        print(f"    {name:<7s} rolled out {ROLLOUT_STEPS} steps x "
              f"{error.shape[1]} orbits in {elapsed:5.1f} s   "
              f"mean final error = {error_filled[-1].mean():.3f} AU{note}")

    return pd.concat(curves, ignore_index=True), pd.DataFrame(per_orbit), panels


def horizon_table(per_orbit):
    """Median usable horizon per model and tolerance (median, not mean,
    because an orbit that never crosses the threshold contributes inf)."""
    rows = []
    for name in MODELS:
        subset = per_orbit[per_orbit["model"] == name]
        for level, label in THRESHOLDS:
            values = subset[f"horizon_{level}"].to_numpy()
            finite = values[np.isfinite(values)]
            rows.append({
                "model": name, "threshold": level, "label": label,
                "time_to_threshold": float(np.median(finite)) if len(finite) else np.inf,
                "orbits_never_exceeding": int((~np.isfinite(values)).sum()),
            })
    return pd.DataFrame(rows)


# -------------------- EXPERIMENT 3: APPROACH B (DIRECT MAP) --------------------

def direct_map_experiment(train, a_test, e_test, times):
    """
    Fit (a, e, t) -> (x, y, vx, vy) and evaluate it over the same horizon,
    including well beyond the training time window.
    """
    features = train[["a", "e", "t"]].to_numpy()
    targets = train[STATE_COLUMNS].to_numpy()
    t_train_max = float(train["t"].max())

    query_times = np.arange(0, DIRECT_TEST_YEARS, DT_STORE)
    exact = np.stack([
        kepler_solution(query_times, a, e) for a, e in zip(a_test, e_test)
    ], axis=1)

    rows, curves = [], []
    for name in MODELS:
        model = DirectModel(make_state_model(name)).fit(features, targets)

        errors = np.empty((len(query_times), len(a_test)))
        for j, (a, e) in enumerate(zip(a_test, e_test)):
            frame = np.column_stack([
                np.full_like(query_times, a), np.full_like(query_times, e), query_times
            ])
            predicted = model.predict(frame)
            errors[:, j] = np.linalg.norm(predicted[:, 0:2] - exact[:, j, 0:2], axis=1)

        curves.append(pd.DataFrame({
            "model": name, "t": query_times, "mean_error": errors.mean(axis=1),
        }))

        inside = query_times <= t_train_max
        rows.append({
            "model": name,
            "mean_error_in_window": float(errors[inside].mean()),
            "mean_error_beyond_window": float(errors[~inside].mean()),
            "train_seconds": model.train_seconds,
        })
        print(f"    {name:<7s} inside training window: {errors[inside].mean():.3e} AU   "
              f"beyond it: {errors[~inside].mean():.3e} AU")

    # Same-horizon slice, for the three-approach comparison figure
    within = query_times <= times.max()
    comparison = []
    best = min(rows, key=lambda r: r["mean_error_in_window"])["model"]
    errors_best = [c for c in curves if c["model"].iloc[0] == best][0]
    comparison.append(pd.DataFrame({
        "approach": "B_direct",
        "t": errors_best["t"][within].to_numpy(),
        "mean": errors_best["mean_error"][within].to_numpy(),
        "q25": errors_best["mean_error"][within].to_numpy(),
        "q75": errors_best["mean_error"][within].to_numpy(),
    }))

    return (pd.DataFrame(rows), pd.concat(curves, ignore_index=True),
            pd.concat(comparison, ignore_index=True), t_train_max, best)


# -------------------- EXPERIMENT 4: APPROACH C (HISTORY) --------------------

def history_experiment(train, times, truths, model_names=("forest", "mlp")):
    """
    Same trajectories, same models, but the input is a window of HISTORY
    consecutive states rather than one.
    """
    windows = make_history_pairs(train, history=HISTORY)
    feature_names = history_feature_names(HISTORY)
    X = windows[feature_names].to_numpy()
    Y = windows[[f"next_{c}" for c in STATE_COLUMNS]].to_numpy()

    rows, curves = [], []
    for name in model_names:
        model = HistoryModel(make_state_model(name), history=HISTORY).fit(X, Y)

        # The rollout needs HISTORY true states to start from. We take them
        # from the reference trajectory and do not score them.
        errors = np.empty((len(times) - HISTORY + 1, truths.shape[1]))
        for j in range(truths.shape[1]):
            warmup = truths[:HISTORY, j, :]
            predicted = model.rollout(warmup, len(times) - HISTORY)
            step_error = np.linalg.norm(
                predicted[HISTORY - 1:, 0:2] - truths[HISTORY - 1:, j, 0:2], axis=1
            )
            errors[:, j] = np.where(np.isfinite(step_error), step_error,
                                    DIVERGENCE_RADIUS)

        scored_times = times[HISTORY - 1:]
        curves.append(pd.DataFrame({
            "model": name, "t": scored_times,
            "mean": errors.mean(axis=1),
            "q25": np.percentile(errors, 25, axis=1),
            "q75": np.percentile(errors, 75, axis=1),
        }))
        rows.append({
            "model": name, "history": HISTORY,
            "final_error": float(errors[-1].mean()),
            "horizon_0.01AU": time_to_error(scored_times, errors.mean(axis=1), 0.01),
            "train_seconds": model.train_seconds,
        })
        print(f"    {name:<7s} history={HISTORY}: mean final error = "
              f"{errors[-1].mean():.3f} AU")

    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


# -------------------- EXPERIMENT 5: LEARNING CURVE --------------------

def planetary_learning_curve(pairs, test_pairs, times, truths, states0):
    """
    How do one-step accuracy and usable horizon respond to more data?

    Subsets are drawn at random from the full pool rather than taking whole
    trajectories, so every subset still covers the whole (a, e) range and the
    only thing changing is the density of the sampling.
    """
    X_all, Y_all = pairs[STATE_COLUMNS].to_numpy(), pairs[NEXT_COLUMNS].to_numpy()
    Xt, Yt = test_pairs[STATE_COLUMNS].to_numpy(), test_pairs[NEXT_COLUMNS].to_numpy()
    rng = np.random.default_rng(5)

    rows = []
    for n_train in TRAIN_SIZES:
        index = rng.choice(len(X_all), size=min(n_train, len(X_all)), replace=False)
        X, Y = X_all[index], Y_all[index]

        for name in MODELS:
            model = OneStepModel(make_state_model(name), mode="delta").fit(X, Y)

            one_step = float(np.linalg.norm(
                model.step(Xt)[:, 0:2] - Yt[:, 0:2], axis=1).mean())

            predicted = model.rollout(states0, ROLLOUT_STEPS)
            error = np.linalg.norm(predicted[:, :, 0:2] - truths[:, :, 0:2], axis=2)
            error = np.where(np.isfinite(error), error, DIVERGENCE_RADIUS).mean(axis=1)

            row = {"n_train": len(index), "model": name,
                   "one_step_error": one_step,
                   "final_error": float(error[-1]),
                   "train_seconds": model.train_seconds}
            for level, _ in THRESHOLDS:
                row[f"horizon_{level}AU"] = time_to_error(times, error, level)
            rows.append(row)

        print(f"    n_train = {len(index):>6,} done")

    return pd.DataFrame(rows)


# -------------------- EXPERIMENT 6: SENSITIVITY TO INITIAL CONDITIONS ------

def sensitivity_experiment(a=1.0, e=0.25, offsets=(1e-10, 1e-8, 1e-6)):
    """
    Integrate the EXACT dynamics from slightly different initial conditions
    and measure how the two trajectories separate. No machine learning here
    at all -- this quantifies how much of the rollout divergence the physics
    itself is responsible for.
    """
    base = state_from_elements(a, e)
    query = np.arange(0, ROLLOUT_YEARS, DT_STORE)
    reference = kepler_solution(query, a, e)

    growth = []
    for delta in offsets:
        perturbed0 = base.copy()
        perturbed0[0] += delta       # nudge the starting x position
        _, perturbed = simulate_reference(perturbed0, DT_STORE, len(query) - 1, SUBSTEPS)
        separation = np.linalg.norm(perturbed[:, 0:2] - reference[:, 0:2], axis=1)
        growth.append((delta, query, separation))

        fit = error_doubling_time(query, separation)
        print(f"    offset {delta:.0e} AU -> separation after {ROLLOUT_YEARS:.0f} yr = "
              f"{separation[-1]:.3e} AU   (amplification {separation[-1] / delta:,.0f}x)")

    return growth


# -------------------- MAIN --------------------

if __name__ == "__main__":
    apply_style()
    RESULTS.mkdir(exist_ok=True)

    print("=" * 72)
    print("STAGE 11 -- MACHINE LEARNING FOR PLANETARY MOTION")
    print("=" * 72)

    train = load_or_make("train", n_trajectories=60, n_orbits=3.0, seed=11, start_id=0)
    test = load_or_make("test", n_trajectories=20, n_orbits=3.0, seed=99, start_id=1000)
    pairs = make_one_step_pairs(train)
    test_pairs = make_one_step_pairs(test)

    print(f"Training trajectories : {train['traj_id'].nunique()} "
          f"({len(pairs):,} one-step pairs)")
    print(f"Held-out trajectories : {test['traj_id'].nunique()} "
          f"({len(test_pairs):,} one-step pairs)")
    print(f"Rollout horizon       : {ROLLOUT_YEARS} yr = {ROLLOUT_STEPS:,} model calls")
    print(f"Step size             : dt = {DT_STORE} yr "
          f"({int(1 / DT_STORE)} steps per 1-year orbit)\n")

    a_test, e_test, states0 = test_initial_conditions()
    times, truths = reference_rollouts(states0)

    # reference error of the numerical baseline itself
    exact0 = kepler_solution(times, a_test[REPRESENTATIVE], e_test[REPRESENTATIVE])
    numerical_error = float(np.linalg.norm(
        truths[:, REPRESENTATIVE, 0:2] - exact0[:, 0:2], axis=1).max())
    print(f"RK4 reference error vs analytic Kepler over the horizon: "
          f"{numerical_error:.2e} AU\n")

    # ---- 1
    print("[1] One-step accuracy (Approach A), absolute vs delta targets")
    one_step, fitted = one_step_accuracy(pairs, test_pairs)
    one_step.to_csv(RESULTS / "stage11_one_step_accuracy.csv", index=False)

    # ---- 2
    print("\n[2] Autoregressive rollout (Approach A, delta targets)")
    curves, per_orbit, panels = rollout_curves(fitted, times, truths, states0)
    curves.to_csv(RESULTS / "stage11_rollout_error.csv", index=False)
    per_orbit.to_csv(RESULTS / "stage11_rollout_per_orbit.csv", index=False)

    horizons = horizon_table(per_orbit)
    horizons.to_csv(RESULTS / "stage11_horizons.csv", index=False)
    curves.attrs["horizon_table"] = horizons

    print("\n    median usable horizon (years) before exceeding a tolerance:")
    print("    " + horizons.pivot(index="model", columns="label",
                                  values="time_to_threshold")
          .reindex(MODELS).round(3).to_string().replace("\n", "\n    "))

    print("\n    exponential-growth fit of the rollout error:")
    for name in MODELS:
        subset = per_orbit[per_orbit["model"] == name]
        escaped = subset["escaped"].mean()
        print(f"      {name:<7s} e-folding time = {subset['e_folding_time'].median():6.3f} yr, "
              f"log-linear fit R^2 = {subset['r_squared'].median():.3f}"
              + (f", {escaped:.0%} of rollouts unstable" if escaped else ""))

    # ---- 3
    print("\n[3] Approach B -- learned closed form (a, e, t) -> state")
    direct_summary, direct_curve, direct_comparison, t_train_max, best_direct = \
        direct_map_experiment(train, a_test, e_test, times)
    direct_summary.to_csv(RESULTS / "stage11_direct_map.csv", index=False)
    direct_curve.to_csv(RESULTS / "stage11_direct_map_curve.csv", index=False)

    # ---- 4
    print(f"\n[4] Approach C -- {HISTORY}-state history window")
    history_summary, history_curve = history_experiment(train, times, truths)
    history_summary.to_csv(RESULTS / "stage11_history.csv", index=False)

    # ---- 5
    print("\n[5] Learning curve: does more data extend the horizon?")
    learning = planetary_learning_curve(pairs, test_pairs, times, truths, states0)
    learning.to_csv(RESULTS / "stage11_learning_curve.csv", index=False)
    print("\n    one-step error (AU) / horizon at 0.01 AU (yr):")
    for name in MODELS:
        rows = learning[learning["model"] == name].sort_values("n_train")
        cells = "  ".join(
            f"n={int(r['n_train']):>5,}: {r['one_step_error']:.1e}/{r['horizon_0.01AU']:.2f}"
            for _, r in rows.iterrows()
        )
        print(f"      {name:<7s} {cells}")

    # ---- 6
    print("\n[6] Sensitivity of the EXACT dynamics to initial conditions")
    true_growth = sensitivity_experiment()

    # -------------------- FIGURES --------------------
    print("\nFigures:")

    plot_one_step_accuracy(one_step, MODELS, RESULTS / "stage11_one_step_accuracy.png")

    orbit_panels = [{
        "model": name,
        "predicted": panels[name]["predicted"],
        "truth": panels[name]["truth"],
        "final_error": float(np.linalg.norm(
            panels[name]["predicted"][-1, 0:2] - panels[name]["truth"][-1, 0:2])),
        "final_energy_error": float(
            panels[name]["diagnostics"]["energy_rel_error"].iloc[-1]),
        "escape_time": float(per_orbit[(per_orbit["model"] == name)
                                       & (per_orbit["orbit"] == REPRESENTATIVE)]
                             ["escape_time"].iloc[0]),
        "a": a_test[REPRESENTATIVE], "e": e_test[REPRESENTATIVE],
    } for name in MODELS]
    plot_rollout_orbits(orbit_panels, ROLLOUT_YEARS, RESULTS / "stage11_rollout_orbits.png")

    plot_error_growth(curves, MODELS, numerical_error, THRESHOLDS,
                      RESULTS / "stage11_error_growth.png")

    representative_rollouts = pd.concat(
        [panels[name]["diagnostics"].assign(model=name) for name in MODELS],
        ignore_index=True,
    )
    representative_rollouts.to_csv(
        RESULTS / "stage11_representative_rollout.csv", index=False)

    conservation_panels = {name: panels[name]["diagnostics"] for name in MODELS}
    conservation_panels["numerical_energy_error"] = 1e-11
    plot_ml_conservation(conservation_panels, MODELS,
                         RESULTS / "stage11_conservation_ml.png")

    best_flow = one_step[one_step["mode"] == "delta"].nsmallest(
        1, "mean_position_error")["model"].iloc[0]
    approach_curves = pd.concat([
        curves[curves["model"] == best_flow]
        .assign(approach="A_flow_map")[["approach", "t", "mean", "q25", "q75"]],
        direct_comparison,
        history_curve[history_curve["model"] == best_flow]
        .assign(approach="C_history")[["approach", "t", "mean", "q25", "q75"]]
        if best_flow in set(history_curve["model"]) else
        history_curve[history_curve["model"] == "mlp"]
        .assign(approach="C_history")[["approach", "t", "mean", "q25", "q75"]],
    ], ignore_index=True)
    approach_curves.to_csv(RESULTS / "stage11_approach_comparison.csv", index=False)
    plot_approach_comparison(approach_curves,
                             RESULTS / "stage11_approach_comparison.png")

    plot_direct_extrapolation(direct_curve, t_train_max,
                              RESULTS / "stage11_direct_extrapolation.png")

    learning_renamed = learning.rename(
        columns={f"horizon_{level}AU": f"horizon_{level}AU" for level, _ in THRESHOLDS}
    )
    plot_planetary_learning_curve(learning_renamed, MODELS,
                                  RESULTS / "stage11_learning_curve.png")

    ml_growth = {
        name: (curves[curves["model"] == name]["t"].to_numpy(),
               curves[curves["model"] == name]["mean"].to_numpy())
        for name in MODELS
    }
    plot_sensitivity(true_growth, ml_growth, RESULTS / "stage11_sensitivity.png")

    print("\nHeadline numbers for the report:")
    best_row = one_step[one_step["mode"] == "delta"].nsmallest(1, "mean_position_error")
    print(f"  Best one-step accuracy : {best_row['model'].iloc[0]} at "
          f"{best_row['mean_position_error'].iloc[0]:.2e} AU per step")
    h = horizons[(horizons["model"] == best_row["model"].iloc[0])
                 & (horizons["threshold"] == 0.01)]
    print(f"  ...yet it drifts past 0.01 AU after only "
          f"{h['time_to_threshold'].iloc[0]:.2f} years "
          f"({h['time_to_threshold'].iloc[0] / 1.0:.2f} orbits)")
    print(f"  RK4, by contrast, holds {numerical_error:.0e} AU over the whole horizon.")
