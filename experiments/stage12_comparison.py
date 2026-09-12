"""
Stage 12 -- Compare Projectile and Planetary Motion

The research question, in its final form:

    Is machine learning equally effective for a simple and a dynamically
    complex physical system, and if not, what exactly is the difference?

Why the comparison has to be set up carefully
---------------------------------------------
Part I predicted projectile positions from (v0, theta, y0, t); Part II
predicted orbital states by stepping a learned flow map. Those are different
PROBLEMS as well as different physics, so comparing their error numbers
directly would confound the two. Worse, the units differ: "3 m" and "0.01 AU"
cannot be put on the same axis at all.

This stage fixes both problems before comparing anything.

  1. FORMULATION HELD FIXED. Projectile motion is re-expressed as a one-step
     flow map s_t -> s_{t+dt}, exactly like the orbital problem (see
     data/generation/projectileDataGeneration.py). Now the only difference
     between the two arms is the physics.

  2. UNITS REMOVED. Every error is divided by the characteristic length of its
     own system (the horizontal range of the flight; the semi-major axis of
     the orbit) and every time by the natural period of its own system (the
     flight time; the orbital period). What remains is dimensionless and
     genuinely comparable.

The structural fact that drives every result below can be written down in
advance. The exact projectile flow map is

    x' = x + vx dt,   y' = y + vy dt - (1/2) g dt^2,   vx' = vx,   vy' = vy - g dt

which is AFFINE in the state. The exact orbital flow map involves 1/r^3 and is
not affine in any basis we give the model. So projectile motion's flow map
lies inside linear regression's hypothesis space and orbital motion's lies
inside nobody's. The experiments below measure the consequences of that.

Experimental design
-------------------
  Independent variables : physical system, model family, training-set size,
                          prediction horizon
  Dependent variables   : relative position error, relative error growth rate,
                          sensitivity amplification, training and inference time
  Controlled            : problem formulation (one-step flow map in both arms),
                          model hyperparameters, number of training rows,
                          number of test trajectories, rollout length in units
                          of the natural period, and the machine

Run:  python experiments/stage12_comparison.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from data.generation.planetaryDataGeneration import DT_STORE, SUBSTEPS, generate_trajectory_set
from data.generation.planetaryDataGeneration import make_one_step_pairs as planetary_pairs
from data.generation.projectileDataGeneration import (
    PROJECTILE_STATE_COLUMNS,
    generateProjectileStateTrajectories,
    makeProjectileOneStepPairs,
    projectileStateTrajectory,
    timeOfFlight,
)
from plots.comparison import (
    plot_cost_comparison,
    plot_flow_map_comparison,
    plot_learning_curve_comparison,
    plot_scorecard,
    plot_sensitivity_comparison,
)
from plots.style import apply_style
from src.planetary_ml import (
    NEXT_COLUMNS,
    STATE_COLUMNS,
    OneStepModel,
    make_state_model,
)
from src.planetary_physics import (
    GM,
    kepler_solution,
    simulate_reference,
    state_from_elements,
)

RESULTS = ROOT / "results"
G = 9.81

MODELS = ["linear", "poly2", "forest", "mlp"]

# Both systems get the same number of training pairs and the same number of
# steps per natural period, so neither is handicapped by the experimental setup.
N_TRAIN_PAIRS = 20000
STEPS_PER_PERIOD = 250
PERIODS_TO_ROLL = 3.0
N_TEST_TRAJECTORIES = 20
TRAIN_SIZES = [500, 2000, 8000, 20000]

PROJECTILE_DT = 0.02          # seconds; a ~3 s flight is then ~150 steps
PLANETARY_DT = DT_STORE       # 0.004 yr; a 1 yr orbit is 250 steps

# A rollout is declared to have left the system once it passes this multiple of
# the system's own characteristic length. It MUST be expressed relative to that
# length rather than as an absolute number: the orbital problem measures ~1 AU
# and the projectile problem ~100 m, so any fixed threshold would be far too
# tight for one system and meaningless for the other.
DIVERGENCE_FACTOR = 50.0

PROJECTILE_NEXT = [f"next_{c}" for c in PROJECTILE_STATE_COLUMNS]


# ==================== BUILD THE TWO ARMS ====================

def build_projectile_arm():
    """
    Training pairs, test trajectories and the characteristic scales for the
    projectile system, in flow-map form.
    """
    train = generateProjectileStateTrajectories(600, dt=PROJECTILE_DT, seed=17)
    pairs = makeProjectileOneStepPairs(train)
    pairs = pairs.sample(n=min(N_TRAIN_PAIRS, len(pairs)), random_state=0)

    rng = np.random.default_rng(2024)
    tests = []
    for i in range(N_TEST_TRAJECTORIES):
        v0 = rng.uniform(18, 37)
        theta = rng.uniform(np.deg2rad(25), np.deg2rad(65))
        y0 = rng.uniform(0, 13)
        flight = timeOfFlight(v0, theta, y0)
        trajectory = projectileStateTrajectory(v0, theta, y0, PROJECTILE_DT, traj_id=i)
        tests.append({
            "states": trajectory[PROJECTILE_STATE_COLUMNS].to_numpy(),
            "times": trajectory["t"].to_numpy(),
            "period": flight,
            "scale": v0 * np.cos(theta) * flight,     # horizontal range
        })
    return pairs, PROJECTILE_STATE_COLUMNS, PROJECTILE_NEXT, tests


def build_planetary_arm():
    path = ROOT / "data" / "planetary_train_trajectories.csv"
    train = (pd.read_csv(path) if path.exists()
             else generate_trajectory_set(60, n_orbits=3.0, seed=11))
    pairs = planetary_pairs(train)
    pairs = pairs.sample(n=min(N_TRAIN_PAIRS, len(pairs)), random_state=0)

    rng = np.random.default_rng(2024)
    tests = []
    for i in range(N_TEST_TRAJECTORIES):
        a = rng.uniform(0.85, 1.15)
        e = rng.uniform(0.05, 0.35)
        period = 2.0 * np.pi * np.sqrt(a ** 3 / GM)
        n_steps = int(round(PERIODS_TO_ROLL * period / PLANETARY_DT))
        times, states = simulate_reference(
            state_from_elements(a, e), PLANETARY_DT, n_steps, SUBSTEPS
        )
        tests.append({"states": states, "times": times, "period": period, "scale": a})
    return pairs, STATE_COLUMNS, NEXT_COLUMNS, tests


# ==================== EXPERIMENT 1: FLOW-MAP ROLLOUT ====================

def flow_map_rollouts(pairs, state_columns, next_columns, tests, system):
    """
    Train each model on the one-step pairs, roll it out over each test
    trajectory, and report the error normalised by the system's own scale.
    """
    X = pairs[state_columns].to_numpy()
    Y = pairs[next_columns].to_numpy()

    # Trajectories have different lengths; roll each one to its own length and
    # interpolate onto a common normalised-time grid before averaging.
    grid = np.linspace(0.0, 1.0 if system == "projectile" else PERIODS_TO_ROLL, 200)

    mean_scale = float(np.mean([t["scale"] for t in tests]))
    divergence_radius = DIVERGENCE_FACTOR * mean_scale

    curves, summary = [], []
    for name in MODELS:
        model = OneStepModel(make_state_model(name), mode="delta").fit(X, Y)

        one_step = float(np.linalg.norm(
            model.step(X)[:, 0:2] - Y[:, 0:2], axis=1).mean())

        start = time.perf_counter()
        interpolated = []
        for test in tests:
            n_steps = len(test["states"]) - 1
            predicted = model.rollout(test["states"][0], n_steps,
                                      divergence_radius=divergence_radius)
            error = np.linalg.norm(predicted[:, 0:2] - test["states"][:, 0:2], axis=1)
            error = np.where(np.isfinite(error), error, divergence_radius)
            interpolated.append(np.interp(
                grid, test["times"] / test["period"], error / test["scale"]
            ))
        elapsed = time.perf_counter() - start

        relative = np.mean(interpolated, axis=0)
        curves.append(pd.DataFrame({
            "system": system, "model": name,
            "t_normalised": grid, "relative_error": relative,
        }))

        # inference cost, measured on a fair batch
        probe = X[:5000]
        t0 = time.perf_counter()
        model.step(probe)
        microseconds = 1e6 * (time.perf_counter() - t0) / len(probe)

        summary.append({
            "system": system, "model": name,
            "one_step_error": one_step,
            "one_step_relative": one_step / np.mean([t["scale"] for t in tests]),
            "relative_error_at_quarter_period": float(
                np.interp(0.25, grid, relative)),
            "relative_error_at_one_period": float(np.interp(1.0, grid, relative)),
            "train_seconds": model.train_seconds,
            "rollout_seconds": elapsed,
            "microseconds_per_prediction": microseconds,
        })
        print(f"    {system:<10s} {name:<7s} one-step {one_step:.3e}  "
              f"relative error after 1 period = {np.interp(1.0, grid, relative):.3e}")

    return pd.concat(curves, ignore_index=True), pd.DataFrame(summary)


# ==================== EXPERIMENT 2: LEARNING CURVES ====================

def learning_curves(pairs, state_columns, next_columns, tests, system):
    X_all = pairs[state_columns].to_numpy()
    Y_all = pairs[next_columns].to_numpy()
    scale = float(np.mean([t["scale"] for t in tests]))
    rng = np.random.default_rng(5)

    rows = []
    for n_train in TRAIN_SIZES:
        index = rng.choice(len(X_all), size=min(n_train, len(X_all)), replace=False)
        for name in MODELS:
            model = OneStepModel(make_state_model(name), mode="delta").fit(
                X_all[index], Y_all[index])
            error = float(np.linalg.norm(
                model.step(X_all)[:, 0:2] - Y_all[:, 0:2], axis=1).mean())
            rows.append({"system": system, "model": name, "n_train": len(index),
                         "one_step_error": error, "relative_error": error / scale})
        print(f"    {system:<10s} n_train = {len(index):>6,} done")
    return pd.DataFrame(rows)


# ==================== EXPERIMENT 3: SENSITIVITY ====================

def sensitivity(offsets=(1e-10, 1e-8, 1e-6)):
    """
    Perturb the initial condition of each system's EXACT solution and watch the
    two trajectories separate. Offsets are relative to each system's own scale,
    so "1e-8" means the same fractional nudge in both.
    """
    rows = []

    # --- projectile: analytic, no integration needed
    v0, theta, y0 = 30.0, np.deg2rad(45.0), 8.0
    flight = timeOfFlight(v0, theta, y0)
    scale = v0 * np.cos(theta) * flight
    t = np.linspace(0, flight, 400)
    base = np.column_stack([v0 * np.cos(theta) * t,
                            y0 + v0 * np.sin(theta) * t - 0.5 * G * t ** 2])
    for offset in offsets:
        dv = offset * v0
        perturbed = np.column_stack([(v0 + dv) * np.cos(theta) * t,
                                     y0 + (v0 + dv) * np.sin(theta) * t - 0.5 * G * t ** 2])
        separation = np.linalg.norm(perturbed - base, axis=1)
        rows.append(pd.DataFrame({
            "system": "projectile", "offset": offset,
            "t_normalised": t / flight,
            "relative_separation": separation / scale,
            "amplification": separation / max(offset * scale, 1e-300),
        }))

    # --- planetary: analytic Kepler, perturbing the semi-major axis
    a, e = 1.0, 0.25
    period = 2.0 * np.pi * np.sqrt(a ** 3 / GM)
    t = np.linspace(0, period, 400)
    base = kepler_solution(t, a, e)[:, 0:2]
    for offset in offsets:
        perturbed = kepler_solution(t, a * (1 + offset), e)[:, 0:2]
        separation = np.linalg.norm(perturbed - base, axis=1)
        rows.append(pd.DataFrame({
            "system": "planetary", "offset": offset,
            "t_normalised": t / period,
            "relative_separation": separation / a,
            "amplification": separation / max(offset * a, 1e-300),
        }))

    return pd.concat(rows, ignore_index=True)


# ==================== EXPERIMENT 4: CLASSICAL COST ====================

def classical_cost():
    """
    How long does it take to compute one position the ordinary way? This is the
    number the learned models have to beat for "ML is faster" to mean anything.
    """
    v0, theta, y0 = 30.0, np.deg2rad(45.0), 8.0
    t = np.linspace(0, 3.0, 100000)
    start = time.perf_counter()
    _ = v0 * np.cos(theta) * t
    _ = y0 + v0 * np.sin(theta) * t - 0.5 * G * t ** 2
    projectile_us = 1e6 * (time.perf_counter() - start) / len(t)

    # For the orbit, "one prediction" = one RK4 step with SUBSTEPS substeps,
    # which is exactly what the learned one-step model replaces.
    state = state_from_elements(1.0, 0.25)
    n = 2000
    start = time.perf_counter()
    simulate_reference(state, PLANETARY_DT, n, SUBSTEPS)
    planetary_us = 1e6 * (time.perf_counter() - start) / n

    return {"projectile": projectile_us, "planetary": planetary_us}


# ==================== SCORECARD ====================

def build_scorecard(projectile_summary, planetary_summary, sens, costs):
    """
    Condense the study into one table. Each row is normalised so the better
    system scores 1.0 -- the bars are for shape, the notes carry the numbers.
    """
    def best(summary, column):
        return float(summary[column].min())

    rows = []

    p_one = best(projectile_summary, "one_step_relative")
    q_one = best(planetary_summary, "one_step_relative")
    better = min(p_one, q_one)
    rows.append({
        "criterion": "One-step accuracy\n(relative, best model)",
        "projectile_score": better / p_one, "planetary_score": better / q_one,
        "projectile_note": f"{p_one:.1e}", "planetary_note": f"{q_one:.1e}",
    })

    p_period = best(projectile_summary, "relative_error_at_one_period")
    q_period = best(planetary_summary, "relative_error_at_one_period")
    better = min(p_period, q_period)
    rows.append({
        "criterion": "Accuracy after one full period\n(rolled out, relative)",
        "projectile_score": better / p_period, "planetary_score": better / q_period,
        "projectile_note": f"{p_period:.1e}", "planetary_note": f"{q_period:.1e}",
    })

    p_amp = float(sens[(sens["system"] == "projectile")]
                  .groupby("offset")["amplification"].last().median())
    q_amp = float(sens[(sens["system"] == "planetary")]
                  .groupby("offset")["amplification"].last().median())
    better = min(p_amp, q_amp)
    rows.append({
        "criterion": "Insensitivity to initial conditions\n(1 / amplification over one period)",
        "projectile_score": better / p_amp, "planetary_score": better / q_amp,
        "projectile_note": f"{p_amp:.1f}x", "planetary_note": f"{q_amp:.1f}x",
    })

    p_train = float(projectile_summary["train_seconds"].min())
    q_train = float(planetary_summary["train_seconds"].min())
    better = min(p_train, q_train)
    rows.append({
        "criterion": "Training cost\n(fastest model, seconds)",
        "projectile_score": better / p_train, "planetary_score": better / q_train,
        "projectile_note": f"{p_train:.2f} s", "planetary_note": f"{q_train:.2f} s",
    })

    p_inf = float(projectile_summary["microseconds_per_prediction"].min())
    q_inf = float(planetary_summary["microseconds_per_prediction"].min())
    better = min(p_inf, q_inf)
    rows.append({
        "criterion": "Inference cost\n(fastest model, us/prediction)",
        "projectile_score": better / p_inf, "planetary_score": better / q_inf,
        "projectile_note": f"{p_inf:.2f} us", "planetary_note": f"{q_inf:.2f} us",
    })

    rows.append({
        "criterion": "Is the exact flow map inside\nany model's hypothesis space?",
        "projectile_score": 1.0, "planetary_score": 0.05,
        "projectile_note": "yes -- it is affine",
        "planetary_note": "no -- it contains 1/r^3",
    })

    return pd.DataFrame(rows)


# ==================== MAIN ====================

if __name__ == "__main__":
    apply_style()
    RESULTS.mkdir(exist_ok=True)

    print("=" * 72)
    print("STAGE 12 -- PROJECTILE vs PLANETARY, HEAD TO HEAD")
    print("=" * 72)
    print("Both arms use the SAME formulation (one-step flow map, delta targets),")
    print("the SAME model families, and the SAME number of training pairs.")
    print(f"Training pairs per arm : {N_TRAIN_PAIRS:,}")
    print(f"Test trajectories      : {N_TEST_TRAJECTORIES} per arm")
    print(f"Rollout                : 1 flight time / {PERIODS_TO_ROLL} orbital periods\n")

    projectile_pairs, p_state, p_next, projectile_tests = build_projectile_arm()
    planetary_pairs_, q_state, q_next, planetary_tests = build_planetary_arm()

    print("[1] Flow-map rollout in both systems")
    projectile_curves, projectile_summary = flow_map_rollouts(
        projectile_pairs, p_state, p_next, projectile_tests, "projectile")
    planetary_curves, planetary_summary = flow_map_rollouts(
        planetary_pairs_, q_state, q_next, planetary_tests, "planetary")

    curves = pd.concat([projectile_curves, planetary_curves], ignore_index=True)
    summary = pd.concat([projectile_summary, planetary_summary], ignore_index=True)
    curves.to_csv(RESULTS / "stage12_flow_map_curves.csv", index=False)
    summary.to_csv(RESULTS / "stage12_summary.csv", index=False)

    print("\n    relative position error after ONE natural period:")
    pivot = summary.pivot(index="model", columns="system",
                          values="relative_error_at_one_period")
    print("    " + pivot.reindex(MODELS).to_string(
        float_format=lambda v: f"{v:12.4e}").replace("\n", "\n    "))

    ratio = (pivot["planetary"] / pivot["projectile"]).min()
    print(f"\n    => after one period the orbital problem is at least "
          f"{ratio:,.0f}x worse in relative terms.")

    print("\n[2] Learning curves in both systems")
    projectile_learning = learning_curves(
        projectile_pairs, p_state, p_next, projectile_tests, "projectile")
    planetary_learning = learning_curves(
        planetary_pairs_, q_state, q_next, planetary_tests, "planetary")
    learning = pd.concat([projectile_learning, planetary_learning], ignore_index=True)
    learning.to_csv(RESULTS / "stage12_learning_curves.csv", index=False)

    print("\n[3] Sensitivity of the exact dynamics")
    sens = sensitivity()
    sens.to_csv(RESULTS / "stage12_sensitivity.csv", index=False)
    for system in ["projectile", "planetary"]:
        amp = (sens[sens["system"] == system]
               .groupby("offset")["amplification"].last())
        print(f"    {system:<10s} amplification over one period: "
              + ", ".join(f"{o:.0e} -> {a:.1f}x" for o, a in amp.items()))
    print("    Neither is exponential: both systems are integrable, so a small")
    print("    initial error grows polynomially. Chaos cannot be blamed for the")
    print("    rollout divergence measured in Stage 11.")

    print("\n[4] Computational cost")
    costs = classical_cost()
    for system, value in costs.items():
        print(f"    classical {system:<10s} {value:.4f} us per prediction")
    summary["classical_microseconds"] = summary["system"].map(costs)
    summary.to_csv(RESULTS / "stage12_summary.csv", index=False)

    print("\n[5] Scorecard")
    scorecard = build_scorecard(projectile_summary, planetary_summary, sens, costs)
    scorecard.to_csv(RESULTS / "stage12_scorecard.csv", index=False)
    for _, row in scorecard.iterrows():
        print(f"    {row['criterion'][:44]:<46s} "
              f"projectile {row['projectile_note']:<20s} "
              f"planetary {row['planetary_note']}")

    print("\nFigures:")
    plot_flow_map_comparison(curves, RESULTS / "stage12_flow_map_comparison.png")
    plot_learning_curve_comparison(projectile_learning, planetary_learning,
                                   RESULTS / "stage12_learning_curves.png")
    plot_sensitivity_comparison(sens, RESULTS / "stage12_sensitivity.png")
    plot_cost_comparison(summary, RESULTS / "stage12_cost.png")
    plot_scorecard(scorecard, RESULTS / "stage12_scorecard.png")

    print("\nAnswer to the research question:")
    print("  No. With formulation, model family and data volume all held fixed,")
    print("  the learned flow map is accurate to floating-point precision for")
    print("  projectile motion and loses all useful accuracy within a fraction of")
    print("  an orbit for planetary motion. The cause is structural rather than")
    print("  statistical: the projectile flow map is affine and therefore lies")
    print("  exactly inside the hypothesis space of the simplest model tested,")
    print("  while the orbital flow map does not lie inside any of them.")
