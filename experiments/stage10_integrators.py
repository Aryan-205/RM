"""
Stage 10 -- Numerical Planetary Simulation

Research sub-questions answered here:
    Does the choice of numerical integrator materially change the physics we
    obtain? How accurate is the baseline against which the ML model in Stage
    11 will be judged?

Why this stage is not optional
------------------------------
In Part I the "ground truth" was an algebraic formula: exact, instantaneous,
beyond argument. In Part II there is no such formula for position as a
function of time, so the ground truth is itself produced by a numerical
method -- and a numerical method can be wrong. Before we are allowed to say
"the ML model has an error of X AU", we must establish that the reference
has an error far smaller than X. That is what this stage does, and it does it
against the analytic Kepler solution rather than against another simulation.

Experiments
-----------
  1. Four integrators, one orbit configuration, identical step size, 40
     orbits. Qualitative: does the orbit stay closed?
  2. Energy and angular momentum drift over time, for a circular and an
     eccentric orbit.
  3. Convergence study: position error after one orbit vs step size, giving
     the measured order of accuracy of each method. Also error vs number of
     force evaluations, which is the honest cost comparison.
  4. Kepler's third law recovered from the simulator.
  5. The orbit family that becomes the Stage 11 training data.

  Independent variables : integrator, step size, eccentricity, semi-major axis
  Dependent variables   : position error vs Kepler, |dE/E|, |dL/L|, runtime
  Controlled            : GM, initial conditions for a given (a, e), the
                          number of orbits simulated, the machine

Run:  python experiments/stage10_integrators.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from plots.planetary import (
    plot_conservation,
    plot_convergence,
    plot_integrator_orbits,
    plot_kepler_third_law,
    plot_orbit_family,
)
from plots.style import apply_style
from src.planetary_physics import (
    GM,
    INTEGRATOR_FORCE_EVALS,
    INTEGRATORS,
    kepler_solution,
    orbital_elements,
    simulate,
    simulate_orbits,
    specific_angular_momentum,
    specific_energy,
    state_from_elements,
)

RESULTS = ROOT / "results"
METHODS = ["euler", "euler_cromer", "verlet", "rk4"]

# Deliberately coarse, so the differences between methods are visible at all.
DEMO_STEPS_PER_ORBIT = 200
DEMO_ORBITS = 40
DEMO_E = 0.3

# Long enough to distinguish "bounded oscillation" from "slow linear drift".
CONSERVATION_ORBITS = 200
CONSERVATION_STEPS_PER_ORBIT = 400

CONVERGENCE_STEPS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
# Sample the error a quarter of the way round (the honest order measurement)
# and after exactly one full period (where symplectic conjugacy shows up).
CONVERGENCE_FRACTIONS = [0.25, 1.0]

# The orbit family used for Stage 11's ML dataset.
FAMILY_E = [0.0, 0.2, 0.4, 0.6]


# -------------------- EXPERIMENT 1: ORBIT SHAPES --------------------

def integrator_orbits():
    runs = {}
    for method in METHODS:
        _, states = simulate_orbits(
            a=1.0, e=DEMO_E, n_orbits=DEMO_ORBITS,
            steps_per_orbit=DEMO_STEPS_PER_ORBIT, method=method,
        )
        runs[method] = states

    # One exact orbit, densely sampled, as the black reference ellipse.
    t_fine = np.linspace(0, 1.0, 2000)
    truth = kepler_solution(t_fine, 1.0, DEMO_E)
    return runs, truth


# -------------------- EXPERIMENT 2: CONSERVED QUANTITIES --------------------

def conservation_panel(e, title):
    """Relative drift of E and L over many orbits, for every integrator."""
    energy, angular = {}, {}
    times = None

    for method in METHODS:
        t, states = simulate_orbits(
            a=1.0, e=e, n_orbits=CONSERVATION_ORBITS,
            steps_per_orbit=CONSERVATION_STEPS_PER_ORBIT, method=method,
        )
        E = specific_energy(states)
        L = specific_angular_momentum(states)
        # Relative error, so the numbers are comparable between orbits with
        # different absolute energies.
        energy[method] = (E - E[0]) / np.abs(E[0])
        angular[method] = (L - L[0]) / np.abs(L[0])
        times = t

    return {"title": title, "time": times, "energy": energy,
            "angular_momentum": angular}


def conservation_summary(panels):
    rows = []
    for panel in panels:
        for method in METHODS:
            rows.append({
                "orbit": panel["title"],
                "method": method,
                "max_rel_energy_error": float(np.abs(panel["energy"][method]).max()),
                "final_rel_energy_error": float(np.abs(panel["energy"][method][-1])),
                "max_rel_angmom_error": float(np.abs(panel["angular_momentum"][method]).max()),
            })
    return pd.DataFrame(rows)


# -------------------- EXPERIMENT 3: CONVERGENCE --------------------

def convergence_study(a=1.0, e=0.3):
    """
    Position error as a function of step size, measured against the analytic
    Kepler solution, at TWO sampling times.

    Why two? Because the answer depends on when you look, and that turns out
    to be a real result rather than an inconvenience:

      fraction = 0.25 -- a quarter of an orbit, an arbitrary instant. This is
        the honest measurement of the global order of accuracy, and it is what
        the phrase "first order" or "fourth order" refers to.

      fraction = 1.00 -- after exactly one full period. Here Euler-Cromer
        measures as SECOND order even though it is a first-order method. The
        explanation is a known result in geometric numerical integration: the
        symplectic Euler method is *conjugate* to Stormer-Verlet, meaning the
        two produce the same trajectory up to a fixed O(dt) change of
        coordinates. That coordinate shift is periodic with the orbit, so it
        cancels out when you sample at whole periods, exposing the underlying
        second-order behaviour. It is a concrete demonstration that a
        symplectic method's long-term behaviour can be better than its formal
        order suggests.
    """
    T = 2.0 * np.pi * np.sqrt(a ** 3 / GM)
    state0 = state_from_elements(a, e)

    rows = []
    for fraction in CONVERGENCE_FRACTIONS:
        for n_per_orbit in CONVERGENCE_STEPS:
            dt = T / n_per_orbit
            n_steps = int(round(n_per_orbit * fraction))
            exact_final = kepler_solution(np.array([n_steps * dt]), a, e)[0]

            for method in METHODS:
                start = time.perf_counter()
                _, states = simulate(state0, dt, n_steps, method)
                runtime = time.perf_counter() - start

                rows.append({
                    "method": method,
                    "fraction_of_orbit": fraction,
                    "steps_per_orbit": n_per_orbit,
                    "dt": dt,
                    "n_steps": n_steps,
                    "position_error": float(
                        np.linalg.norm(states[-1, 0:2] - exact_final[0:2])
                    ),
                    "force_evals": n_steps * INTEGRATOR_FORCE_EVALS[method],
                    "runtime_s": runtime,
                    "runtime_per_step_us": 1e6 * runtime / n_steps,
                })
    return pd.DataFrame(rows)


# Only errors inside this band are used to fit the convergence order.
# Above it, the error is comparable to the orbit itself and the asymptotic
# expansion error ~ C dt^p no longer holds (explicit Euler at coarse dt).
# Below it, double-precision round-off dominates and the curve flattens.
ORDER_FIT_BAND = (1e-12, 1e-1)


def measured_orders(convergence, fraction):
    """Fit log(error) = p log(dt) + c inside the asymptotic band."""
    subset = convergence[convergence["fraction_of_orbit"] == fraction]
    orders = {}
    for method in METHODS:
        rows = subset[subset["method"] == method].sort_values("dt")
        usable = rows[
            (rows["position_error"] > ORDER_FIT_BAND[0])
            & (rows["position_error"] < ORDER_FIT_BAND[1])
        ]
        if len(usable) < 3:
            usable = rows.tail(4)
        slope = np.polyfit(
            np.log10(usable["dt"]), np.log10(usable["position_error"]), 1
        )[0]
        orders[method] = float(slope)
    return orders


# -------------------- EXPERIMENT 4: KEPLER'S THIRD LAW --------------------

def kepler_third_law(semi_major_axes=(0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0)):
    """
    Measure each orbit's period by detecting when the planet crosses back
    through its starting angle, using linear interpolation between the two
    bracketing steps so the answer is not limited to a multiple of dt.
    """
    rows = []
    for a in semi_major_axes:
        state0 = state_from_elements(a, 0.0)
        T_theory = 2.0 * np.pi * np.sqrt(a ** 3 / GM)
        dt = T_theory / 4000
        _, states = simulate(state0, dt, 4200, "rk4")

        # The orbit starts on the +x axis with y = 0 moving to +y. One full
        # period later, y returns to 0 from below, i.e. the first index > 10
        # where y changes from negative to non-negative.
        y = states[:, 1]
        crossings = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
        i = crossings[crossings > 10][0]
        # linear interpolation for the exact zero crossing
        frac = -y[i] / (y[i + 1] - y[i])
        T_measured = (i + frac) * dt

        rows.append({
            "a": a, "T_theory": T_theory, "T_measured": T_measured,
            "rel_error": abs(T_measured - T_theory) / T_theory,
        })
    return pd.DataFrame(rows)


# -------------------- EXPERIMENT 5: THE ORBIT FAMILY --------------------

def orbit_family():
    family = []
    for e in FAMILY_E:
        t, states = simulate_orbits(a=1.0, e=e, n_orbits=1,
                                    steps_per_orbit=2000, method="rk4")
        family.append((e, t, states))
    return family


# -------------------- MAIN --------------------

if __name__ == "__main__":
    apply_style()
    RESULTS.mkdir(exist_ok=True)

    print("=" * 72)
    print("STAGE 10 -- NUMERICAL PLANETARY SIMULATION")
    print("=" * 72)
    print(f"GM = {GM:.6f} AU^3/yr^2   (units: AU, year, solar mass)")
    print(f"Reference solution: analytic Kepler (Newton-solved E - e sinE = M)\n")

    # ---- 1. orbit shapes
    print(f"[1] Integrating {DEMO_ORBITS} orbits at "
          f"{DEMO_STEPS_PER_ORBIT} steps/orbit, e = {DEMO_E}")
    runs, truth = integrator_orbits()
    for method, states in runs.items():
        r0 = np.hypot(*states[0, 0:2])
        r1 = np.hypot(*states[-1, 0:2])
        el = orbital_elements(states[-1])
        print(f"    {method:<13s} final/initial radius = {r1 / r0:7.4f}   "
              f"final a = {el['a']:7.4f} AU   final e = {el['e']:.4f}")

    # ---- 2. conservation
    print(f"\n[2] Conservation over {CONSERVATION_ORBITS} orbits "
          f"at {CONSERVATION_STEPS_PER_ORBIT} steps/orbit")
    panels = [
        conservation_panel(0.0, "circular orbit (e = 0)"),
        conservation_panel(0.6, "eccentric orbit (e = 0.6)"),
    ]
    cons = conservation_summary(panels)
    cons.to_csv(RESULTS / "stage10_conservation.csv", index=False)
    for orbit in cons["orbit"].unique():
        print(f"    {orbit}")
        for _, r in cons[cons["orbit"] == orbit].iterrows():
            print(f"      {r['method']:<13s} max|dE/E| = {r['max_rel_energy_error']:.3e}   "
                  f"final|dE/E| = {r['final_rel_energy_error']:.3e}   "
                  f"max|dL/L| = {r['max_rel_angmom_error']:.3e}")

    # ---- 3. convergence
    print("\n[3] Convergence study (position error vs step size, against Kepler)")
    convergence = convergence_study()
    convergence.to_csv(RESULTS / "stage10_convergence.csv", index=False)

    orders_quarter = measured_orders(convergence, 0.25)
    orders_full = measured_orders(convergence, 1.0)

    from src.planetary_physics import INTEGRATOR_ORDER
    print("    measured order of accuracy:")
    print(f"      {'method':<14s} {'theory':>7s} {'at 0.25 T':>11s} {'at 1.00 T':>11s}")
    for method in METHODS:
        print(f"      {method:<14s} {INTEGRATOR_ORDER[method]:>7d} "
              f"{orders_quarter[method]:>11.3f} {orders_full[method]:>11.3f}")
    print("    Note: Euler-Cromer measures as 2nd order at whole periods because")
    print("    symplectic Euler is conjugate to Verlet -- its O(dt) error is a")
    print("    periodic coordinate shift that cancels after a full orbit.")

    for fraction in CONVERGENCE_FRACTIONS:
        subset = convergence[convergence["fraction_of_orbit"] == fraction]
        pivot = subset.pivot(index="steps_per_orbit", columns="method",
                             values="position_error")
        print(f"\n    position error after {fraction:.2f} orbit (AU):")
        print(pivot[METHODS].to_string(float_format=lambda v: f"{v:11.3e}"))

    cost = convergence.groupby("method")["runtime_per_step_us"].mean()
    print("\n    mean wall-clock cost per step (microseconds, pure Python):")
    for method in METHODS:
        print(f"      {method:<13s} {cost[method]:6.2f} us "
              f"({INTEGRATOR_FORCE_EVALS[method]} force evaluations)")

    # ---- 4. Kepler's third law
    print("\n[4] Kepler's third law from the simulator")
    k3 = kepler_third_law()
    k3.to_csv(RESULTS / "stage10_kepler_third_law.csv", index=False)
    print(k3.to_string(index=False, float_format=lambda v: f"{v:12.6g}"))
    slope = np.polyfit(np.log10(k3["a"]), np.log10(k3["T_measured"]), 1)[0]
    print(f"    fitted log-log slope = {slope:.6f}   (Kepler: 1.5)")

    # ---- 5. the orbit family
    print("\n[5] Orbit family for Stage 11")
    family = orbit_family()
    for e, t, states in family:
        speed = np.linalg.norm(states[:, 2:4], axis=1)
        el = orbital_elements(states[0])
        print(f"    e = {e:.2f}:  r in [{np.hypot(states[:,0], states[:,1]).min():.3f}, "
              f"{np.hypot(states[:,0], states[:,1]).max():.3f}] AU,  "
              f"speed in [{speed.min():.3f}, {speed.max():.3f}] AU/yr,  "
              f"T = {el['T']:.4f} yr")

    # ---- figures
    print("\nFigures:")
    plot_integrator_orbits(runs, truth, DEMO_ORBITS, DEMO_STEPS_PER_ORBIT, DEMO_E,
                           RESULTS / "stage10_integrator_orbits.png")
    plot_conservation(panels, RESULTS / "stage10_conservation.png")
    plot_convergence(convergence, orders_quarter, orders_full,
                     RESULTS / "stage10_convergence.png")
    plot_kepler_third_law(k3, RESULTS / "stage10_kepler_third_law.png")
    plot_orbit_family(family, RESULTS / "stage10_orbit_family.png")

    print("\nConclusion for the methodology section:")
    best = convergence[(convergence["method"] == "rk4")
                       & (convergence["steps_per_orbit"] == 3200)
                       & (convergence["fraction_of_orbit"] == 1.0)]
    print(f"    RK4 at 3,200 steps/orbit has a one-orbit position error of "
          f"{best['position_error'].iloc[0]:.2e} AU.")
    print("    Stage 11's ML errors will be many orders of magnitude larger than")
    print("    this, so the numerical baseline can be treated as exact truth.")
