"""Figures for Stages 9-10 -- the numerical planetary simulation."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from plots.style import INTEGRATOR_STYLE, TRUTH_STYLE, readable_log_axes, save


def plot_integrator_orbits(runs, truth, n_orbits, steps_per_orbit, e, save_path):
    """
    The picture that makes the whole of Stage 10 obvious at a glance.

    Each panel is the SAME initial condition and the SAME step size, run for
    the same number of orbits, differing only in how the step is taken.
    Explicit Euler spirals outwards -- it is silently adding energy to the
    system every step. The other three trace the exact ellipse.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    for ax, (name, states) in zip(axes.ravel(), runs.items()):
        style = INTEGRATOR_STYLE[name]
        ax.plot(truth[:, 0], truth[:, 1], label="Exact (Kepler)", **TRUTH_STYLE)
        ax.plot(states[:, 0], states[:, 1], linewidth=0.7,
                color=style["color"], label=style["label"])
        ax.plot(0, 0, marker="*", markersize=16, color="goldenrod",
                markeredgecolor="black", linestyle="none", label="Central body")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("y (AU)")

        r_final = np.hypot(states[-1, 0], states[-1, 1])
        r_start = np.hypot(states[0, 0], states[0, 1])
        ax.set_title(
            f"{style['label']}\nfinal radius / initial radius = {r_final / r_start:.3f}"
        )
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Stage 10: the same orbit integrated four ways\n"
        f"(e = {e}, {n_orbits} orbits, {steps_per_orbit} steps per orbit -- "
        "identical step size in every panel)"
    )
    fig.tight_layout()
    save(fig, save_path, "why the integrator choice matters")


def plot_conservation(panels, save_path):
    """
    Energy and angular momentum drift -- the quantitative version of the
    orbit picture, and the diagnostic an astrophysicist would actually use.

    We plot |E(t) - E(0)| / |E(0)|, the RELATIVE energy error. The true
    dynamics conserve E exactly, so every non-zero value here is numerical
    error, measurable without knowing the exact solution at all.

    Read the SHAPE, not just the height:
      a straight rising line = systematic drift (energy is being created or
        destroyed); the simulation degrades without limit.
      a flat, oscillating band = a bounded error; the simulation is wrong by
        a fixed small amount for ever, which is a completely different and
        far more usable kind of wrong.
    """
    fig, axes = plt.subplots(2, len(panels), figsize=(6.4 * len(panels), 9),
                             squeeze=False)

    for col, panel in enumerate(panels):
        ax = axes[0][col]
        for name, series in panel["energy"].items():
            style = INTEGRATOR_STYLE[name]
            ax.plot(panel["time"], np.abs(series), linewidth=1.0,
                    color=style["color"], label=style["label"])
        ax.set_yscale("log")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("|$\\Delta E$ / $E_0$|  (relative energy error)")
        ax.set_title(f"Energy conservation -- {panel['title']}")
        ax.legend(fontsize=8)

        ax = axes[1][col]
        for name, series in panel["angular_momentum"].items():
            style = INTEGRATOR_STYLE[name]
            ax.plot(panel["time"], np.abs(series), linewidth=1.0,
                    color=style["color"], label=style["label"])
        ax.set_yscale("log")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("|$\\Delta L$ / $L_0$|  (relative ang. mom. error)")
        ax.set_title(f"Angular momentum conservation -- {panel['title']}")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Stage 10: conserved quantities as a free, exact-solution-free measure "
        "of integrator quality"
    )
    fig.tight_layout()
    save(fig, save_path, "energy and angular momentum drift")


def plot_convergence(convergence, orders_quarter, orders_full, save_path):
    """
    Global position error versus step size, on log-log axes.

    A method of order p has global error ~ C dt^p, so log(error) = p log(dt) +
    const: a straight line of SLOPE p. The dotted reference lines show slopes
    1, 2 and 4. Measuring those slopes from the data is how you verify that an
    integrator was implemented correctly -- a coding error almost always shows
    up as the wrong slope.

    Left panel (error after a quarter orbit) is the honest order measurement
    and recovers 1, 1, 2, 4 exactly as theory says.

    Middle panel (error after one full orbit) is the same experiment sampled
    at a whole period, and there Euler-Cromer measures as SECOND order. That
    is not an error: symplectic Euler is conjugate to Stormer-Verlet, so its
    leading O(dt) error is a periodic coordinate shift that cancels after a
    complete orbit. Explicit Euler, meanwhile, has an error comparable to the
    orbit radius itself at coarse dt, which is why its line bends -- the
    asymptotic expansion has stopped being valid.

    Right panel replaces dt with the number of force evaluations, which is
    what actually costs time. RK4 does four times the work per step, and the
    panel shows it still wins by many orders of magnitude at equal cost.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    panels = [
        (axes[0], 0.25, orders_quarter, "Error after a quarter orbit\n(the true order of accuracy)"),
        (axes[1], 1.00, orders_full, "Error after one full orbit\n(symplectic conjugacy inflates Euler-Cromer)"),
    ]

    for ax, fraction, orders, title in panels:
        subset = convergence[convergence["fraction_of_orbit"] == fraction]
        for name in INTEGRATOR_STYLE:
            if name not in set(subset["method"]):
                continue
            rows = subset[subset["method"] == name].sort_values("dt")
            style = INTEGRATOR_STYLE[name]
            ax.plot(rows["dt"], rows["position_error"], "o-", markersize=4,
                    color=style["color"],
                    label=f"{style['label']}  (slope {orders[name]:.2f})")

        dt_ref = np.array([subset["dt"].min(), subset["dt"].max()])
        for p_order, anchor in [(1, 3e-1), (2, 3e-3), (4, 3e-7)]:
            ax.plot(dt_ref, anchor * (dt_ref / dt_ref.max()) ** p_order,
                    "k:", linewidth=1)
            ax.annotate(f"slope {p_order}",
                        xy=(dt_ref[0], anchor * (dt_ref[0] / dt_ref[1]) ** p_order),
                        fontsize=8, va="bottom")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Step size $\\Delta t$ (years)")
        ax.set_ylabel("Position error vs exact Kepler solution (AU)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    subset = convergence[convergence["fraction_of_orbit"] == 1.0]
    for name in INTEGRATOR_STYLE:
        if name not in set(subset["method"]):
            continue
        rows = subset[subset["method"] == name].sort_values("force_evals")
        style = INTEGRATOR_STYLE[name]
        ax.plot(rows["force_evals"], rows["position_error"], "o-", markersize=4,
                color=style["color"], label=style["label"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Force evaluations used (the real cost)")
    ax.set_ylabel("Position error after one orbit (AU)")
    ax.set_title("Accuracy per unit of work\n(RK4 costs 4 force evaluations per step)")
    ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Stage 10: verifying the integrators against the analytic Kepler solution"
    )
    fig.tight_layout()
    save(fig, save_path, "convergence order and cost-accuracy trade-off")


def plot_orbit_family(family, save_path):
    """
    Left : the orbit shapes we will later ask a machine-learning model to
           reproduce. All share a = 1 AU, so by Kepler's third law all share
           a period of exactly 1 year -- only the shape changes.
    Right: speed against time. Kepler's second law in disguise: the planet is
           fastest at periapsis and slowest at apoapsis, and the contrast
           grows sharply with eccentricity. This is precisely what will make
           the eccentric orbits hard to learn -- the dynamics have a narrow,
           fast feature that a uniformly-sampled model barely sees.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    ax = axes[0]
    colours = plt.cm.plasma(np.linspace(0.1, 0.8, len(family)))
    for (e, times, states), colour in zip(family, colours):
        ax.plot(states[:, 0], states[:, 1], color=colour, linewidth=1.4,
                label=f"e = {e:.2f}")
    ax.plot(0, 0, marker="*", markersize=18, color="goldenrod",
            markeredgecolor="black", linestyle="none", label="Central body")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_title("Orbit family used for the ML dataset\n(all with a = 1 AU, hence T = 1 yr)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for (e, times, states), colour in zip(family, colours):
        speed = np.linalg.norm(states[:, 2:4], axis=1)
        ax.plot(times, speed, color=colour, linewidth=1.4, label=f"e = {e:.2f}")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Speed (AU / yr)")
    ax.set_title("Speed along the orbit (Kepler's second law)\n"
                 "higher eccentricity = sharper, faster periapsis passage")
    ax.legend(fontsize=8)

    fig.suptitle("Stage 10: the physical variety in the planetary dataset")
    fig.tight_layout()
    save(fig, save_path, "orbit family and speed profiles")


def plot_kepler_third_law(table, save_path):
    """
    An end-to-end check that the simulator reproduces real celestial
    mechanics rather than merely being self-consistent.

    Kepler's third law says T^2 = (4 pi^2 / GM) a^3. We never imposed this;
    we only coded Newton's inverse-square law. If the measured periods fall
    on a line of slope 3/2 in log(T) vs log(a), the law has emerged from the
    dynamics -- which is the historically correct direction of the argument.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.plot(table["a"], table["T_measured"], "o", markersize=7,
            color="tab:blue", label="Measured from simulation")
    a_fine = np.linspace(table["a"].min(), table["a"].max(), 200)
    ax.plot(a_fine, a_fine ** 1.5, "k-", linewidth=1.6,
            label="Kepler's third law: $T = a^{3/2}$")
    ax.set_xlabel("Semi-major axis a (AU)")
    ax.set_ylabel("Orbital period T (years)")
    ax.set_title("Period vs semi-major axis")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.loglog(table["a"], table["T_measured"], "o", markersize=7, color="tab:blue")
    ax.loglog(a_fine, a_fine ** 1.5, "k-", linewidth=1.6)
    slope = np.polyfit(np.log10(table["a"]), np.log10(table["T_measured"]), 1)[0]
    ax.set_xlabel("a (AU, log scale)")
    ax.set_ylabel("T (years, log scale)")
    ax.set_title(f"Same data on log-log axes\nfitted slope = {slope:.5f}  (theory: 1.5)")
    readable_log_axes(ax)

    fig.suptitle(
        "Stage 10: Kepler's third law recovered from a simulator that was only "
        "told Newton's law of gravitation"
    )
    fig.tight_layout()
    save(fig, save_path, "Kepler's third law emerges")
