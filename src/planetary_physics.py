"""
Stage 9 & 10 -- Two-body gravitational dynamics.

THE PHYSICS
===========
Newton's law of universal gravitation says two point masses attract each
other along the line joining them with a force

    F = G m M / r^2 .

Written as a vector acting on the small body (the "planet") at position r
relative to the large body (the "star") at the origin, and divided by the
planet's own mass to get an acceleration:

    a_vec = -(G M / r^3) r_vec                                        (9.1)

The r^3 (not r^2) is not a different force law -- r_vec/r is the unit vector,
so (1/r^3) r_vec = (1/r^2) * r_hat. Writing it this way avoids computing a
square root twice.

Assumptions built into (9.1), all of which the report must state:
  * TWO-BODY: no other planets, so no perturbations.
  * RESTRICTED: the star is held fixed at the origin. Valid when m << M
    (for Sun-Earth, m/M ~ 3e-6), and it removes the centre-of-mass motion.
  * POINT MASSES: no tidal distortion, no oblateness (no J2 term).
  * NEWTONIAN: no general-relativistic precession, no finite speed of
    gravity. For Mercury this assumption is measurably wrong (43"/century);
    for an Earth-like orbit it is far below our numerical error.
  * No drag, no radiation pressure, no mass loss.

UNITS
=====
We work in ASTRONOMICAL units rather than SI:

    length = 1 AU,   time = 1 year,   mass = 1 solar mass

In these units a circular orbit of radius 1 AU takes exactly 1 year, and
Kepler's third law T^2 = (4 pi^2 / GM) a^3 forces

    GM = 4 pi^2  (AU^3 / yr^2)                                        (9.2)

This matters practically as well as aesthetically: in SI the same quantities
are ~1e11 m and ~1e30 kg, and squaring them inside an energy calculation
throws away precision for no reason. In AU-year units every quantity in the
simulation is of order 1.

WHY WE NEED NUMERICAL INTEGRATION
=================================
Unlike projectile motion, (9.1) has no closed-form solution for r(t): the
acceleration depends on the position, which is what we are solving for. The
orbit SHAPE is solvable (it is a conic section -- Kepler), but the position
AS A FUNCTION OF TIME requires solving Kepler's transcendental equation
E - e sin E = M, which has no algebraic solution. So we step the system
forward in small time increments instead. Stage 10 is about the fact that
HOW we take those steps matters enormously.
"""

import numpy as np

# Gravitational parameter of the central body, in AU^3 / yr^2 (eq. 9.2).
GM = 4.0 * np.pi ** 2

STATE_COLUMNS = ["x", "y", "vx", "vy"]


# -------------------- CORE PHYSICS --------------------

def acceleration(position, gm=GM):
    """
    Gravitational acceleration at `position` (shape (2,) or (N, 2)), eq. (9.1).
    Units: AU / yr^2.
    """
    position = np.asarray(position, dtype=float)
    r = np.linalg.norm(position, axis=-1, keepdims=True)
    return -gm * position / r ** 3


def specific_energy(state, gm=GM):
    """
    Total energy per unit planet mass:  E = v^2/2 - GM/r   (AU^2 / yr^2).

    This is the single most useful diagnostic in the whole of Stage 10. The
    true system conserves it exactly, so ANY drift in E is pure numerical
    error -- we can measure the quality of an integrator without knowing the
    exact solution. A bound orbit has E < 0; E = 0 is escape velocity.
    """
    state = np.atleast_2d(np.asarray(state, dtype=float))
    r = np.linalg.norm(state[:, 0:2], axis=1)
    v2 = np.sum(state[:, 2:4] ** 2, axis=1)
    return np.squeeze(0.5 * v2 - gm / r)


def specific_angular_momentum(state):
    """
    z-component of r x v per unit mass:  L = x*vy - y*vx   (AU^2 / yr).

    Conserved because gravity is a CENTRAL force (it points along r, so it
    exerts no torque about the origin). Equivalent to Kepler's second law:
    the radius vector sweeps equal areas in equal times, since dA/dt = L/2.
    """
    state = np.atleast_2d(np.asarray(state, dtype=float))
    return np.squeeze(state[:, 0] * state[:, 3] - state[:, 1] * state[:, 2])


def orbital_elements(state, gm=GM):
    """
    Semi-major axis, eccentricity and period from an instantaneous state.

    a comes from the vis-viva equation  v^2 = GM(2/r - 1/a);
    e from  L^2 = GM a (1 - e^2);
    T from Kepler's third law  T = 2 pi sqrt(a^3 / GM).
    """
    state = np.asarray(state, dtype=float).ravel()
    r = np.linalg.norm(state[0:2])
    v2 = np.sum(state[2:4] ** 2)
    a = 1.0 / (2.0 / r - v2 / gm)
    L = specific_angular_momentum(state)
    e = np.sqrt(max(0.0, 1.0 - L ** 2 / (gm * a)))
    T = 2.0 * np.pi * np.sqrt(a ** 3 / gm)
    return {"a": float(a), "e": float(e), "T": float(T)}


def state_from_elements(a, e, gm=GM):
    """
    Initial state for an orbit of semi-major axis `a` and eccentricity `e`,
    started at APOAPSIS on the +x axis moving in the +y direction.

    Starting at apoapsis is a deliberate choice: it is the slowest, most
    gently curving point of the orbit, so a fixed-step integrator begins in
    its most comfortable regime and any later breakdown is clearly caused by
    the fast periapsis passage rather than by a bad start.

        r_apo = a(1 + e)
        v_apo = sqrt( GM/a * (1 - e)/(1 + e) )     [from vis-viva at r_apo]
    """
    r_apo = a * (1.0 + e)
    v_apo = np.sqrt(gm / a * (1.0 - e) / (1.0 + e))
    return np.array([r_apo, 0.0, 0.0, v_apo])


# -------------------- ANALYTIC (KEPLER) SOLUTION --------------------

def _solve_kepler(mean_anomaly, e, tol=1e-14, max_iter=100):
    """
    Solve Kepler's equation  E - e sin E = M  for the eccentric anomaly E,
    by Newton-Raphson.  f(E) = E - e sin E - M,  f'(E) = 1 - e cos E.

    Converges quadratically; for e < 0.9 it needs ~5 iterations from the
    starting guess E = M. This is the step that has no algebraic solution
    and is the formal reason orbital positions require numerics at all.
    """
    M = np.asarray(mean_anomaly, dtype=float)
    E = M.copy()
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        if np.max(np.abs(f)) < tol:
            break
        E = E - f / (1.0 - e * np.cos(E))
    return E


def kepler_solution(times, a, e, gm=GM):
    """
    EXACT states at the given times for the orbit produced by
    state_from_elements(a, e). Returns an (N, 4) array [x, y, vx, vy].

    This is the ground truth against which every integrator in Stage 10 is
    judged. Having an analytic reference is a real methodological advantage:
    it means the reported integrator errors are absolute, not "error relative
    to a finer run of the same possibly-biased method".

    Frame bookkeeping: the standard perifocal frame puts PERIAPSIS on the +x'
    axis, but state_from_elements starts at APOAPSIS on +x. Those differ by a
    rotation of pi, i.e. (x, y) = (-x', -y'). At t = 0 we are at apoapsis, so
    E = pi and therefore M0 = E - e sin E = pi.
    """
    times = np.asarray(times, dtype=float)
    T = 2.0 * np.pi * np.sqrt(a ** 3 / gm)
    n = 2.0 * np.pi / T                      # mean motion (rad / yr)

    M = np.pi + n * times                    # mean anomaly, starting at apoapsis
    E = _solve_kepler(M, e)

    b = a * np.sqrt(1.0 - e ** 2)            # semi-minor axis
    xp = a * (np.cos(E) - e)
    yp = b * np.sin(E)

    # dE/dt from differentiating Kepler's equation: (1 - e cos E) dE/dt = n
    Edot = n / (1.0 - e * np.cos(E))
    vxp = -a * np.sin(E) * Edot
    vyp = b * np.cos(E) * Edot

    return np.column_stack([-xp, -yp, -vxp, -vyp])


# -------------------- INTEGRATORS (STAGE 10) --------------------
# Every integrator has the same signature so the comparison is apples to
# apples: (state, dt, gm) -> new state.

def step_euler(state, dt, gm=GM):
    """
    Explicit (forward) Euler -- the definition of a derivative, rearranged:

        r_{n+1} = r_n + v_n dt
        v_{n+1} = v_n + a(r_n) dt

    Both updates use quantities from the START of the step. First-order
    accurate (local error ~dt^2, global ~dt). Its fatal flaw for orbits is
    not the accuracy but the BIAS: for oscillatory motion it systematically
    adds energy every step, so the planet spirals outwards for ever. Halving
    dt halves the rate of the spiral but never removes it.
    """
    x, y, vx, vy = state
    ax, ay = acceleration([x, y], gm)
    return np.array([x + vx * dt, y + vy * dt, vx + ax * dt, vy + ay * dt])


def step_euler_cromer(state, dt, gm=GM):
    """
    Euler-Cromer (semi-implicit Euler) -- one character different from Euler:

        v_{n+1} = v_n + a(r_n) dt
        r_{n+1} = r_n + v_{n+1} dt          <-- the NEW velocity

    Still first-order accurate, but it is SYMPLECTIC: it exactly conserves a
    slightly-wrong energy function, so the true energy oscillates around the
    correct value instead of drifting away from it. The orbit stays closed
    for ever even though each individual position is only first-order
    accurate. This is the cheapest possible demonstration that, for long-term
    dynamics, the STRUCTURE of an integrator matters more than its order.
    """
    x, y, vx, vy = state
    ax, ay = acceleration([x, y], gm)
    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    return np.array([x + vx_new * dt, y + vy_new * dt, vx_new, vy_new])


def step_verlet(state, dt, gm=GM):
    """
    Velocity Verlet -- second order, symplectic, one force evaluation per step
    if the previous acceleration is cached (we recompute it here for clarity).

        r_{n+1} = r_n + v_n dt + 1/2 a_n dt^2
        v_{n+1} = v_n + 1/2 (a_n + a_{n+1}) dt

    It is time-reversible, which is exactly the symmetry the true equations
    have, and that is why it does not drift. The workhorse of molecular
    dynamics and N-body astrophysics for precisely this reason.
    """
    x, y, vx, vy = state
    ax, ay = acceleration([x, y], gm)

    x_new = x + vx * dt + 0.5 * ax * dt ** 2
    y_new = y + vy * dt + 0.5 * ay * dt ** 2

    ax_new, ay_new = acceleration([x_new, y_new], gm)
    vx_new = vx + 0.5 * (ax + ax_new) * dt
    vy_new = vy + 0.5 * (ay + ay_new) * dt

    return np.array([x_new, y_new, vx_new, vy_new])


def _derivative(state, gm=GM):
    """d/dt [x, y, vx, vy] = [vx, vy, ax, ay] -- the system in first-order form."""
    x, y, vx, vy = state
    ax, ay = acceleration([x, y], gm)
    return np.array([vx, vy, ax, ay])


def step_rk4(state, dt, gm=GM):
    """
    Classical Runge-Kutta 4 -- fourth order, four force evaluations per step.

        k1 = f(s)              k2 = f(s + dt/2 k1)
        k3 = f(s + dt/2 k2)    k4 = f(s + dt k3)
        s_{n+1} = s + dt/6 (k1 + 2k2 + 2k3 + k4)

    Probes the acceleration field at four points inside the step and takes a
    weighted average, cancelling error terms up to dt^4. Per-step accuracy is
    far better than Verlet -- but RK4 is NOT symplectic, so over very many
    orbits it slowly loses energy. It is the right choice for high accuracy
    over a moderate number of orbits, which is exactly our use case for
    generating training data.
    """
    k1 = _derivative(state, gm)
    k2 = _derivative(state + 0.5 * dt * k1, gm)
    k3 = _derivative(state + 0.5 * dt * k2, gm)
    k4 = _derivative(state + dt * k3, gm)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


INTEGRATORS = {
    "euler": step_euler,
    "euler_cromer": step_euler_cromer,
    "verlet": step_verlet,
    "rk4": step_rk4,
}

INTEGRATOR_ORDER = {"euler": 1, "euler_cromer": 1, "verlet": 2, "rk4": 4}
INTEGRATOR_FORCE_EVALS = {"euler": 1, "euler_cromer": 1, "verlet": 2, "rk4": 4}


def simulate(state0, dt, n_steps, method="rk4", gm=GM):
    """
    Advance `state0` for `n_steps` steps of size `dt`.

    Returns (times, states) where states has shape (n_steps + 1, 4) and row 0
    is the initial condition, so times[i] corresponds to states[i].
    """
    step = INTEGRATORS[method]
    states = np.empty((n_steps + 1, 4), dtype=float)
    states[0] = np.asarray(state0, dtype=float)

    for i in range(n_steps):
        states[i + 1] = step(states[i], dt, gm)

    return np.arange(n_steps + 1) * dt, states


def simulate_orbits(a=1.0, e=0.0, n_orbits=10, steps_per_orbit=500,
                    method="rk4", gm=GM):
    """
    Convenience wrapper that expresses the run in ORBITS rather than steps,
    so that runs with different semi-major axes remain comparable.
    """
    state0 = state_from_elements(a, e, gm)
    T = 2.0 * np.pi * np.sqrt(a ** 3 / gm)
    dt = T / steps_per_orbit
    return simulate(state0, dt, int(n_orbits * steps_per_orbit), method, gm)


def simulate_reference(state0, dt_store, n_store, substeps=10, method="rk4", gm=GM):
    """
    Produce a trajectory sampled every `dt_store` years, but integrated
    internally with `substeps` RK4 steps per stored point.

    This separates two things that are easy to confuse:

      dt_store : the time interval the MACHINE-LEARNING model will learn to
                 step over. It has to be reasonably large, or a rollout of a
                 few orbits would need an absurd number of model calls.
      dt_store/substeps : the interval the NUMERICAL method actually uses.
                 It has to be small, or the "ground truth" the model is
                 trained against would carry its own visible error.

    Using one dt for both would force a bad compromise. Here the stored
    trajectory is accurate to ~1e-11 AU per orbit while still being sampled
    coarsely enough for a practical rollout.
    """
    step = INTEGRATORS[method]
    dt_inner = dt_store / substeps

    states = np.empty((n_store + 1, 4), dtype=float)
    states[0] = np.asarray(state0, dtype=float)

    current = states[0].copy()
    for i in range(n_store):
        for _ in range(substeps):
            current = step(current, dt_inner, gm)
        states[i + 1] = current

    return np.arange(n_store + 1) * dt_store, states
