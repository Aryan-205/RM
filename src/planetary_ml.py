"""
Stage 11 -- Machine-learning formulations for orbital motion.

THE THREE WAYS TO POSE THE PROBLEM
==================================
For projectile motion there was only one sensible formulation, because a
closed-form solution exists: give the model the initial conditions and a time,
ask for a position. Orbital motion has no such formula, so there is a genuine
modelling choice to make, and the choice matters more than the model.

Approach A -- LEARNED FLOW MAP:      s_t  ->  s_{t+dt}
    The model replaces the integrator. Give it a state, it returns the state
    one timestep later; feed that back in to go further. This is the general
    formulation: it needs no knowledge of orbital elements and would work for
    a system with no analytic solution at all.
    Its weakness is structural. Every prediction becomes the input to the
    next, so errors do not merely persist, they COMPOUND. If each step has a
    relative error eps and the dynamics amplify nearby trajectories by a
    factor lambda per step, the error after n steps grows roughly like
    eps * (lambda^n - 1)/(lambda - 1). This is the same mechanism that limits
    weather forecasting, and it is the central phenomenon of Part II.

Approach B -- LEARNED CLOSED FORM:   (a, e, t)  ->  (x, y)
    The model plays the role of the Kepler solution. There is no feedback
    loop, so there is no error accumulation at all: the error at t = 100 yr
    is no worse than at t = 1 yr.
    Its weakness is that it is barely dynamics. It requires the orbit to be
    labelled by parameters that already summarise the whole solution, and
    those parameters exist only because this system happens to be integrable.
    For a chaotic or non-integrable system there is nothing to put in place
    of (a, e), so the approach does not generalise.

Approach C -- LEARNED MULTISTEP MAP: [s_{t-k+1} ... s_t]  ->  s_{t+dt}
    A window of past states instead of one. The extra states carry finite-
    difference information about velocity and acceleration, so in principle
    the model has more to work with. It is still autoregressive, so it still
    accumulates error; the question is whether the richer input reduces the
    per-step error enough to matter.

ABSOLUTE vs DELTA TARGETS
=========================
For A and C there is a second, smaller choice that turns out to matter a lot.
The model can predict the next state directly (absolute), or the CHANGE

    Delta_s = s_{t+dt} - s_t                                         (11.1)

and we reconstruct s_{t+dt} = s_t + Delta_s (delta form).

Delta form wins for a concrete, checkable reason. Over dt = 0.004 yr a planet
moves about 0.025 AU while sitting at |r| ~ 1 AU, so the change is ~40x
smaller than the state. In absolute form the model must produce ~1.0 to a
precision of 1e-4 -- it spends all its capacity reproducing the identity map
and only the last digits carry the physics. In delta form the identity is
built into the arithmetic for free and every bit of model capacity is spent
on the part that is actually dynamics. This is standard practice in learned
simulators and is worth stating explicitly in the report as a design decision
with a physical justification.
"""

import time

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.planetary_physics import (
    GM,
    specific_angular_momentum,
    specific_energy,
)

STATE_COLUMNS = ["x", "y", "vx", "vy"]
NEXT_COLUMNS = [f"next_{c}" for c in STATE_COLUMNS]

RANDOM_STATE = 42

# A rollout that passes this radius has left the two-body system entirely
# (the orbits studied here have apoapsis <= 1.7 AU). Beyond it the state is
# recorded as NaN: the trajectory is not "very wrong", it no longer exists.
DIVERGENCE_RADIUS = 100.0


# -------------------- MODEL BUILDERS --------------------

def make_state_model(name):
    """
    Regressors configured for state-to-state prediction.

    Two differences from the projectile versions in src/models.py:
      * the MLP is larger and trained longer, because the orbital map is
        harder (it contains 1/r^3, which is very steep near periapsis);
      * targets are standardised as well as inputs, via
        TransformedTargetRegressor. Delta targets have magnitudes around 1e-2
        to 1e-1, and an unscaled squared-error loss on numbers that small
        gives the optimiser almost nothing to descend.
    """
    if name == "linear":
        return LinearRegression()

    if name == "poly2":
        return make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1e-6),
        )

    if name == "forest":
        return RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        )

    if name == "mlp":
        network = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 128, 128),
                activation="tanh",
                solver="adam",
                learning_rate_init=1e-3,
                max_iter=3000,
                tol=1e-9,
                n_iter_no_change=50,
                random_state=RANDOM_STATE,
            ),
        )
        return TransformedTargetRegressor(
            regressor=network, transformer=StandardScaler()
        )

    raise ValueError(f"unknown model name: {name!r}")


# -------------------- APPROACH A: LEARNED FLOW MAP --------------------

class OneStepModel:
    """
    A learned replacement for one step of a numerical integrator.

    mode='delta'    : the wrapped regressor predicts s_{t+dt} - s_t   (11.1)
    mode='absolute' : it predicts s_{t+dt} directly
    """

    def __init__(self, base, mode="delta"):
        self.base = base
        self.mode = mode
        self.train_seconds = None

    def fit(self, states, next_states):
        states = np.asarray(states, dtype=float)
        next_states = np.asarray(next_states, dtype=float)
        target = next_states - states if self.mode == "delta" else next_states

        start = time.perf_counter()
        self.base.fit(states, target)
        self.train_seconds = time.perf_counter() - start
        return self

    def step(self, states):
        """Advance a batch of states by one dt."""
        states = np.atleast_2d(np.asarray(states, dtype=float))
        prediction = self.base.predict(states)
        return states + prediction if self.mode == "delta" else prediction

    def rollout(self, state0, n_steps, divergence_radius=DIVERGENCE_RADIUS):
        """
        Apply the learned step repeatedly, feeding each output back in.

        This is the operation that separates a model that fits one-step data
        from a model that can actually simulate. Nothing here corrects the
        trajectory; whatever error the model makes at step k is carried into
        step k+1 as if it were fact.

        `state0` may be a single state (4,) or a batch of initial states
        (N, 4). Batching is not a micro-optimisation here: a Random Forest
        call costs milliseconds of Python overhead almost regardless of how
        many rows it is given, so rolling 20 test orbits out together costs
        barely more than rolling out one, and it is what makes averaging over
        many initial conditions affordable.

        DIVERGENCE. Some learned maps are not merely inaccurate, they are
        unstable: the rollout runs away to infinity and then to NaN, which
        would crash the model on the next call. That is a genuine result and
        must be recorded rather than avoided, so once a trajectory passes
        `divergence_radius` (100 AU, sixty times the largest orbit here) we
        mark every later point NaN and keep the model's input finite. A NaN in
        the output therefore means "this rollout left the system", not "the
        code failed".

        Returns (n_steps + 1, 4) for a single state, or
                (n_steps + 1, N, 4) for a batch.
        """
        state0 = np.asarray(state0, dtype=float)
        single = state0.ndim == 1
        current = np.atleast_2d(state0).copy()

        trajectory = np.empty((n_steps + 1, current.shape[0], 4), dtype=float)
        trajectory[0] = current
        alive = np.ones(current.shape[0], dtype=bool)
        anchor = current.copy()          # finite placeholder for dead rows

        for i in range(n_steps):
            nxt = self.step(current)

            escaped = (
                ~np.isfinite(nxt).all(axis=1)
                | (np.hypot(nxt[:, 0], nxt[:, 1]) > divergence_radius)
            )
            alive &= ~escaped

            # keep the model's input finite so the next call cannot crash
            nxt[~alive] = anchor[~alive]
            current = nxt

            trajectory[i + 1] = np.where(alive[:, None], nxt, np.nan)

            if not alive.any():
                trajectory[i + 2:] = np.nan
                break

        return trajectory[:, 0, :] if single else trajectory


# -------------------- APPROACH C: LEARNED MULTISTEP MAP --------------------

class HistoryModel:
    """
    Same idea as OneStepModel, but the input is the last `history` states
    flattened into one vector (oldest first, most recent last).

    Rollout needs a warm-up: the first `history` states must come from the
    true trajectory, because the model cannot produce a prediction until it
    has a full window. Those warm-up states are therefore NOT counted as
    predictions when the error is measured.
    """

    def __init__(self, base, history=3, mode="delta"):
        self.base = base
        self.history = history
        self.mode = mode
        self.train_seconds = None

    def fit(self, windows, next_states):
        windows = np.asarray(windows, dtype=float)
        next_states = np.asarray(next_states, dtype=float)
        # delta is measured from the MOST RECENT state in the window, which
        # occupies the last four columns
        latest = windows[:, -4:]
        target = next_states - latest if self.mode == "delta" else next_states

        start = time.perf_counter()
        self.base.fit(windows, target)
        self.train_seconds = time.perf_counter() - start
        return self

    def step(self, windows):
        windows = np.atleast_2d(np.asarray(windows, dtype=float))
        prediction = self.base.predict(windows)
        return windows[:, -4:] + prediction if self.mode == "delta" else prediction

    def rollout(self, warmup_states, n_steps, divergence_radius=DIVERGENCE_RADIUS):
        """
        `warmup_states` must have exactly `history` rows of true states.
        Divergence is handled as in OneStepModel.rollout: once the trajectory
        passes `divergence_radius` the remaining points are NaN.
        """
        warmup = np.asarray(warmup_states, dtype=float)
        assert warmup.shape[0] == self.history

        trajectory = np.empty((self.history + n_steps, 4), dtype=float)
        trajectory[: self.history] = warmup
        working = trajectory.copy()      # finite copy used to build the windows

        for i in range(n_steps):
            window = working[i: i + self.history].reshape(1, -1)
            nxt = self.step(window)[0]

            if not np.isfinite(nxt).all() or np.hypot(nxt[0], nxt[1]) > divergence_radius:
                trajectory[self.history + i:] = np.nan
                break

            working[self.history + i] = nxt
            trajectory[self.history + i] = nxt

        return trajectory


# -------------------- APPROACH B: LEARNED CLOSED FORM --------------------

class DirectModel:
    """
    (a, e, t) -> (x, y, vx, vy). No feedback, so no accumulation.

    Note that `t` is passed through unchanged rather than wrapped into a
    phase. That is deliberate: it means asking for t = 20 yr when training
    only saw t <= 3 yr is a genuine extrapolation, and the model's failure
    there is one of the clearer results in Stage 11.
    """

    def __init__(self, base):
        self.base = base
        self.train_seconds = None

    def fit(self, features, targets):
        start = time.perf_counter()
        self.base.fit(np.asarray(features, dtype=float),
                      np.asarray(targets, dtype=float))
        self.train_seconds = time.perf_counter() - start
        return self

    def predict(self, features):
        return self.base.predict(np.asarray(features, dtype=float))


# -------------------- ROLLOUT DIAGNOSTICS --------------------

def rollout_diagnostics(times, predicted, truth):
    """
    Per-timestep comparison of a predicted trajectory against the truth.

    Four quantities, each answering a different question:
      position_error : how far from the right place (AU)? the headline number
      velocity_error : how wrong is the motion (AU/yr)? leads the position error
      energy_error   : |dE/E| of the PREDICTED trajectory. A numerical
                       integrator conserves this to ~1e-11; a learned map has
                       no reason to conserve anything, and watching this
                       quantity wander is the clearest evidence that the model
                       did not learn the physics, only a good local fit to it.
      radius         : |r| over time, which shows whether a failing rollout
                       spirals in, spirals out, or leaves the system entirely
    """
    predicted = np.asarray(predicted, dtype=float)
    truth = np.asarray(truth, dtype=float)

    energy = specific_energy(predicted)
    energy_true = specific_energy(truth)
    ang_mom = specific_angular_momentum(predicted)

    return pd.DataFrame({
        "t": times,
        "position_error": np.linalg.norm(predicted[:, 0:2] - truth[:, 0:2], axis=1),
        "velocity_error": np.linalg.norm(predicted[:, 2:4] - truth[:, 2:4], axis=1),
        "energy": energy,
        "energy_rel_error": np.abs(energy - energy_true[0]) / np.abs(energy_true[0]),
        "ang_mom_rel_error": np.abs(ang_mom - ang_mom[0]) / np.abs(ang_mom[0]),
        "radius": np.linalg.norm(predicted[:, 0:2], axis=1),
        "radius_true": np.linalg.norm(truth[:, 0:2], axis=1),
    })


def error_doubling_time(times, errors, floor=1e-8):
    """
    Fit log(error) = alpha * t + c over the growth phase and report the time
    for the error to grow by a factor of e, and by a factor of 10.

    If the growth really is exponential this is a meaningful number and is the
    practical answer to "how far ahead can this model be trusted?". If the
    growth is only linear the fit still returns something, so the report must
    always quote the fit quality alongside it -- which is why r_squared is
    returned too.
    """
    times = np.asarray(times, dtype=float)
    errors = np.asarray(errors, dtype=float)

    mask = (errors > floor) & np.isfinite(errors) & (times > 0)
    if mask.sum() < 10:
        return {"growth_rate": np.nan, "e_folding_time": np.nan,
                "decade_time": np.nan, "r_squared": np.nan}

    t_fit, log_e = times[mask], np.log(errors[mask])
    alpha, intercept = np.polyfit(t_fit, log_e, 1)

    residual = log_e - (alpha * t_fit + intercept)
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((log_e - log_e.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "growth_rate": float(alpha),                        # per year
        "e_folding_time": float(1.0 / alpha) if alpha > 0 else np.inf,
        "decade_time": float(np.log(10) / alpha) if alpha > 0 else np.inf,
        "r_squared": float(r_squared),
    }


def time_to_error(times, errors, threshold):
    """
    First time the error exceeds `threshold`, by linear interpolation.
    The operational version of "how long is this forecast good for?".
    """
    times = np.asarray(times, dtype=float)
    errors = np.asarray(errors, dtype=float)
    above = np.where(errors > threshold)[0]
    if len(above) == 0:
        return np.inf
    i = above[0]
    if i == 0:
        return float(times[0])
    e0, e1 = errors[i - 1], errors[i]
    frac = (threshold - e0) / (e1 - e0) if e1 != e0 else 0.0
    return float(times[i - 1] + frac * (times[i] - times[i - 1]))
