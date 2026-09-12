"""
Stage 11 dataset generation -- orbital trajectories for machine learning.

The projectile dataset (Part I) was a bag of INDEPENDENT rows: each row was
its own little universe, and the order of the rows carried no information.
Orbital data is not like that. A trajectory is a sequence, each state caused
by the one before it, and that difference is the whole reason Part II is
harder than Part I.

We therefore generate and store whole trajectories, tagged with a trajectory
id, and build the ML tables from them afterwards. Three tables come out of the
same raw trajectories, one per modelling approach:

  A. one-step pairs     s_t -> s_{t+dt}      (a learned flow map)
  B. direct map         (a, e, t) -> (x, y)  (a learned closed-form solution)
  C. history pairs      [s_{t-k+1} ... s_t] -> s_{t+dt}

Keeping them derived from the SAME trajectories matters: any difference in
accuracy between the approaches is then a property of the formulation, not of
the data they happened to get.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from src.planetary_physics import (
    GM,
    kepler_solution,
    orbital_elements,
    simulate_reference,
    specific_angular_momentum,
    specific_energy,
    state_from_elements,
)

STATE_COLUMNS = ["x", "y", "vx", "vy"]

# The interval the ML model steps over. 0.004 yr is 250 stored points per
# 1-year orbit: fine enough that consecutive states are strongly related
# (so the map is learnable), coarse enough that five orbits is ~1,250 model
# calls rather than tens of thousands.
DT_STORE = 0.004

# RK4 substeps taken between stored points. 10 puts the reference trajectory
# at ~4e-11 AU per orbit -- utterly negligible next to any ML error.
SUBSTEPS = 10


def generate_trajectory(a, e, n_orbits, dt_store=DT_STORE, substeps=SUBSTEPS,
                        traj_id=0):
    """
    One orbit, integrated with RK4 and returned as a tidy DataFrame.

    Every row carries the orbital parameters (a, e) that produced it as well
    as the state itself, so that Approach B -- which needs to know WHICH orbit
    it is on -- can be built from the same table.
    """
    state0 = state_from_elements(a, e)
    T = 2.0 * np.pi * np.sqrt(a ** 3 / GM)
    n_store = int(round(n_orbits * T / dt_store))

    times, states = simulate_reference(state0, dt_store, n_store, substeps)

    frame = pd.DataFrame(states, columns=STATE_COLUMNS)
    frame.insert(0, "t", times)
    frame.insert(0, "traj_id", traj_id)
    frame["a"] = a
    frame["e"] = e
    frame["period"] = T
    frame["r"] = np.hypot(frame["x"], frame["y"])
    frame["speed"] = np.hypot(frame["vx"], frame["vy"])
    frame["energy"] = specific_energy(states)
    frame["ang_mom"] = specific_angular_momentum(states)
    return frame


def generate_trajectory_set(n_trajectories, a_range=(0.8, 1.2), e_range=(0.0, 0.4),
                            n_orbits=3.0, seed=11, start_id=0):
    """
    A family of trajectories with (a, e) drawn uniformly from the given boxes.

    Why vary a and e at all, rather than learning one single orbit? Because a
    model trained on one trajectory has no way to distinguish "the law of
    motion" from "this particular curve", and would be a lookup table in
    disguise. Varying the orbit forces the one-step model to learn a mapping
    that depends on the STATE, which is the only thing that could plausibly
    be called learning the dynamics.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_trajectories):
        a = rng.uniform(*a_range)
        e = rng.uniform(*e_range)
        frames.append(generate_trajectory(a, e, n_orbits, traj_id=start_id + i))
    return pd.concat(frames, ignore_index=True)


# -------------------- APPROACH A: ONE-STEP PAIRS --------------------

def make_one_step_pairs(trajectories):
    """
    Turn trajectories into (state now -> state one dt later) pairs.

    The shift must be done WITHIN each trajectory, never across the boundary
    between two of them, or the dataset would contain a handful of physically
    impossible transitions -- the end of one orbit followed by the start of an
    unrelated one. Those few poisoned rows would be learned just as eagerly as
    the correct ones.
    """
    pieces = []
    for _, group in trajectories.groupby("traj_id", sort=False):
        current = group.iloc[:-1].reset_index(drop=True)
        nxt = group.iloc[1:].reset_index(drop=True)

        piece = current[["traj_id", "t", "a", "e"] + STATE_COLUMNS].copy()
        for column in STATE_COLUMNS:
            piece[f"next_{column}"] = nxt[column].to_numpy()
        pieces.append(piece)

    return pd.concat(pieces, ignore_index=True)


# -------------------- APPROACH B: DIRECT MAP --------------------

def make_direct_map_table(trajectories):
    """
    (a, e, t) -> (x, y). The learned equivalent of the closed-form Kepler
    solution: ask for a time, get a position, with no stepping involved.
    """
    columns = ["traj_id", "a", "e", "t"] + STATE_COLUMNS
    return trajectories[columns].copy()


# -------------------- APPROACH C: HISTORY PAIRS --------------------

def make_history_pairs(trajectories, history=3):
    """
    [s_{t-k+1}, ..., s_t] -> s_{t+dt}, again strictly within a trajectory.

    Feeding the model a window of past states gives it access to information
    that a single state does not obviously contain -- curvature, and hence
    something like the acceleration. Whether that actually helps is an
    empirical question Stage 11 answers rather than assumes.
    """
    pieces = []
    for _, group in trajectories.groupby("traj_id", sort=False):
        values = group[STATE_COLUMNS].to_numpy()
        n = len(values) - history
        if n <= 0:
            continue

        data = {"traj_id": group["traj_id"].iloc[0],
                "t": group["t"].to_numpy()[history - 1: history - 1 + n],
                "a": group["a"].iloc[0], "e": group["e"].iloc[0]}

        for lag in range(history):
            # lag 0 = oldest state in the window, lag history-1 = most recent
            for j, column in enumerate(STATE_COLUMNS):
                data[f"{column}_lag{history - 1 - lag}"] = values[lag: lag + n, j]
        for j, column in enumerate(STATE_COLUMNS):
            data[f"next_{column}"] = values[history: history + n, j]

        pieces.append(pd.DataFrame(data))

    return pd.concat(pieces, ignore_index=True)


def history_feature_names(history=3):
    """Column order for Approach C, most recent state last."""
    return [
        f"{column}_lag{lag}"
        for lag in range(history - 1, -1, -1)
        for column in STATE_COLUMNS
    ]


# -------------------- MAIN --------------------

if __name__ == "__main__":
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)

    print("Generating planetary trajectories with RK4 "
          f"(dt_store = {DT_STORE} yr, {SUBSTEPS} substeps)")

    train = generate_trajectory_set(60, n_orbits=3.0, seed=11, start_id=0)
    test = generate_trajectory_set(20, n_orbits=3.0, seed=99, start_id=1000)

    for name, frame in [("train", train), ("test", test)]:
        path = out_dir / f"planetary_{name}_trajectories.csv"
        frame.to_csv(path, index=False)
        n_traj = frame["traj_id"].nunique()
        drift = (frame.groupby("traj_id")["energy"]
                 .apply(lambda s: abs(s.iloc[-1] - s.iloc[0]) / abs(s.iloc[0])).max())
        print(f"  {name:5s}: {n_traj} trajectories, {len(frame):,} states "
              f"-> {path.name}")
        print(f"         a in [{frame['a'].min():.3f}, {frame['a'].max():.3f}] AU, "
              f"e in [{frame['e'].min():.3f}, {frame['e'].max():.3f}], "
              f"worst relative energy drift = {drift:.2e}")

    pairs = make_one_step_pairs(train)
    pairs.to_csv(out_dir / "planetary_one_step_pairs.csv", index=False)
    print(f"  one-step pairs: {len(pairs):,} rows -> planetary_one_step_pairs.csv")

    # Independent check that the stored trajectories really are Kepler orbits.
    sample = train[train["traj_id"] == 0]
    exact = kepler_solution(sample["t"].to_numpy(), sample["a"].iloc[0],
                            sample["e"].iloc[0])
    err = np.abs(sample[["x", "y"]].to_numpy() - exact[:, 0:2]).max()
    print(f"\n  verification: trajectory 0 differs from the analytic Kepler "
          f"solution by at most {err:.2e} AU")
