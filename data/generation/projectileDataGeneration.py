from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# -------------------- CONSTANTS --------------------

g = 9.81  # acceleration due to gravity (m/s^2)


# -------------------- FUNCTIONS --------------------

def generateProjectileData(numSamples, v0, theta, y0):
    """
    Generate projectile motion data.

    Parameters:
        numSamples : Number of points in the trajectory
        v0         : Initial velocity (m/s)
        theta      : Launch angle (radians)
        y0         : Initial height (m)

    Returns:
        pandas DataFrame containing time, x and y
    """

    # Calculate time of flight
    timeOfFlight = (
        v0 * np.sin(theta)
        + np.sqrt(
            (v0 * np.sin(theta)) ** 2
            + 2 * g * y0
        )
    ) / g

    # Generate time values
    t = np.linspace(0, timeOfFlight, numSamples)

    # Generate x(t)
    x = v0 * np.cos(theta) * t

    # Generate y(t)
    y = (
        y0
        + v0 * np.sin(theta) * t
        - 0.5 * g * t**2
    )

    # Create dataframe
    df = pd.DataFrame({
        "time": t,
        "x": x,
        "y": y
    })

    return df


def generateProjectileDataset(numSamples, seed=42):
    """
    Generate an ML-ready projectile dataset.

    Each row = one random (v0, theta, y0, t) input combination and
    the resulting (x, y) position. Rows are independent samples, not
    points along a single trajectory, so the model sees how x and y
    vary WITH v0, theta, y0 -- not just with t.

    Parameters:
        numSamples : number of rows to generate

    Returns:
        pandas DataFrame with columns v0, theta, y0, t, x, y
    """

    rng = np.random.default_rng(seed)

    # Feature ranges -- keep them physically reasonable
    v0 = rng.uniform(5, 50, numSamples)        # m/s
    theta = rng.uniform(np.deg2rad(5), np.deg2rad(85), numSamples)  # rad
    y0 = rng.uniform(0, 20, numSamples)        # m

    # Time of flight for each sample (when y returns to 0)
    timeOfFlight = (
        v0 * np.sin(theta)
        + np.sqrt((v0 * np.sin(theta)) ** 2 + 2 * g * y0)
    ) / g

    # Pick t randomly within [0, timeOfFlight] for each sample
    # so t is a feature too, not the sweep variable.
    t = rng.uniform(0, timeOfFlight)

    x = v0 * np.cos(theta) * t
    y = y0 + v0 * np.sin(theta) * t - 0.5 * g * t**2

    df = pd.DataFrame({
        "v0": v0,
        "theta": theta,
        "y0": y0,
        "t": t,
        "x": x,
        "y": y
    })

    return df


def timeOfFlight(v0, theta, y0):
    """
    Time at which the projectile returns to y = 0, from solving

        y0 + v0 sin(theta) t - (1/2) g t^2 = 0

    with the quadratic formula and keeping the positive root. Works
    elementwise on NumPy arrays as well as on scalars.

    Assumptions inherited from the physics: constant g, no air resistance,
    flat ground at y = 0, point mass.
    """
    vy0 = v0 * np.sin(theta)
    return (vy0 + np.sqrt(vy0 ** 2 + 2 * g * y0)) / g


def generateProjectileDatasetInRange(
    numSamples,
    v0_range=(5.0, 50.0),
    theta_deg_range=(5.0, 85.0),
    y0_range=(0.0, 20.0),
    t_fraction_range=(0.0, 1.0),
    seed=42,
):
    """
    The same simulator as generateProjectileDataset, but with every sampling
    range exposed as an argument.

    This is what makes Stage 8 possible. To ask "can the model extrapolate?"
    we must be able to build a training set from one region of parameter
    space and a test set from a DIFFERENT region. A generator with hard-coded
    ranges cannot do that.

    Parameters
    ----------
    numSamples       : number of independent rows to generate
    v0_range         : (low, high) initial speed in m/s
    theta_deg_range  : (low, high) launch angle in DEGREES (converted to rad)
    y0_range         : (low, high) initial height in m
    t_fraction_range : (low, high) fraction of each sample's OWN flight time
                       from which t is drawn. (0, 1) = anywhere in the flight;
                       (0, 0.6) = only the first 60% of the flight, which is
                       how we test extrapolation forward in time.
    seed             : RNG seed, so the dataset is reproducible

    Returns
    -------
    DataFrame with columns v0, theta (rad), y0, t, x, y
    """
    rng = np.random.default_rng(seed)

    v0 = rng.uniform(v0_range[0], v0_range[1], numSamples)
    theta = rng.uniform(
        np.deg2rad(theta_deg_range[0]), np.deg2rad(theta_deg_range[1]), numSamples
    )
    y0 = rng.uniform(y0_range[0], y0_range[1], numSamples)

    # t is drawn as a fraction of each sample's own flight time, so that a
    # slow, low launch and a fast, high launch are both sampled across their
    # whole trajectory rather than the fast one being sampled only near its
    # start.
    flight = timeOfFlight(v0, theta, y0)
    fraction = rng.uniform(t_fraction_range[0], t_fraction_range[1], numSamples)
    t = fraction * flight

    x = v0 * np.cos(theta) * t
    y = y0 + v0 * np.sin(theta) * t - 0.5 * g * t ** 2

    return pd.DataFrame({
        "v0": v0, "theta": theta, "y0": y0, "t": t, "x": x, "y": y,
        "t_fraction": fraction, "flight_time": flight,
    })


def plotProjectileData(dataset):
    """
    Plot the projectile trajectory.
    """

    plt.plot(
        dataset["x"],
        dataset["y"],
        "o"
    )

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Projectile Motion")

    plt.grid(True)
    plt.show()


# -------------------- MAIN --------------------

if __name__ == "__main__":

    # Generate ML dataset: many random (v0, theta, y0, t) -> (x, y) rows
    dataset = generateProjectileDataset(numSamples=2000)

    # Display first few rows
    print(dataset.head())

    # Save dataset
    outputPath = Path(__file__).resolve().parent.parent / "projectile_dataset.csv"
    dataset.to_csv(outputPath, index=False)

    print("\nDataset generated successfully.")
    print(f"Saved to: {outputPath}")
    print(f"Rows: {len(dataset)}, Columns: {list(dataset.columns)}")

# -------------------- PROJECTILE AS A DYNAMICAL SYSTEM (STAGE 12) --------------------
#
# Everything above treats projectile motion the way Part I did: as a lookup
# from (v0, theta, y0, t) to a position, which is possible only because a
# closed-form solution exists. Part II could not do that, so it learned a
# one-step FLOW MAP instead: state now -> state one dt later.
#
# Comparing "projectile with a closed-form map" against "orbit with a flow
# map" would confound two things at once -- the physical system AND the
# problem formulation. The functions below remove that confound by expressing
# projectile motion in exactly the same form as the orbital problem, so Stage
# 12 can change the physics while holding the formulation fixed.
#
# The exact flow map for projectile motion is worth writing out, because it is
# the reason the comparison comes out the way it does:
#
#     x'  = x + vx*dt
#     y'  = y + vy*dt - (1/2) g dt^2
#     vx' = vx
#     vy' = vy - g dt
#
# Every one of those is an AFFINE function of the state. So for a fixed dt the
# projectile flow map lies exactly inside the hypothesis space of ordinary
# linear regression, and a linear model should recover it to floating-point
# precision. The orbital flow map contains 1/r^3 and lies inside no model's
# hypothesis space. That single structural difference -- not "orbits are
# harder to fit" in any vague sense -- is what Stage 12 measures.

PROJECTILE_STATE_COLUMNS = ["x", "y", "vx", "vy"]


def projectileStateTrajectory(v0, theta, y0, dt, traj_id=0):
    """
    One projectile flight as a sequence of states sampled every `dt`, computed
    from the exact solution (not stepped), so the data carries no integrator
    error of its own.
    """
    flight = timeOfFlight(v0, theta, y0)
    n_steps = max(2, int(np.floor(flight / dt)))
    t = np.arange(n_steps + 1) * dt

    return pd.DataFrame({
        "traj_id": traj_id,
        "t": t,
        "x": v0 * np.cos(theta) * t,
        "y": y0 + v0 * np.sin(theta) * t - 0.5 * g * t ** 2,
        "vx": np.full_like(t, v0 * np.cos(theta)),
        "vy": v0 * np.sin(theta) - g * t,
        "v0": v0, "theta": theta, "y0": y0,
    })


def generateProjectileStateTrajectories(
    n_trajectories, dt=0.05, v0_range=(15.0, 40.0), theta_deg_range=(20.0, 70.0),
    y0_range=(0.0, 15.0), seed=17, start_id=0,
):
    """A family of projectile flights in state-sequence form."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_trajectories):
        v0 = rng.uniform(*v0_range)
        theta = rng.uniform(np.deg2rad(theta_deg_range[0]), np.deg2rad(theta_deg_range[1]))
        y0 = rng.uniform(*y0_range)
        frames.append(projectileStateTrajectory(v0, theta, y0, dt, traj_id=start_id + i))
    return pd.concat(frames, ignore_index=True)


def makeProjectileOneStepPairs(trajectories):
    """(state now -> state one dt later), shifted strictly within each flight."""
    pieces = []
    for _, group in trajectories.groupby("traj_id", sort=False):
        current = group.iloc[:-1].reset_index(drop=True)
        nxt = group.iloc[1:].reset_index(drop=True)
        piece = current[["traj_id", "t"] + PROJECTILE_STATE_COLUMNS].copy()
        for column in PROJECTILE_STATE_COLUMNS:
            piece[f"next_{column}"] = nxt[column].to_numpy()
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)
