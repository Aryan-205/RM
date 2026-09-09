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
    dataset.to_csv(
        "../projectile_dataset.csv",
        index=False
    )

    print("\nDataset generated successfully.")
    print("Saved to: ../projectile_dataset.csv")
    print(f"Rows: {len(dataset)}, Columns: {list(dataset.columns)}")