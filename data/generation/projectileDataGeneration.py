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

    # Generate projectile data
    dataset = generateProjectileData(
        numSamples=100,
        v0=10,
        theta=np.pi / 4,
        y0=0
    )

    # Display first few rows
    print(dataset.head())

    # Plot trajectory
    plotProjectileData(dataset)

    # Save dataset
    dataset.to_csv(
        "../projectile.csv",
        index=False
    )

    print("\nDataset generated successfully.")
    print("Saved to: ../projectile.csv")