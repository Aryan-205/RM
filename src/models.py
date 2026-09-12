"""
The regression models used throughout the study, in one place.

Why a factory function instead of just calling sklearn inline?
Because an experiment is only valid if every arm of it uses *identical*
model settings. Stage 7 varies the training-set size; Stage 8 varies the
region of parameter space; Stage 12 compares two physical systems. In each
case the model must be the controlled variable, so all of them ask this
module for a fresh, untrained model with fixed hyperparameters.

The four families, and the reason each one is in the study
----------------------------------------------------------
linear    : the simplest possible baseline. It can only form weighted sums of
            the raw inputs, so it *cannot* represent v0*cos(theta)*t. Its
            failure is informative, not embarrassing -- it quantifies how much
            of the problem is genuinely non-linear.
poly2     : degree-2 polynomial features + ridge. It CAN form products of two
            inputs (v0*t, t^2), so it captures part of the physics, but it
            still cannot represent cos(theta) exactly.
forest    : Random Forest. Makes no functional assumption at all; it carves
            the input space into boxes and averages the training targets in
            each box. Very accurate inside the training region, and -- as
            Stage 8 shows -- completely unable to extrapolate outside it.
mlp       : a small neural network. Smooth, differentiable, and able to
            approximate any continuous function given enough data. Included
            so that the study is not accused of dismissing neural networks
            without testing one.

physics   : NOT a general model. It is linear regression on hand-built
            physics features (v0*cos(theta)*t, v0*sin(theta)*t, t^2, y0).
            In those coordinates the true mapping IS linear, so this model
            should be near-exact. It exists to make a point in Stage 13:
            the useful thing to inject into a model is the *structure* of
            the physics, not more data.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

RANDOM_STATE = 42

# Kept as module constants so the paper can quote them and the reader can
# check that every experiment really did use the same settings.
FOREST_KWARGS = {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1}
MLP_KWARGS = {
    "hidden_layer_sizes": (64, 64),
    "activation": "tanh",      # smooth, matching the smoothness of the physics
    "solver": "adam",
    "max_iter": 2000,
    "random_state": RANDOM_STATE,
    "early_stopping": False,
}

MODEL_LABELS = {
    "linear": "Linear Regression",
    "poly2": "Polynomial Ridge (deg 2)",
    "forest": "Random Forest",
    "mlp": "Neural Network (MLP)",
    "physics": "Physics-Informed Features",
}


def make_model(name):
    """Return a fresh, untrained model. Never reuse a fitted model."""
    if name == "linear":
        return LinearRegression()

    if name == "poly2":
        # Ridge rather than plain least squares because polynomial features
        # are strongly correlated with each other; a small penalty keeps the
        # fit numerically stable without meaningfully biasing it.
        return make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1e-6),
        )

    if name == "forest":
        return RandomForestRegressor(**FOREST_KWARGS)

    if name == "mlp":
        # Scaling is mandatory here: theta is ~1 rad while x can be ~200 m.
        # Without it the optimiser is dominated by the large-magnitude inputs.
        return make_pipeline(StandardScaler(), MLPRegressor(**MLP_KWARGS))

    raise ValueError(f"unknown model name: {name!r}")


def make_models(names):
    """Dict of fresh models, in the order given."""
    return {name: make_model(name) for name in names}


# -------------------- PHYSICS-INFORMED FEATURE MODEL --------------------

def projectile_physics_features(df):
    """
    Re-express [v0, theta, y0, t] in the coordinates the physics actually
    uses. The exact solution is

        x = (v0 cos(theta)) * t
        y = y0 + (v0 sin(theta)) * t - (g/2) * t^2

    so in the basis below both targets are *linear* with known coefficients
    (1, 0, 0, 0) and (0, 1, 1, -g/2). A linear model given these features is
    therefore solving a problem it can solve exactly.

    This is "physics-informed machine learning" in its simplest and most
    honest form: we are not teaching the model physics, we are handing it a
    coordinate system in which the physics is trivial.
    """
    v0 = np.asarray(df["v0"], dtype=float)
    theta = np.asarray(df["theta"], dtype=float)
    y0 = np.asarray(df["y0"], dtype=float)
    t = np.asarray(df["t"], dtype=float)

    return np.column_stack([
        v0 * np.cos(theta) * t,   # exactly x
        v0 * np.sin(theta) * t,   # the rising part of y
        y0,                       # the offset part of y
        t ** 2,                   # the gravity part of y
    ])


PHYSICS_FEATURE_NAMES = ["v0*cos(th)*t", "v0*sin(th)*t", "y0", "t^2"]


class ProjectilePhysicsModel:
    """
    Linear regression in the physics basis above, wrapped so it has the same
    .fit(DataFrame, y) / .predict(DataFrame) interface as every sklearn model
    in this project and can be dropped into the same experiment loops.
    """

    def __init__(self):
        self.linear = LinearRegression()

    def fit(self, X, y):
        self.linear.fit(projectile_physics_features(X), y)
        return self

    def predict(self, X):
        return self.linear.predict(projectile_physics_features(X))

    @property
    def learned_coefficients(self):
        """
        The fitted coefficients, which should reproduce the physical
        constants. Row 'y', column 't^2' should come out at -g/2 = -4.905.
        """
        return self.linear.coef_
