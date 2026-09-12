"""
Shared evaluation utilities for the whole study.

Every stage of this project ends in the same question -- "how far is the
prediction from the truth?" -- so the metrics live in one place. Using one
implementation everywhere means a number quoted in Stage 12 is directly
comparable with the same number quoted in Stage 5.

The four metrics, in plain language
-----------------------------------
MAE  : mean |error|. Same units as the quantity (metres, AU). Typical error.
MSE  : mean error^2. Punishes a few large errors much harder than many small
       ones. Not in physical units, so we rarely report it directly.
RMSE : sqrt(MSE). Back in physical units, but still dominated by the worst
       cases. If RMSE >> MAE, the error distribution has a heavy tail.
R^2  : 1 - SS_res/SS_tot. Unitless. "What fraction of the variance in the
       target did the model explain?" 1.0 = perfect, 0.0 = no better than
       always predicting the mean, negative = worse than the mean.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred):
    """Root mean squared error, in the same units as y."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_row(y_true, y_pred):
    """All four metrics for a single 1-D target, as a dict."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_multioutput(y_true, y_pred, target_names):
    """
    Metrics for each output column separately, returned as a tidy DataFrame
    (one row per target). Multi-output models are evaluated per coordinate
    because "3 m of error in x" and "3 m of error in y" mean different things
    physically and should never be silently averaged together.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rows = []
    for i, name in enumerate(target_names):
        row = {"target": name}
        row.update(metric_row(y_true[:, i], y_pred[:, i]))
        rows.append(row)
    return pd.DataFrame(rows)


def radial_error(y_true, y_pred):
    """
    Euclidean distance between predicted and true position, per sample.

    For a 2-D position this is the only error measure a physicist really
    cares about: "how far away, in metres, is the predicted point from the
    true point?" -- independent of the coordinate system we happened to use.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.sum((y_pred - y_true) ** 2, axis=1))


def summarise_radial(y_true, y_pred):
    """Mean / median / 95th-percentile / max radial error, as a dict."""
    err = radial_error(y_true, y_pred)
    return {
        "mean_radial": float(err.mean()),
        "median_radial": float(np.median(err)),
        "p95_radial": float(np.percentile(err, 95)),
        "max_radial": float(err.max()),
    }


def format_metric_table(df, float_fmt="{:.4g}"):
    """Render a metrics DataFrame as a GitHub-flavoured Markdown table."""
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda v: float_fmt.format(v))
    header = "| " + " | ".join(formatted.columns) + " |"
    divider = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in formatted.itertuples(index=False)
    ]
    return "\n".join([header, divider, *body])
