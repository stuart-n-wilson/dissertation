import pandas as pd
import numpy as np

# -- Internal functions ---

def _rmse(array: np.ndarray) -> float:
    """
    Calculate RMSE from an array of errors (predicted - true).
    """
    return np.sqrt(np.mean(array ** 2))

def _mae(array: np.ndarray) -> float:
    """
    Calculate MAE from an array of errors (predicted - true).
    """
    return np.mean(abs(array))


# --- External functions ---

def reconstruction_errors(
    true: pd.DataFrame,
    imputed: pd.DataFrame,
    mask: pd.DataFrame,
) -> tuple[float, float]:
    """RMSE and MAE at the masked cells, for one run.

    Args:
        true: the complete, original training frame (ground truth).
        imputed: the reconstructed frame for this run.
        mask: boolean frame, True where a value was removed.

    Returns:
        (rmse, mae) over the masked cells only.
    """
    errors = (imputed - true).to_numpy()[mask.to_numpy()]
    return _rmse(errors), _mae(errors)

def pointwise_bias_variance(
        forecasts: np.ndarray,
        reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
     """
     Per-horizon-step bias² and variance across the seed forecasts.

    Args:
        forecasts: shape (n_seeds, horizon) — one forecast per seed, for a single
            (mechanism, proportion, imputer, forecaster) condition.
        reference: shape (horizon,) — the curve bias is measured against
            (true OT, or the complete-data baseline forecast).

    Returns:
        (bias_sq, variance), each of shape (horizon,).
    """
     
     mean_forecast = forecasts.mean(axis=0)
     bias_sq = (mean_forecast - reference) ** 2
     variance = forecasts.var(axis=0)

     return bias_sq, variance

def decompose_bias_variance(
        forecasts: np.ndarray,
        reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    
    """Horizon-averaged bias² and variance — one (bias², variance) per condition (mech, prop, imp, forecast)."""

    bias_sq, variance = pointwise_bias_variance(forecasts, reference)

    return float(bias_sq.mean()), float(variance.mean())
