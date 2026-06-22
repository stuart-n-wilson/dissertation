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


def _bias():
    pass

def _variance():
    pass


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

def measure_prediction(test: pd.DataFrame, pred_dfs: pd.DataFrame):
    """
    Evaluates performance of forecasting.

    Args:
        test: Complete, orgininal test data.
        pred_dfs: List of forecasting predictions.
    
    Returns:
        RMSE, MAPE, bias, variance
    """    
    
    
    pass