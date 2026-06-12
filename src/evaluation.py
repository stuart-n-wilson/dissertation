import pandas as pd
import numpy as np

# -- Internal functions ---

def _rmse(array: np.ndarray) -> float:
    """
    Calculate RMSE from an array of errors y_i-pred - y_i.
    """
    return np.sqrt(np.mean(array ** 2))

def _mae(array: np.ndarray) -> float:
    """
    Calculate MAE from an array of errors y_i-pred - y_i.
    """
    return np.mean(abs(array))


def _bias():
    pass

def _variance():
    pass


# --- External functions ---

def get_errors(
    df: pd.DataFrame,
    imputed_data: dict[str, dict[str, list[pd.DataFrame]]],
    mask_dict: dict[str, list[pd.DataFrame]],
    models: list[str]
) -> dict[str, dict[str, list[np.ndarray]]]:
    """
    Extract imputation errors at the masked (imputed) positions.

    Args:
        df: The complete, original DataFrame used as ground truth.
        imputed_data: Dictionary with key:value as model:{mechanism-proportion:[list of imputed DataFrames]}.
        mask_dict: Dictionary with key:value as mechanism-proportion:[list of missingness mask DataFrames].
        models: List of imputation model names.

    Returns:
        errors: Dictionary with key:value as model:{mechanism-proportion:[list of 1D numpy arrays
        of errors (imputed - true) at the masked positions]}.
    """
    
    errors = {}

    for model in models:

        for mech_prop, df_list in imputed_data[model].items():

            if model not in errors:
                errors[model] = {}

            errors[model][mech_prop] = []

            for i, df_imputed in enumerate(df_list):

                error_array = (df_imputed - df).to_numpy()
                mask_array = mask_dict[mech_prop][i].to_numpy()

                errors[model][mech_prop].append(error_array[mask_array])

    return errors


def measure_reconstruction(
    errors: dict[str, dict[str, list[np.ndarray]]],
    models: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Calculate aggregated reconstruction metrics from imputation errors.

    RMSE and MAE are computed per repeat, then averaged across repeats for each
    model and mechanism-proportion combination.

    Args:
        errors: Dictionary with key:value as model:{mechanism-proportion:[list of 1D numpy arrays of errors]}.
        models: List of imputation model names.

    Returns:
        reconstruction: Dictionary with key:value as model:{mechanism-proportion:{"rmse": value, "mae": value}}.
    """

    reconstruction = {}

    for model in models:

        for mech_prop, error_lists in errors[model].items():
            
            if model not in reconstruction:
                reconstruction[model] = {}

            rmse_list = []
            mae_list = []

            for error_list in error_lists:

                rmse_list.append(_rmse(error_list))
                mae_list.append(_mae(error_list))
        
            if mech_prop not in reconstruction[model]:
                reconstruction[model][mech_prop] = {}
        
            reconstruction[model][mech_prop]["rmse"] = np.mean(rmse_list)
            reconstruction[model][mech_prop]["mae"] = np.mean(mae_list)

    return reconstruction


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