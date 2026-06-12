# Libaries
import numpy as np
import pandas as pd


# --- Internal functions ---

def _impute_locf(df: pd.DataFrame):
    return df

def _impute_ks(df: pd.DataFrame):
    return df

def _impute_arima(df: pd.DataFrame):
    return df

def _impute_lstm(df: pd.DataFrame):
    return df

# --- External functions ---

def impute_ts(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Impute missing values in a time series DataFrame.

    Args:
        df: The input DataFrame containing missing values (NaN).
        model: Imputation model - locf, ks, arima or lstm.

    Returns:
        A new DataFrame of the same shape, index, and columns as df,
        with missing values imputed.
    """
    
    if model not in ["locf", "ks", "arima", "lstm"]:
        raise ValueError("Model must be locf, ks, arima or lstm.")
    
    if model == "locf":
        return _impute_locf(df)

    elif model == "ks":
        return _impute_ks(df)
    
    elif model == "arima":
        return _impute_arima(df)
    
    elif model == "lstm":
        return _impute_lstm(df)
    
def repeated_imputation(
    models: list[str],
    missing_dict: dict[str, list[pd.DataFrame]]
) -> dict[str, dict[str, list[pd.DataFrame]]]:
    """
    Impute every DataFrame in missing_dict with every imputation model.

    Args:
        models: List of imputation model names - locf, ks, arima or lstm.
        missing_dict: Dictionary with key:value as mechanism-proportion:[list of DataFrames with missingness].

    Returns:
        imputed_data: Dictionary with key:value as model:{mechanism-proportion:[list of imputed DataFrames]}.
    """

    imputed_data = {}

    for model in models:

        for mech_prop, df_list in missing_dict.items():
            
            if model not in imputed_data:
                imputed_data[model] = {}

            imputed_data[model][mech_prop] = []

            for df in df_list:
                imputed_data[model][mech_prop].append(impute_ts(df, model))

    return imputed_data