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

def impute_ts(df: pd.DataFrame, model: str):
    
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
    
