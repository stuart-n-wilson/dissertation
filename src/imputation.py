# Libaries
import numpy as np
import pandas as pd


# --- Internal functions ---

def _impute_locf(df: pd.DataFrame):
    pass

def _impute_ks(df: pd.DataFrame):
    pass

def _impute_arima(df: pd.DataFrame):
    pass

def _impute_lstm(df: pd.DataFrame):
    pass

# --- External functions ---

def impute_ts(df: pd.DataFrame, model: str):
    
    if model not in ["locf", "ks", "arima", "lstm"]:
        raise ValueError("Model must be locf, ks, arima or lstm.")
    
    if model == "locf":
        _impute_locf(df)

    elif model == "ks":
        _impute_ks(df)
    
    elif model == "arima":
        _impute_arima(df)
    
    elif model == "lstm":
        _impute_lstm(df)
    
