# Libaries
import numpy as np
import pandas as pd
import random

# --- Internal functions ---
def _simulate_mar(df, n):
    pass

def _simulate_mnar(df, n):
    pass

def _simulate_mcar(df:pd.DataFrame, n:float) -> pd.DataFrame:
    """
    Simulates Missing Completely At Random (MCAR) data in a DataFrame.
    """
    
    no_missing = int(n * df.size)
    
    # Flatten df into list.
    flat = df.to_numpy().flatten().astype(float)
    
    # Randomly select indices (without repeats) from flat.
    idx = np.random.choice(len(flat), no_missing, replace=False)

    # Make selected indices NaN.
    flat[idx] = np.nan

    reshaped = flat.reshape(df.shape)

    return pd.DataFrame(reshaped, columns=df.columns, index=df.index)


# --- External functions ---

def simulate_missingness(df:pd.DataFrame, mechanism:str, n:float, random_seed:int=42) -> pd.DataFrame:
    """
    Simulate missigness mechanisms.
    
    Args:
        df: The input DataFrame to introduce missing values into.
        mechanism: Missingness mechanisms - mar, mcar or mnar.
        n: The proportion of values to set as missing, between 0 and 1.
        random_seed: Random seed, default is 42.

    Returns:
        A new DataFrame of the same shape, index, and columns as df,
        with approximately n * df.size values replaced by NaN.
    """

    if not 0 <= n <= 1:
        raise ValueError("n must be between 0 and 1.")

    random.seed(random_seed)

    if mechanism not in ['mar', 'mnar', 'mcar']:
        raise ValueError('Mechanisms must be mar, mnar or mcar.')

    if mechanism == 'mar':
        return _simulate_mar(df, n)
    
    elif mechanism == 'mnar':
        return _simulate_mnar(df, n)

    elif mechanism == 'mcar':
        return _simulate_mcar(df, n)
    
