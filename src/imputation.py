# Libaries
import numpy as np
import pandas as pd
import random

# Internal functions ---
def _simulate_mar(df, n):
    pass

def _simulate_mnar(df, n):
    pass

def simulate_mcar(df:pd.DataFrame, n:float, random_seed:int=42) -> pd.DataFrame:
    """Simulates Missing Completely At Random (MCAR) data in a DataFrame.

    Completely randomly replaces a given proportion of values across the 
    entire DataFrame with NaN.

    Args:
        df: The input DataFrame to introduce missing values into.
        n: The proportion of values to set as missing, between 0 and 1.
        random_seed: Random seed, default is 42.

    Returns:
        A new DataFrame of the same shape, index, and columns as df,
        with approximately n * df.size values replaced by NaN.
    """
    
    random.seed(random_seed)

    if not 0 <= n <= 1:
        raise ValueError("n must be between 0 and 1.")
    
    no_missing = int(n * df.size)
    
    # Flatten df into list.
    flat = df.to_numpy().flatten().astype(float)
    
    # Randomly select indices (without repeats) from flat.
    idx = np.random.choice(len(flat), no_missing, replace=False)

    # Make selected indices NaN.
    flat[idx] = np.nan

    reshaped = flat.reshape(df.shape)

    return pd.DataFrame(reshaped, columns=df.columns, index=df.index)


# External functions ---

def simulate_missingness(df:pd.DataFrame, mech:str, n:float) -> pd.DataFrame:

    if mech not in ['mar', 'mnar', 'mcar']:
        raise ValueError('Mechanisms must be mar, mnar or mcar.')

    if mech == 'mar':
        _simulate_mar(df, n)
    
    elif mech == 'mnar':
        _simulate_mnar(df, n)

    # elif mech == 'mcar':
    #      _simulate_mcar(df, n)