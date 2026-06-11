# Libaries
import numpy as np
import pandas as pd

# --- Internal functions ---
def _simulate_mar(df, n):
    return df

def _create_mar_mask(df, n):
    return df

def _simulate_mnar(df, n):
    return df

def _create_mnar_mask(df, n):
    return df

def _simulate_mcar(df:pd.DataFrame, n:float) -> pd.DataFrame:
    """
    Simulates Missing Completely At Random (MCAR) data in a DataFrame.
    """
    
    no_missing = int((n / 100) * df.size)
    
    # Flatten df into list.
    flat = df.to_numpy().flatten().astype(float)
    
    # Randomly select indices (without repeats) from flat.
    idx = np.random.choice(len(flat), no_missing, replace=False)

    # Make selected indices NaN.
    flat[idx] = np.nan

    reshaped = flat.reshape(df.shape)

    return pd.DataFrame(reshaped, columns=df.columns, index=df.index)

def _create_mcar_mask(df:pd.DataFrame, n:float) -> pd.DataFrame:

    no_missing = int((n / 100) * df.size)

    mask = np.zeros(len(df.to_numpy().flatten()), dtype=bool)

    # Randomly select indices (without repeats) from flat.
    idx = np.random.choice(len(df.to_numpy().flatten()), no_missing, replace=False)

    mask[idx] = True
    mask = mask.reshape(df.shape)

    return pd.DataFrame(mask, columns=df.columns, index=df.index)


# --- External functions ---

def simulate_missingness(df:pd.DataFrame, mechanism:str, n:float, random_seed:int=42) -> pd.DataFrame:
    """
    Simulate missigness mechanisms.
    
    Args:
        df: The input DataFrame to introduce missing values into.
        mechanism: Missingness mechanisms - mar, mcar or mnar.
        n: The percentage of values to set as missing, between 0 and 100.
        random_seed: Random seed, default is 42.

    Returns:
        mask: Missingness mask, with True for missing values.
        df: A new DataFrame of the same shape, index, and columns as df,
        with approximately n * df.size values replaced by NaN.
    """
    np.random.seed(random_seed)

    if not 0 <= n <= 100:
        raise ValueError("n must be between 0 and 100.")

    if mechanism not in ['mar', 'mnar', 'mcar']:
        raise ValueError('Mechanisms must be mar, mnar or mcar.')

    if mechanism == 'mar':
        return _simulate_mar(df, n)
    
    elif mechanism == 'mnar':
        return _simulate_mnar(df, n)

    elif mechanism == 'mcar':
        return _simulate_mcar(df, n)

def create_mask(df, mechanism:str, n:float, random_seed:int=42):

    np.random.seed(random_seed)

    if not 0 <= n <= 100:
        raise ValueError("n must be between 0 and 100.")
    
    if mechanism  not in ["mar", "mcar", "mnar"]:
        raise ValueError("Mechanism must be mar, mcar or mnar.")

    if mechanism  == "mar":
        return _create_mar_mask(df, n)

    elif mechanism  == "mcar":
        return _create_mcar_mask(df, n)
    
    elif mechanism  == "mnar":
        return _create_mnar_mask(df, n)