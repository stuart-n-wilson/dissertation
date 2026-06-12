# Libaries
import numpy as np
import pandas as pd

# --- Internal functions ---

def _create_mar_mask(df, n):
    return df

def _create_mnar_mask(df, n):
    return df

def _create_mcar_mask(df:pd.DataFrame, n:float) -> pd.DataFrame:
    """
    Creates a missingness mask using synthetic MCAR.
    """
    no_missing = int((n / 100) * df.size)

    mask = np.zeros(len(df.to_numpy().flatten()), dtype=bool)

    # Randomly select indices (without repeats) from flat.
    idx = np.random.choice(len(df.to_numpy().flatten()), no_missing, replace=False)

    mask[idx] = True
    mask = mask.reshape(df.shape)

    return pd.DataFrame(mask, columns=df.columns, index=df.index)


# --- External functions ---

def create_mask(df:pd.DataFrame, mechanism:str, n:float, random_seed:int=42):
    """
    Create a boolean missingness mask for a DataFrame.

    Args:
        df: The input DataFrame.
        mechanism: Missingness mechanism - mar, mcar or mnar.
        n: The percentage of values to set as missing, between 0 and 100.
        random_seed: Random seed, default is 42.

    Returns:
        mask: Boolean DataFrame of the same shape, index, and columns as df,
        with True at positions selected to be missing.
    """

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


def simulate_missingness(df:pd.DataFrame, mechanism:str, n:float, random_seed:int=42) -> pd.DataFrame:
    """
    Simulate missingness in a DataFrame using a generated mask.

    Args:
        df: The input DataFrame to introduce missing values into.
        mechanism: Missingness mechanism - mar, mcar or mnar.
        n: The percentage of values to set as missing, between 0 and 100.
        random_seed: Random seed, default is 42.

    Returns:
        df_missing: A new DataFrame of the same shape, index, and columns as df,
        with approximately (n / 100) * df.size values replaced by NaN.
        mask: Boolean DataFrame, True where values were made missing.
    """
    mask = create_mask(df, mechanism, n, random_seed)

    df_missing = df.copy().astype(float)
    df_missing[mask] = np.nan

    return df_missing, mask