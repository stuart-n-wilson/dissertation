# Libaries
import numpy as np
import pandas as pd

# --- Internal functions ---

# logistic steepness -> controls how strongly the mechanism concentrates missingness
K = 4

def _create_mar_mask(df:pd.DataFrame, n:float) -> pd.DataFrame:
    """
    Creates a missingness mask using synthetic MAR. Probability of missingness uses most
    correlated features.
    """
    N = int((n / 100) * df.size)
    arr = df.to_numpy(dtype=float)

    # Calculate feature Pearson correlations.
    corr = np.abs(np.corrcoef(arr, rowvar=False))

    # Fill diagonal with -inf to prevent self corr.
    np.fill_diagonal(corr, -np.inf)

    # List of indices for most correlated feature for each feature.
    drivers = corr.argmax(axis=1)

    # Create the weights matrix - w[i, j] is how much df[i,j] wants to be missing.
    W = np.empty_like(arr)
    
    # Column j, driver d.
    for j, d in enumerate(drivers):

        # Calculate z scores by driver column.
        zd = (arr[:, d] - arr[:, d].mean()) / arr[:, d].std()

        # Create weights matrix using sigmoid function.
        W[:, j] = 1.0 / (1.0 + np.exp(-K * zd))

    w = W.flatten()

    # Convert weights into probabilities.
    w = w / w.sum()

    # Randomly select without repeats using probability weights.
    idx = np.random.choice(w.size, N, replace=False, p=w)

    mask = np.zeros(w.size, dtype=bool)
    mask[idx] = True
    mask = mask.reshape(df.shape)

    return pd.DataFrame(mask, columns=df.columns, index=df.index)

def _create_mcar_mask(df:pd.DataFrame, n:float) -> pd.DataFrame:
    """
    Creates a missingness mask using synthetic MCAR.
    """
    N = int((n / 100) * df.size)
    arr = df.to_numpy(dtype=float)

    mask = np.zeros(len(arr.flatten()), dtype=bool)

    # Randomly select without repeats.
    idx = np.random.choice(len(df.to_numpy().flatten()), N, replace=False)

    mask[idx] = True
    mask = mask.reshape(df.shape)

    return pd.DataFrame(mask, columns=df.columns, index=df.index)

def _create_mnar_mask(df:pd.DataFrame, n:float) -> pd.DataFrame:
    """
    Synthetic MNAR: missingness in each cell is driven by its own value,
    through a per-column-standardised logistic.
    """
    N = int((n / 100) * df.size)
    arr = df.to_numpy(dtype=float)

    # Calculate z-score per column.
    Z = (arr - arr.mean(axis=0)) / arr.std(axis=0)
    W = 1 / (1 + np.exp(-K * Z))

    w = W.flatten()

    # Convert weights into probabilities.
    w = w / w.sum()

    # Randomly select without repeats using probability weights.
    idx = np.random.choice(w.size, N, replace=False, p=w)

    mask = np.zeros(w.size, dtype=bool)
    mask[idx] = True
    mask = mask.reshape(df.shape)

    return pd.DataFrame(mask, columns=df.columns, index=df.index)
                        
# --- External functions ---

def create_mask(
    df: pd.DataFrame,
    mechanism: str,
    n: float,
    random_seed: int = 42
) -> pd.DataFrame:
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


def simulate_missingness(
    df: pd.DataFrame,
    mechanism: str,
    n: float,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

def repeated_simulated_missingness(
    df: pd.DataFrame,
    mechanisms: list[str],
    proportions: list[float],
    iterations: int
) -> tuple[dict[str, list[pd.DataFrame]], dict[str, list[pd.DataFrame]]]:
    """
    Repeatedly simulates missingness in a DataFrame using different generated masks,
    returning dictionaries of the DataFrames and masks.

    Each iteration uses a different random seed, producing a unique mask and
    corresponding missing-data DataFrame per repeat.

    Args:
        df: The input DataFrame to introduce missing values into.
        mechanisms: List of missingness mechanisms - mar, mcar or mnar.
        proportions: List of missingness percentages, values between 0 and 100.
        iterations: Number of repeats per mechanism-proportion combination.

    Returns:
        missing_dict: Dictionary with key:value as mechanism-proportion:[list of DataFrames with missingness].
        mask_dict: Dictionary with key:value as mechanism-proportion:[list of missingness mask DataFrames].
    """

    missing_dict = {}
    mask_dict = {}

    for i in range(iterations):

        for mech in mechanisms:

            for n in proportions:

                key = mech + str(n)

                if key not in mask_dict:
                    mask_dict[key] = []

                if key not in missing_dict:
                    missing_dict[key] = []
                
                df_missing, mask = simulate_missingness(df=df, mechanism=mech, n=n, random_seed=i)

                missing_dict[key].append(df_missing)
                mask_dict[key].append(mask)

    return missing_dict, mask_dict