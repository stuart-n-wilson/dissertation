import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ETT_PATH = PROJECT_ROOT / "data" / "raw" / "ETTh1.csv"

def load_ett(path: str | Path = DEFAULT_ETT_PATH) -> pd.DataFrame:
    """Load the ETTm1 dataset from CSV with a DatetimeIndex.

    Args:
        path: Path to the ETTm1 CSV file.

    Returns:
        DataFrame of the 7 feature columns indexed by datetime.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{path}\n\n"
            f"PROJECT_ROOT = {PROJECT_ROOT}"
        )
    
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    
    return df.asfreq("1h")

def chronological_split(
    df: pd.DataFrame, test_size: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series into train and test segments without shuffling.

    Args:
        df: Full DataFrame indexed chronologically.
        test_size: Fraction of rows to reserve for the test set.

    Returns:
        Tuple of (train, test), where train contains the earliest
        (1 - test_size) rows and test contains the latest test_size rows.
    """
    cut = int(len(df) * (1 - test_size))
    train = df.iloc[:cut]
    test = df.iloc[cut:]
    return train, test


def scale(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Scale train and test data to [0, 1] using a MinMaxScaler fit on train only.

    Args:
        train: Training DataFrame.
        test: Test DataFrame.

    Returns:
        Tuple of (scaled train DataFrame, scaled test DataFrame, fitted scaler).
        Test values may fall outside [0, 1] if they exceed the training range.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_s = pd.DataFrame(
        scaler.fit_transform(train), index=train.index, columns=train.columns
    )
    test_s = pd.DataFrame(
        scaler.transform(test), index=test.index, columns=test.columns
    )
    return train_s, test_s, scaler