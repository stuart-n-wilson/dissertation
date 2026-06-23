# src/forecasters/base.py
import numpy as np
import pandas as pd

TARGET = "OT"  


class BaseForecaster:
    """Interface every forecaster obeys."""

    name: str = "base"

    def fit(self, train: pd.DataFrame) -> "BaseForecaster":
        """Learn from the (reconstructed) training frame; return self."""
        raise NotImplementedError("Subclasses must implement fit().")

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next `horizon` steps of the target.

        Returns:
            1-D array of length `horizon`.
        """
        raise NotImplementedError("Subclasses must implement predict().")