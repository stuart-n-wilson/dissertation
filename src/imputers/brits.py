import numpy as np
import pandas as pd
from pypots.imputation import BRITS
from .base import BaseImputer


class BRITSImputer(BaseImputer):
    """Bidirectional LSTM-based imputation via PyPOTS."""

    name = "brits"

    def __init__(self, window: int = 24, rnn_hidden_size: int = 64,
                 epochs: int = 250, batch_size: int = 32,
                 device: str | None = "cuda", seed: int = 0):
        self.window = window
        self.rnn_hidden_size = rnn_hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

    def _window_up(self, df: pd.DataFrame) -> np.ndarray:
        """Trim to a multiple of window length, reshape to (n_samples, window, n_features)."""
        n_rows = (len(df) // self.window) * self.window
        return df.iloc[:n_rows].to_numpy().copy().reshape(-1, self.window, df.shape[1])

    def fit(self, incomplete: pd.DataFrame) -> "BRITSImputer":
        from pypots.utils.random import set_random_seed
        set_random_seed(self.seed)

        X = self._window_up(incomplete)
        self.model_ = BRITS(
            n_steps=self.window,
            n_features=incomplete.shape[1],
            rnn_hidden_size=self.rnn_hidden_size,
            epochs=self.epochs,
            batch_size=self.batch_size,
            device=self.device,
            verbose=False,
        )
        self.model_.fit({"X": X})
        return self

    def transform(self, incomplete: pd.DataFrame) -> pd.DataFrame:
        X = self._window_up(incomplete)
        flat = self.model_.impute({"X": X}).reshape(-1, incomplete.shape[1])

        n_rows = flat.shape[0]
        covered = incomplete.iloc[:n_rows]
        mask = covered.isna().to_numpy()

        vals = covered.to_numpy().copy()
        vals[mask] = flat[mask]

        filled = incomplete.copy()
        filled.iloc[:n_rows] = vals
        # tail rows beyond the last whole window: fallback fill
        filled = filled.ffill().bfill()
        return filled