import numpy as np
import pandas as pd
from .base import BaseForecaster, TARGET


class XGBoostForecaster(BaseForecaster):
    """Gradient-boosted trees on lagged + calendar features (multivariate-in, OT-out)."""

    name = "xgboost"

    def __init__(self, n_lags: int = 96, random_state: int = 0, **xgb_params):
        self.n_lags = n_lags                # how far back the lag features reach
        self.random_state = random_state    # fixed -> XGB adds no variance of its own
        self.xgb_params = xgb_params

    def fit(self, train: pd.DataFrame) -> "XGBoostForecaster":
        # TODO (Step 16):
        #   build supervised (X, y) from `train`:
        #     - lagged values of OT *and* the other 6 columns (1..n_lags)
        #     - calendar features from the index (hour, day-of-week, ...)
        #     - target y = OT at t
        #   fit the regressor with random_state=self.random_state  -> self.model_
        self.history_ = train.iloc[-self.n_lags:].copy()   # buffer to seed forecasting
        self.freq_ = train.index.freq
        self.last_time_ = train.index[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        # TODO (Step 16): multi-step forecast of OT.
        #   recursive sketch: roll a buffer seeded from self.history_; each step
        #   build one feature row, predict OT, append, advance the timestamp.

        # --- temporary stub so notebook 03 runs end-to-end; delete at Step 16 ---
        return np.full(horizon, self.history_[TARGET].iloc[-1])