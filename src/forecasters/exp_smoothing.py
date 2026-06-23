import numpy as np
import pandas as pd
from .base import BaseForecaster, TARGET


class ExpSmoothingForecaster(BaseForecaster):
    """Holt-Winters on OT (univariate, daily seasonality)."""

    name = "exp_smoothing"

    def __init__(self, seasonal_periods: int = 96, seasonal: str = "add",
                 trend: str | None = "add"):
        self.seasonal_periods = seasonal_periods   # 96 = one day at 15-min steps
        self.seasonal = seasonal
        self.trend = trend

    def fit(self, train: pd.DataFrame) -> "ExpSmoothingForecaster":
        # TODO (Step 16): fit Holt-Winters on train[TARGET] with
        #   ExponentialSmoothing(train[TARGET], trend=self.trend,
        #       seasonal=self.seasonal, seasonal_periods=self.seasonal_periods).fit()
        #   store the fitted result -> self.result_
        self.train_target_ = train[TARGET]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        # TODO (Step 16): return np.asarray(self.result_.forecast(horizon))

        # --- temporary stub so notebook 03 runs end-to-end; delete at Step 16 ---
        return np.full(horizon, self.train_target_.iloc[-1])