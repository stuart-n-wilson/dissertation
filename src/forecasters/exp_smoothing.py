import numpy as np
import pandas as pd
from .base import BaseForecaster, TARGET
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ExpSmoothingForecaster(BaseForecaster):
    """Holt-Winters on OT (univariate, daily seasonality)."""

    name = "exp_smoothing"

    def __init__(self, seasonal_periods: int = 96, seasonal: str = "add",
                 trend: str | None = "add", damped_trend: bool = True):
        self.seasonal_periods = seasonal_periods   # 96 = one day at 15-min steps
        self.seasonal = seasonal
        self.trend = trend
        self.damped_trend = damped_trend

    def fit(self, train: pd.DataFrame) -> "ExpSmoothingForecaster":
        """Fit Holt-Winters model."""
        self.result_ = ExponentialSmoothing(
            train[TARGET],
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods).fit()
        
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Predict a horizon with trained Holt-Winters model."""
        return np.asarray(self.result_.forecast(horizon))