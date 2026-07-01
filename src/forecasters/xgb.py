import numpy as np
import pandas as pd
import xgboost as xgb
from .base import BaseForecaster, TARGET

def _build_features(df: pd.DataFrame, lags: list) -> pd.DataFrame:
    """
    Create time series and lag features.
    """

    df_ts_fe = df.copy()

    # Time series features extracted from index.
    df_ts_fe["month"] = df_ts_fe.index.month
    df_ts_fe["dayofweek"] = df_ts_fe.index.dayofweek
    df_ts_fe["hour"] = df_ts_fe.index.hour

    # Lag features, n observations into the past for each original feature.
    # Must read columns from df so to not lag wrong cols.
    for n in lags:
        for col in df.columns:
            df_ts_fe[f"{col}_lag_{n}"] = df[col].shift(n)

    return df_ts_fe


class XGBoostForecaster(BaseForecaster):
    """Gradient-boosted trees on lagged + calendar features (multivariate-in, OT-out)."""

    name = "xgboost"

    def __init__(self, lags: list | None = None, random_state: int = 0, **xgb_params):
        self.lags = lags if lags is not None else [1, 24, 168]
        self.random_state = random_state
        self.xgb_params = xgb_params

    def fit(self, train: pd.DataFrame) -> "XGBoostForecaster":
        self.history_ = train.iloc[-max(self.lags):].copy()
        self.freq_ = pd.tseries.frequencies.to_offset(
            train.index.freq or pd.infer_freq(train.index))
        self.last_time_ = train.index[-1]

        og_columns = list(train.columns)

        # Build features on training data.
        feats_df = _build_features(df=train, lags=self.lags)
        
        # Drop rows with NaN lags (first n rows - n is max lags).
        feats_df = feats_df.dropna()

        X = feats_df.drop(og_columns, axis=1)
        y = feats_df[TARGET]

        # Store X features
        self.features_ = list(X.columns)

        self.model_ = xgb.XGBRegressor(random_state = self.random_state, **self.xgb_params)
        self.model_ = self.model_.fit(X, y)

        return self

    def predict(self, horizon: int) -> np.ndarray:
        """
        Predict one observation at a time, create a loop of new df's to predict each new
        observation within the horizon.
        """

        buffer = self.history_.copy()
        last_time = self.last_time_
        preds = []

        for _ in range(horizon):

            # Timestamp of the step being predicted.
            last_time = last_time + self.freq_

            # Build features on the buffer, take the last row in the model's column order.
            feats = _build_features(buffer, self.lags)
            row = feats.iloc[[-1]][self.features_]

            pred = self.model_.predict(row)[0]
            preds.append(pred)

            # Append a new raw row so the next step's lags can see this prediction.
            new_row = buffer.iloc[[-1]].copy()
            new_row.index = [last_time]
            new_row[TARGET] = pred
            buffer = pd.concat([buffer, new_row])

        return np.array(preds)
