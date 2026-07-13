import pandas as pd
import numpy as np
from .base import BaseImputer
from statsmodels.tsa.statespace.sarimax import SARIMAX
import gc

class ARIMAImputer(BaseImputer):
    """
    ARIMA imputation model.
    """
    name = "arima"

    def fit(self, train:pd.DataFrame) -> "ARIMAImputer":

        self.smoothed_ = {}

        for col in train.columns:
            model = SARIMAX(train[col], order=(1,0,0), seasonal_order=(0, 1, 1, 24))
            res = model.fit(disp=False, maxiter=50)
            self.smoothed_[col] = res

            self.smoothed_[col] = np.asarray(
                res.predict(information_set="smoothed")
            ).ravel()
            del res, model
        
        return self
    
    def transform(self, incomplete:pd.DataFrame) -> pd.DataFrame:

        df = incomplete.copy()

        for col in df.columns:
            mask = df[col].isna().to_numpy()     
            smoothed = self.smoothed_[col]
            df.loc[mask, col] = smoothed[mask] 

        return df 