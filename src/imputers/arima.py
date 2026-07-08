import pandas as pd
import numpy as np
from .base import BaseImputer
from statsmodels.tsa.statespace.sarimax import SARIMAX

class ARIMAImputer(BaseImputer):
    """
    ARIMA imputation model.
    """
    name = "arima"

    def fit(self, train:pd.DataFrame) -> "ARIMAImputer":

        self.models_ = {}

        for col in train.columns:
            model = SARIMAX(train[col], order=(1,0,0), seasonal_order=(0, 1, 1, 24))
            res = model.fit(disp=False, maxiter=50)
            self.models_[col] = res
        
        return self
    
    def transform(self, incomplete:pd.DataFrame) -> pd.DataFrame:

        df = incomplete.copy()

        for col in df.columns:
            mask = df[col].isna().to_numpy()     
            smoothed = np.asarray(
                self.models_[col].predict(information_set="smoothed")
            ).ravel()
            df.loc[mask, col] = smoothed[mask] 

        return df 