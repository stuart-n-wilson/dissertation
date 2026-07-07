import pandas as pd
import numpy as np
from .base import BaseImputer
from statsmodels.tsa.statespace.structural import UnobservedComponents

class KSImputer(BaseImputer):
    """
    Kalman Smoothing imputation model.
    """

    name = "ks"

    def fit(self, train:pd.DataFrame) -> "KSImputer":

        self.models_ = {}

        for col in train.columns:
            model = UnobservedComponents(train[col], level="local level")
            res = model.fit(disp=False)
            self.models_[col] = res
        
        return self
    
    def transform(self, incomplete:pd.DataFrame) -> pd.DataFrame:

        df = incomplete.copy()

        for col in df.columns:
            mask = df[col].isna().to_numpy()     
            smoothed = np.asarray(
                self.models_[col].smoother_results.smoothed_forecasts
            ).ravel()
            df.loc[mask, col] = smoothed[mask] 

        return df 
