import pandas as pd
from .base import BaseImputer

class ARIMAImputer(BaseImputer):
    """
    ARIMA imputation model.
    """
    name = "arima"

    # def __init__(self, order=(1, 0, 0)):
    #     self.order = order          
    #     # hyperparameter, set when you construct it

    # def fit(self, incomplete):
    #     self.models_ = ...          
    #     # fit per column, stash on self (trailing _ = "learned in fit")
    #     return self

    # def transform(self, incomplete):
    #     # read self.models_ to fill the NaNs
    #     pass


    # TEMPORARY IMPLEMENTATION
    def transform(self, incomplete:pd.DataFrame) -> pd.DataFrame:

        return incomplete.ffill().bfill()