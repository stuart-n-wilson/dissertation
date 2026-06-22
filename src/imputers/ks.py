import pandas as pd
from .base import BaseImputer

class KSImputer(BaseImputer):
    """
    Kalman Smoothing imputation model.
    """

    name = "ks"

    # TEMPORARY
    def transform(self, incomplete:pd.DataFrame) -> pd.DataFrame:

        return incomplete.ffill().bfill()