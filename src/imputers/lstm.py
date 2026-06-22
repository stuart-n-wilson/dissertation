import pandas as pd
from .base import BaseImputer

class LSTMImputer(BaseImputer):
    """
    Long Short-Term Memory imputation model.
    """

    name = "lstm"

    # TEMPORARY
    def transform(self, incomplete:pd.DataFrame) -> pd.DataFrame:

        return incomplete.ffill().bfill()