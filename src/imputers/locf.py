import pandas as pd
from .base import BaseImputer

class LOCFImputer(BaseImputer):
    """
    Last observation carried forward (forward fill).
    """

    name = "locf"

    def transform(self, incomplete:pd.DataFrame) -> pd.DataFrame:
        """
        ffill imputes next value, any remaining values with with bfill.
        """
        return incomplete.ffill().bfill()