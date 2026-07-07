import pandas as pd
from .base import BaseImputer


class MeanImputer(BaseImputer):
    """Fill each column's gaps with that column's observed mean."""

    name = "mean"

    def fit(self, incomplete: pd.DataFrame) -> "MeanImputer":
        self.means_ = incomplete.mean()
        return self

    def transform(self, incomplete: pd.DataFrame) -> pd.DataFrame:
        return incomplete.fillna(self.means_)