from .base import BaseImputer
from .mean import MeanImputer
from .locf import LOCFImputer
from .ks import KSImputer
from .arima import ARIMAImputer
from .lstm import LSTMImputer


IMPUTERS: dict[str, type[BaseImputer]] = {
    "mean": MeanImputer,
    "locf": LOCFImputer,
    "ks": KSImputer,
    "arima": ARIMAImputer,
    "lstm": LSTMImputer,
}


def get_imputer(name: str, **kwargs) -> BaseImputer:
    """Look up an imputer by name and return a fresh instance."""
    if name not in IMPUTERS:
        raise ValueError(f"Unknown imputer {name!r}. Available: {list(IMPUTERS)}")
    return IMPUTERS[name](**kwargs)