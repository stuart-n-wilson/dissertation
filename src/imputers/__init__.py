from .base import BaseImputer
from .mean import MeanImputer
from .locf import LOCFImputer
from .ks import KSImputer
from .arima import ARIMAImputer
from .brits import BRITSImputer


IMPUTERS: dict[str, type[BaseImputer]] = {
    "mean": MeanImputer,
    "locf": LOCFImputer,
    "ks": KSImputer,
    "arima": ARIMAImputer,
    "brits": BRITSImputer,
}


def get_imputer(name: str, **kwargs) -> BaseImputer:
    """Look up an imputer by name and return a fresh instance."""
    if name not in IMPUTERS:
        raise ValueError(f"Unknown imputer {name!r}. Available: {list(IMPUTERS)}")
    return IMPUTERS[name](**kwargs)