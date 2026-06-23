from .base import BaseForecaster
from .xgb import XGBoostForecaster
from .exp_smoothing import ExpSmoothingForecaster

FORECASTERS: dict[str, type[BaseForecaster]] = {
    cls.name: cls for cls in (XGBoostForecaster, ExpSmoothingForecaster)
}

def get_forecaster(name: str, **kwargs) -> BaseForecaster:
    if name not in FORECASTERS:
        raise ValueError(f"Unknown forecaster {name!r}. Available: {list(FORECASTERS)}")
    return FORECASTERS[name](**kwargs)