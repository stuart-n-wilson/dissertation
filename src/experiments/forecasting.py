import pandas as pd
import numpy as np
from src.experiments.artifacts import ReconstructedStore
from src.forecasters import get_forecaster
from src.evaluation import decompose_bias_variance
from collections import defaultdict
from tqdm.auto import tqdm


class ForecastingExperiment:

    def __init__(self, forecasters):

        self.forecasters = forecasters

    def run(
        self,
        horizon: int,
        store: ReconstructedStore,
        truth: np.ndarray,
        baselines: dict[str, np.ndarray],
        unscale_fn: function
    ) -> pd.DataFrame:
        """Forecast from every reconstructed dataset and decompose error per condition.

        Args:
            horizon: forecast length H.
            store: populated store of reconstructed training datasets.
            truth: true target over the horizon, shape (H,).
            baselines: complete-data baseline forecast, keyed by forecaster name.

        Returns:
            One row per (mechanism, proportion, imputer, forecaster) with
            bias_vs_truth, bias_vs_baseline and variance.
        """

        predictions = defaultdict(list)

        for origin in tqdm(list(store.keys()), desc="forecasting"):

            reconstructed = store.get(origin)

            for model in self.forecasters:

                forecaster = get_forecaster(model)

                fitted = forecaster.fit(reconstructed)

                # Key forecast by mechanism, proportion, imputer and forecaster.
                predictions[(origin.mechanism, origin.proportion, origin.imputer, model)].append(unscale_fn(fitted.predict(horizon)))

        records = []

        for (mechanism, proportion, imputer, forecaster), forecast_list in predictions.items():

            preds = np.array(forecast_list)
            bias_vs_truth, variance = decompose_bias_variance(preds, truth)
            bias_vs_baseline, _ = decompose_bias_variance(preds, baselines[forecaster])

            records.append({
                "mechanism": mechanism, "proportion": proportion,
                "imputer": imputer, "forecaster": forecaster,
                "bias_vs_truth": bias_vs_truth,
                "bias_vs_baseline": bias_vs_baseline,
                "variance": variance,
            })

        return pd.DataFrame(records)