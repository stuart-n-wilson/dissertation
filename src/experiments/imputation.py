from dataclasses import asdict
import pandas as pd
from tqdm import tqdm
from itertools import product
from src.missingness import create_mask
from src.imputers import get_imputer
from src.evaluation import reconstruction_errors
from src.experiments.artifacts import OriginKey, ExperimentRun, ReconstructedStore


class ImputationExperiment:
    """Runs the (mechanism × proportion × seed × imputer) grid.

    Produces the per-run reconstruction table and fills the store with one
    reconstructed dataset per run.
    """

    def __init__(self, mechanisms, proportions, seeds, imputer_names):

        self.mechanisms = mechanisms
        self.proportions = proportions
        self.seeds = seeds
        self.imputer_names = imputer_names

    # def run(self, train: pd.DataFrame, store: ReconstructedStore) -> pd.DataFrame:

    #     runs = []

    #     for seed in self.seeds:

    #         for mechanism in self.mechanisms:

    #             for proportion in self.proportions:

    #                 mask = create_mask(train, mechanism, proportion, seed)
    #                 incomplete = train.mask(mask)

    #                 for name in self.imputer_names:
                        
    #                     # Inititate new model each time - prevent leakage.
    #                     imputer = get_imputer(name)
    #                     imputed = imputer.fit_transform(incomplete)
    #                     rmse, mae = reconstruction_errors(train, imputed, mask)
    #                     origin = OriginKey(mechanism, proportion, seed, name)

    #                     # Save reconstructed dataset under Origin key.
    #                     store.put(origin, imputed)
    #                     runs.append(ExperimentRun(
    #                         mechanism, proportion, seed, name, rmse, mae
    #                     ))

    #     return pd.DataFrame([asdict(r) for r in runs])

    def run(self, train: pd.DataFrame, store: ReconstructedStore) -> pd.DataFrame:

        runs = []

        # Flatten the grid so tqdm knows the total and can show % + ETA.
        combos = list(product(self.seeds, self.mechanisms, self.proportions, self.imputer_names))

        for seed, mechanism, proportion, name in tqdm(combos, desc="imputation"):

            mask = create_mask(train, mechanism, proportion, seed)
            incomplete = train.mask(mask)

            # Inititate new model each time - prevent leakage.
            imputer = get_imputer(name)
            imputed = imputer.fit_transform(incomplete)
            rmse, mae = reconstruction_errors(train, imputed, mask)
            origin = OriginKey(mechanism, proportion, seed, name)

            # Save reconstructed dataset under Origin key.
            store.put(origin, imputed)
            runs.append(ExperimentRun(
                mechanism, proportion, seed, name, rmse, mae
            ))

        return pd.DataFrame([asdict(r) for r in runs])