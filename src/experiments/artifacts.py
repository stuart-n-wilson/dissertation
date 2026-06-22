from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class OriginKey:
    """What uniquely identifies one run: which mechanism and proportion, and which imputer."""
    mechanism: str
    proportion: int
    seed: int
    imputer: str

@dataclass
class ExperimentRun:
    """One finished run's reconstruction result."""
    mechanism: str
    proportion: int
    seed: int
    imputer: str
    reconstruction_rmse: float
    reconstruction_mae: float

class ReconstructedStore:
    """Reads/writes reconstructed datasets, keyed by OriginKey."""
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _filename(self, origin: OriginKey) -> str:
        return (f"{origin.mechanism}__p{origin.proportion}"
                f"__s{origin.seed:02d}__{origin.imputer}.parquet")

    def put(self, origin: OriginKey, df: pd.DataFrame) -> None:
        df.to_parquet(self.root / self._filename(origin))   
        
    def get(self, origin: OriginKey) -> pd.DataFrame:
        return pd.read_parquet(self.root / self._filename(origin))

    def exists(self, origin: OriginKey) -> bool:
        return (self.root / self._filename(origin)).exists()

    def keys(self) -> list[OriginKey]:
        keys = []
        for path in self.root.glob("*.parquet"):
            mech, prop, seed, imp = path.stem.split("__")
            keys.append(OriginKey(mech, int(prop[1:]), int(seed[1:]), imp))
        return keys