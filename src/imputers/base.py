import pandas as pd

class BaseImputer:
    """Base class for all imputers. Subclasses must declare a class-level
    `name: str` attribute and implement `transform`.
    """
    
    name: str

    def fit(self, incomplete: pd.DataFrame) -> "BaseImputer":
        """Fit the imputer to the incomplete data.

        Args:
            incomplete: DataFrame containing NaN values to learn from.

        Returns:
            self
        """
        return self

    def transform(self, incomplete: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in the DataFrame.

        Args:
            incomplete: DataFrame containing NaN values to fill.

        Returns:
            DataFrame of the same shape, index, and columns with NaNs filled.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def fit_transform(self, incomplete: pd.DataFrame) -> pd.DataFrame:
        """Fit to and impute the incomplete DataFrame.

        Args:
            incomplete: DataFrame containing NaN values.

        Returns:
            DataFrame of the same shape, index, and columns with NaNs filled.
        """
        self.fit(incomplete)
        return self.transform(incomplete)