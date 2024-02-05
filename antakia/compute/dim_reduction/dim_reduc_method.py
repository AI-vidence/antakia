import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from antakia import config
from antakia.utils.long_task import LongTask


class DimReducMethod(LongTask):
    """
    Class that allows to reduce the dimensionality of the data.

    Attributes
    ----------
    dimreduc_method : int, can be PCA, TSNE etc.
    dimension : int
        Dimension reduction methods require a dimension parameter
        We store it in the abstract class
    """

    # Class attributes methods
    dim_reduc_methods = ['PCA', 'UMAP', 'PaCMAP']
    dimreduc_method = -1

    allowed_kwargs = []

    def __init__(
            self,
            dimreduc_method: int,
            dimreduc_model: type[TransformerMixin],
            dimension: int,
            X: pd.DataFrame,
            default_parameters: dict = None,
            progress_updated: callable = None,
    ):
        """
        Constructor for the DimReducMethod class.

        Parameters
        ----------
        dimreduc_method : int
            Dimension reduction methods among DimReducMethod.PCA, DimReducMethod.TSNE, DimReducMethod.UMAP or DimReducMethod.PaCMAP
            We store it here (not in implementation class)
        dimension : int
            Target dimension. Can be DIM_TWO or DIM_THREE
            We store it here (not in implementation class)
        X : pd.DataFrame
            Stored in LongTask instance
        progress_updated : callable
            Stored in LongTask instance
        """
        if not DimReducMethod.is_valid_dimreduc_method(dimreduc_method):
            raise ValueError(
                dimreduc_method, " is a bad dimensionality reduction method"
            )
        if not DimReducMethod.is_valid_dim_number(dimension):
            raise ValueError(dimension, " is a bad dimension number")

        self.dimreduc_method = dimreduc_method
        self.default_parameters = default_parameters
        self.dimension = dimension
        self.dimreduc_model = dimreduc_model
        # IMPORTANT : we set the topic as for ex 'PCA/2' or 't-SNE/3' -> subscribers have to follow this scheme
        LongTask.__init__(self, X, progress_updated)

    @classmethod
    def dimreduc_method_as_str(cls, method: int) -> str:
        if method is None:
            return None
        elif 0 < method <= len(cls.dim_reduc_methods):
            return cls.dim_reduc_methods[method - 1]
        else:
            raise ValueError(f"{method} is an invalid dimensionality reduction method")

    @classmethod
    def dimreduc_method_as_int(cls, method: str) -> int:
        if method is None:
            return None
        try:
            i = cls.dim_reduc_methods.index(method) + 1
            return i
        except ValueError:
            raise ValueError(f"{method} is an invalid dimensionality reduction method")

    @classmethod
    def dimreduc_methods_as_list(cls) -> list[int]:
        return list(map(lambda x: x + 1, range(len(cls.dim_reduc_methods))))

    @classmethod
    def dimreduc_methods_as_str_list(cls) -> list[str]:
        return cls.dim_reduc_methods.copy()

    @staticmethod
    def dimension_as_str(dim) -> str:
        if dim == 2:
            return "2D"
        elif dim == 3:
            return "3D"
        else:
            raise ValueError(f"{dim}, is a bad dimension")

    @classmethod
    def is_valid_dimreduc_method(cls, method: int) -> bool:
        """
        Returns True if it is a valid dimensionality reduction method.
        """
        return 0 <= method - 1 < len(cls.dim_reduc_methods)

    @staticmethod
    def is_valid_dim_number(dim: int) -> bool:
        """
        Returns True if dim is a valid dimension number.
        """
        return dim in [2, 3]

    def get_dimension(self) -> int:
        return self.dimension

    @classmethod
    def parameters(cls) -> dict[str, dict[str, any]]:
        return {}

    def compute(self, **kwargs) -> pd.DataFrame:
        self.publish_progress(0)
        kwargs['n_components'] = self.get_dimension()
        param = self.default_parameters.copy()
        param.update(kwargs)

        dim_red_model = self.dimreduc_model(**param)
        if hasattr(dim_red_model, 'fit_transform'):
            X_red = dim_red_model.fit_transform(self.X)
        else:
            dim_red_model.fit(self.X)
            X_red = dim_red_model.transform(self.X)
        X_red = pd.DataFrame(X_red)

        self.publish_progress(100)
        return X_red

    @classmethod
    def scale_value_space(cls, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Scale the values in X so that it's reduced and centered and weighted with mi
        """
        std = X.std()
        std[std == 0] = 1
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(X, y)
        return (X - X.mean()) / std * mi

    @classmethod
    def default_projection_as_int(cls) -> int:
        return DimReducMethod.dimreduc_method_as_int(config.DEFAULT_PROJECTION)

    @classmethod
    def default_projection_as_str(cls) -> str:
        return config.DEFAULT_PROJECTION
