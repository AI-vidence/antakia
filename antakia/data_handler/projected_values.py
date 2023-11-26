import pandas as pd

from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod


class ProjectedValues:
    """
    A class to hold a Dataframe X, and its projected values (with various dimensionality reduction methods).

    Instance attributes
    ------------------
    X  : pandas.Dataframe
        The dataframe to be used by AntakIA
    _X_proj : a dict with four values :
        - key DimReducMethod.PCA : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PCA-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PCA-projected X values
        - key DimReducMethod.TSNE : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D TSNE-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D TSNE-projected X values
        - key DimReducMethod.UMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D UMAP-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D UMAP-projected X values
        - key DimReducMethod.PaCMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PaCMAP-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PaCMAP-projected X values
    """

    def __init__(self, X: pd.DataFrame):
        """
        Constructor of the class ProjectedValues.

        Parameters
        ----------
        X : pandas.Dataframe
            The dataframe provided by the user.
        """
        self.X = X
        self._X_proj = {}

    def __str__(self):
        text = f"X's shape : {self.X.shape}"
        for proj_method in DimReducMethod.dimreduc_methods_as_list():
            for dim in [2, 3]:
                if self.is_available(proj_method, dim):
                    text += f"\n{DimReducMethod.dimreduc_method_as_str(proj_method)}/{DimReducMethod.dimension_as_str(dim)} : {self.get_proj_values(proj_method, dim).shape}"
        return text

    def get_length(self) -> int:
        return self.X.shape[0]

    def get_proj_values(self, dimreduc_method: int, dimension: int) -> pd.DataFrame:
        """Returns de projected X values using a dimensionality reduction method and target dimension (2 or 3)

        Args:
            dimreduc_method (int): the dimensionality reduction method
            dimension (int, optional): Defaults to DimReducMethod.DIM_TWO.

        Returns:
            pd.DataFrame: the projected X values. May be None
        """
        if not DimReducMethod.is_valid_dimreduc_method(dimreduc_method):
            raise ValueError("Bad dimensionality reduction method")
        if not DimReducMethod.is_valid_dim_number(dimension):
            raise ValueError(dimension, " is a bad dimension number")

        if (
                dimreduc_method not in self._X_proj
                or dimension not in self._X_proj[dimreduc_method]
                or self._X_proj[dimreduc_method][dimension] is None
        ):
            # ProjectedValues is a "datastore", its role is not trigger projection computations
            return None
        else:
            return self._X_proj[dimreduc_method][dimension]

    def is_available(self, dimreduc_method: int, dimension: int) -> bool:
        return self.get_proj_values(dimreduc_method, dimension) is not None

    def set_proj_values(
            self, dimreduc_method: int, dimension: int, values: pd.DataFrame
    ):
        """Set X_proj alues for this dimensionality reduction and  dimension."""

        # TODO we may want to check values.shape and raise value error if it does not match
        if not DimReducMethod.is_valid_dimreduc_method(dimreduc_method):
            raise ValueError("Bad dimensionality reduction method")
        if not DimReducMethod.is_valid_dim_number(dimension):
            raise ValueError("Bad dimension number")

        if dimreduc_method not in self._X_proj:
            self._X_proj[
                dimreduc_method
            ] = {}  # We create a new dict for this dimreduc_method
        if dimension not in self._X_proj[dimreduc_method]:
            self._X_proj[dimreduc_method][
                dimension
            ] = {}  # We create a new dict for this dimension

        self._X_proj[dimreduc_method][dimension] = values

    def get_shape(self):
        """Returns the shape of the used dataset"""
        return self.X.shape
