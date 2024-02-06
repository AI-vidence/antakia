from collections import namedtuple
import pandas as pd

from antakia import config
from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.compute.dim_reduction.dim_reduction import compute_projection, dim_reduc_factory

Proj = namedtuple('Proj', ['reduction_method', 'dimension'])


class ProjectedValues:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self._projected_values = {}
        self._parameters = {}
        self.current_proj = Proj(DimReducMethod.default_projection_as_int(), config.DEFAULT_DIMENSION)

    def set_parameters(self, projection_method: int, dimension: int, parameters: dict):
        """
        set new parameters for a (projection method, dimension)
        remove previously computed projected value
        Parameters
        ----------
        projection_method : projection method value
        dimension: dimension
        parameters: new parameters

        Returns
        -------

        """
        assert projection_method in DimReducMethod.dimreduc_methods_as_list()
        assert dimension in [2, 3]

        if self._parameters.get(Proj(projection_method, dimension)) is None:
            self.build_default_parameters(projection_method, dimension)

        self._parameters[Proj(projection_method, dimension)]['previous'] = \
            self._parameters[Proj(projection_method, dimension)][
                'current'].copy()
        self._parameters[Proj(projection_method, dimension)]['current'].update(parameters)
        del self._projected_values[Proj(projection_method, dimension)]

    def get_parameters(self, projection_method, dimension):
        """
        get the value of the parameters for a (projection method, dimension)
        build it to default if needed
        Parameters
        ----------
        projection_method
        dimension

        Returns
        -------

        """
        if self._parameters.get(Proj(projection_method, dimension)) is None:
            self.build_default_parameters(projection_method, dimension)
        return self._parameters.get(Proj(projection_method, dimension))

    def build_default_parameters(self, projection_method, dimension):
        """
        build default parameters from DimReductionMethod.parameters()
        Parameters
        ----------
        projection_method
        dimension

        Returns
        -------

        """
        current = {}
        dim_reduc_parameters = dim_reduc_factory[projection_method].parameters()
        for param, info in dim_reduc_parameters.items():
            current[param] = info['default']
        self._parameters[Proj(projection_method, dimension)] = {
            'current': current,
            'previous': current.copy()
        }

    def get_projection(self, projection_method: int, dimension: int, progress_callback: callable = None,
                       set_current=True):
        """
        get a projection value
        computes it if necessary
        Parameters
        ----------
        projection_method
        dimension
        progress_callback

        Returns
        -------

        """
        if not self.is_present(projection_method, dimension):
            self.compute(projection_method, dimension, progress_callback)
        if set_current:
            self.current_proj = Proj(projection_method, dimension)
        return self._projected_values[Proj(projection_method, dimension)]

    def is_present(self, projection_method: int, dimension: int) -> bool:
        """
        tests if the projection is already computed
        Parameters
        ----------
        projection_method
        dimension

        Returns
        -------

        """
        return self._projected_values.get(Proj(projection_method, dimension)) is not None

    def compute(self, projection_method: int, dimension: int, progress_callback: callable):
        """
        computes a projection and store it
        Parameters
        ----------
        projection_method
        dimension
        progress_callback

        Returns
        -------

        """
        projected_values = compute_projection(
            self.X,
            self.y,
            projection_method,
            dimension,
            progress_callback,
            **self.get_parameters(projection_method, dimension)['current']
        )
        self._projected_values[Proj(projection_method, dimension)] = projected_values
