import pandas as pd

from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.compute.dim_reduction.dim_reduction import compute_projection


class ProjectedValues:
    def __init__(self, X, y, callback):
        self.X = X
        self.y = y
        self.callback = callback
        self._projected_values = {}
        self._kwargs = {}

    def set_parameters(self, projection_method, dimension, kwargs):
        assert projection_method in DimReducMethod.dimreduc_methods_as_list()
        assert dimension in [2, 3]

        if self._kwargs.get((projection_method, dimension)) is None:
            self._kwargs[(projection_method, dimension)] = {
                'current': {},
                'previous': {}
            }

        self._kwargs[(projection_method, dimension)]['previous'] = self._kwargs[(projection_method, dimension)][
            'current'].copy()
        self._kwargs[(projection_method, dimension)]['current'].update(kwargs)
        del self._projected_values[(projection_method, dimension)]

    def get_projection(self, projection_method, dimension, callback=None):
        if self._projected_values.get((projection_method, dimension)) is None:
            self.compute(projection_method, dimension, callback)
        return self._projected_values[(projection_method, dimension)]

    def compute(self, projection_method, dimension, callback=None):
        if callback is None:
            callback = self.callback
        projected_values = compute_projection(
            self.X,
            self.y,
            projection_method,
            dimension,
            callback,
            **self._kwargs.get((projection_method, dimension), {
                'current': {},
                'previous': {}
            })['current']
        )
        self._projected_values[(projection_method, dimension)] = projected_values
