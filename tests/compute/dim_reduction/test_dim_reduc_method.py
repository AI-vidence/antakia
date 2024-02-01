import numpy as np
import pandas as pd
import pytest

from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from tests.utils_fct import generate_df_series_callable


def test_init():
    X, function = generate_df_series_callable()[0:3:2]
    drm = DimReducMethod(1, None, 2, X)
    assert drm.dimreduc_method == 1
    assert drm.default_parameters is None
    assert drm.dimension == 2
    assert drm.dimreduc_model is None
    assert drm.X.equals(X)
    assert drm.progress_updated is None

    drm1 = DimReducMethod(2, None, 2, X, progress_updated=function)
    assert drm1.dimreduc_method == 2
    assert drm1.default_parameters is None
    assert drm1.dimension == 2
    assert drm1.dimreduc_model is None
    assert drm.progress_updated == function()

    with pytest.raises(ValueError):
        DimReducMethod(6, None, 2, X, progress_updated=function)

    with pytest.raises(ValueError):
        DimReducMethod(2, None, 4, X, progress_updated=function)


def test_dimreduc_method_as_str():
    X = generate_df_series_callable()[0]
    drm = DimReducMethod(1, None, 2, X)
    assert drm.dimreduc_method_as_str(None) is None
    assert drm.dimreduc_method_as_str(1) == drm.dim_reduc_methods[0]
    with pytest.raises(ValueError):
        drm.dimreduc_method_as_str(0)


def test_dimreduc_method_as_int():
    X = generate_df_series_callable()[0]
    drm = DimReducMethod(1, None, 2, X)
    assert drm.dimreduc_method_as_int(None) is None
    assert drm.dimreduc_method_as_int(drm.dim_reduc_methods[0]) == 1
    with pytest.raises(ValueError):
        drm.dimreduc_method_as_int('Method')


def test_dimreduc_methods_as_list():
    X = generate_df_series_callable()[0]
    drm = DimReducMethod(1, None, 2, X)
    assert drm.dimreduc_methods_as_list() == list(range(1, len(drm.dim_reduc_methods) + 1))


def test_dimreduc_methods_as_str_list():
    X = generate_df_series_callable()[0]
    drm = DimReducMethod(1, None, 2, X)
    assert drm.dimreduc_methods_as_str_list() == drm.dim_reduc_methods


def test_dimension_as_str():
    X = generate_df_series_callable()[0]
    drm = DimReducMethod(1, None, 2, X)
    assert drm.dimension_as_str(2) == '2D'
    assert drm.dimension_as_str(3) == '3D'
    with pytest.raises(ValueError):
        drm.dimension_as_str(1)


def test_is_valid_dimreduc_method():
    X = generate_df_series_callable()[0]
    drm = DimReducMethod(1, None, 2, X)
    assert not drm.is_valid_dimreduc_method(0)
    assert drm.is_valid_dimreduc_method(1)
    assert drm.is_valid_dimreduc_method(len(drm.dim_reduc_methods))
    assert not drm.is_valid_dimreduc_method(len(drm.dim_reduc_methods) + 1)


def test_is_valid_dim_number():
    X = generate_df_series_callable()[0]
    drm = DimReducMethod(1, None, 2, X)
    assert not drm.is_valid_dim_number(1)
    assert drm.is_valid_dim_number(2)
    assert drm.is_valid_dim_number(3)


def test_get_dimension():
    X = generate_df_series_callable()[0]
    drm = DimReducMethod(1, None, 2, X)
    assert drm.get_dimension() == 2
    drm = DimReducMethod(1, None, 3, X)
    assert drm.get_dimension() == 3


def test_parameters():
    X = generate_df_series_callable()[0]
    drm = DimReducMethod(1, None, 2, X)
    assert drm.parameters() == {}


def test_compute():
    X = generate_df_series_callable()[0]
    drm = DimReducMethod(1, None, 2, X)



def test_scale_value_space():  # not ok
    np.random.seed(10)
    X = pd.DataFrame(np.random.randint(0, 100, size=(6, 3)), columns=list('ABC'))
    y = X.sum(axis=1)
    drm = DimReducMethod(1, None, 2, X)
    a = drm.scale_value_space(X, y)
    expected = pd.DataFrame([[-0.034347, -0, 0.006285],
                             [-0.000592, 0, 0.041565],
                             [0.001184, -0, 0.017234],
                             [-0.050337, 0, -0.027778],
                             [-0.021911, -0, -0.005880],
                             [0.106003, 0, -0.031427]],
                            index=list(range(0, 6)), columns=list('ABC'))
    test = (a.equals(expected))
    b = 1
    assert drm.scale_value_space(X, y).equals(expected)
