import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

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
    assert drm1.progress_updated == function

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
    assert not DimReducMethod.is_valid_dimreduc_method(0)
    assert DimReducMethod.is_valid_dimreduc_method(1)
    assert DimReducMethod.is_valid_dimreduc_method(len(DimReducMethod.dim_reduc_methods))
    assert not DimReducMethod.is_valid_dimreduc_method(len(DimReducMethod.dim_reduc_methods) + 1)


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


def test_compute():  # ok rajouter test sur publish_progress
    X = generate_df_series_callable()[0]
    # drm = DimReducMethod(1, PCA, 2, X, default_parameters={'n_components': 2}, progress_updated= lambda x,y,z : k=max(k,z))
    drm = DimReducMethod(1, PCA, 2, X, default_parameters={'n_components': 2})
    a = drm.compute()
    assert drm.default_parameters == {'n_components': 2}
    # assert drm.


def test_scale_value_space():
    np.random.seed(10)
    X = pd.DataFrame(np.random.randint(0, 100, size=(6, 3)), columns=list('ABC'))
    y = X.sum(axis=1)
    drm = DimReducMethod(1, None, 2, X)
    a = drm.scale_value_space(X, y)
    expected = pd.DataFrame([[-0.048086, -0.153033, 0.032684],
                             [-0.000829, 0.350276, 0.216138],
                             [0.001658, -0.200644, 0.089618],
                             [-0.070471, 0.017004, -0.144444],
                             [-0.030676, -0.180239, -0.030576],
                             [0.148405, 0.166636, -0.163422]],
                            index=list(range(0, 6)), columns=list('ABC'))
    assert np.round(drm.scale_value_space(X, y)[::], 6).equals(expected)
