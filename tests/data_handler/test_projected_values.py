import os
import numpy as np
import mock
import pandas as pd

from antakia.data_handler.projected_values import ProjectedValues
from tests.utils_fct import generate_df_series_callable

from antakia.compute.dim_reduction.dim_reduction import compute_projection, dim_reduc_factory


def test_init():
    X, y, tmp = generate_df_series_callable()
    pv = ProjectedValues(X, y)
    assert pv.X.equals(X)
    assert pv.y.equals(y)
    assert pv._projected_values == {}
    assert pv._kwargs == {}
    assert pv.current_proj == (int(os.environ.get('DEFAULT_VS_PROJECTION', 4)),
                               int(os.environ.get('DEFAULT_VS_DIMENSION', 2)))


@mock.patch('antakia.compute.dim_reduction.dim_reduction.compute_projection')
# @mock.patch('antakia.data_handler.projected_values.ProjectedValues.compute')
def test_set_parameters(cpt_proj): #not ok
    X, y, function = generate_df_series_callable()
    pv = ProjectedValues(X, y)
    cpt_proj.return_value = X_red = pd.DataFrame([[4, 7, 10],
                                                  [5, 8, 11],
                                                  [6, 9, 12]],
                                                 index=[1, 2, 3],
                                                 columns=['a', 'b', 'c'])

    pv.compute(1, 2, function)
    pv.set_parameters(1, 2, {(1, 2): {'current': {}, 'previous': {}}})
    assert pv._kwargs == {(1, 2): {'current': {}, 'previous': {}}}


def test_get_parameters(): #not ok
    X, y, tmp = generate_df_series_callable()
    pv = ProjectedValues(X, y)
    pv.build_default_parameters(1, 2)
    # assert pv.get_paramerters()


def test_build_default_parameters():
    X, y, tmp = generate_df_series_callable()
    pv = ProjectedValues(X, y)
    pv.build_default_parameters(1, 2)
    assert pv._kwargs == {(1, 2): {'current': {}, 'previous': {}}}

    pv1 = ProjectedValues(X, y)
    pv1.build_default_parameters(2, 3)
    assert pv1._kwargs == {(2, 3): {'current': {'learning_rate': 'auto', 'perplexity': 12},
                                    'previous': {'learning_rate': 'auto', 'perplexity': 12}}}


@mock.patch('antakia.compute.dim_reduction.dim_reduction.compute_projection')
def test_get_projection(cpt_proj): #not ok
    X_red = pd.DataFrame([[4, 7, 10],
                          [5, 8, 11],
                          [6, 9, 12]],
                         index=[1, 2, 3],
                         columns=['a', 'b', 'c'])
    X, y, tmp = generate_df_series_callable()
    pv = ProjectedValues(X, y)
    pv._projected_values = {(1, 2): X_red}
    np.testing.assert_array_equal(pv.get_projection(1, 2), X_red)

    cpt_proj.return_value = X_red
    pv1 = ProjectedValues(X, y)
    pv1._projected_values = {(1, 2): X_red}
    np.testing.assert_array_equal(pv1.get_projection(1, 2), X_red)


def test_is_present():  # faire set parameters avant
    X, y, tmp = generate_df_series_callable()
    pv = ProjectedValues(X, y)
    assert not pv.is_present(1, 2)

    pv.set_parameters(1, 2, {(1, 2): {'current': {}, 'previous': {}}})
    assert pv.is_present(1, 2)


@mock.patch('antakia.compute.dim_reduction.dim_reduction.compute_projection')
def test_compute(cpt_proj):  # probleme avec mock
    X, y, tmp = generate_df_series_callable()
    pv = ProjectedValues(X, y)

    X_red = pd.DataFrame([[4, 7, 10],
                          [5, 8, 11],
                          [6, 9, 12]],
                         index=[1, 2, 3],
                         columns=['a', 'b', 'c'])

    cpt_proj.return_value = X_red
    pv.compute(1, 2, lambda *args: None)
    a = 1
    # assert pv._projected_values[(1, 2)] == X_red
