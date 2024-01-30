import os
import numpy as np
import mock
import pandas as pd

from antakia.data_handler.projected_values import ProjectedValues

from antakia.compute.dim_reduction.dim_reduction import compute_projection, dim_reduc_factory


def test_init():  # ok
    pv = ProjectedValues(np.empty([2, 2]), np.empty(2))
    assert np.testing.assert_array_equal(pv.X, np.empty([2, 2]))
    assert np.testing.assert_array_equal(pv.y, np.empty(2))
    assert pv._projected_values == {}
    assert pv._kwargs == {}
    assert pv.current_proj == (int(os.environ.get('DEFAULT_VS_PROJECTION', 4)),
                               int(os.environ.get('DEFAULT_VS_DIMENSION', 2)))


def test_set_parameters():
    pass


def test_get_parameters():
    pv = ProjectedValues(np.empty([2, 2]), np.empty(2))
    pv.build_default_parameters(1, 2)
    # assert pv.get_paramerters()


def test_build_default_parameters():  # ok
    pv = ProjectedValues(np.empty([2, 2]), np.empty(2))
    pv.build_default_parameters(1, 2)
    assert pv._kwargs == {(1, 2): {'current': {}, 'previous': {}}}

    pv1 = ProjectedValues(np.empty([2, 2]), np.empty(2))
    pv1.build_default_parameters(2, 3)
    assert pv1._kwargs == {(2, 3): {'current': {'learning_rate': 'auto', 'perplexity': 12},
                                    'previous': {'learning_rate': 'auto', 'perplexity': 12}}}

@mock.patch('antakia.compute.dim_reduction.dim_reduction.compute_projection')
def test_get_projection(cpt_proj):
    X_red = pd.DataFrame([[4, 7, 10],
                          [5, 8, 11],
                          [6, 9, 12]],
                         index=[1, 2, 3],
                         columns=['a', 'b', 'c'])
    pv = ProjectedValues(np.empty([2, 2]), np.empty(2))
    pv._projected_values = {(1, 2): X_red}
    np.testing.assert_array_equal(pv.get_projection(1, 2), X_red)


    cpt_proj.return_value = X_red
    pv1 = ProjectedValues(np.empty([2, 2]), np.empty(2))
    pv._projected_values = {(1, 2): X_red}
    np.testing.assert_array_equal(pv.get_projection(1, 2), X_red)


def test_is_present():  # faire set parameters avant
    pv = ProjectedValues(np.empty([2, 2]), np.empty(2))
    assert not pv.is_present(1, 2)

    pv.set_parameters(1, 2, {(1, 2): {'current': {}, 'previous': {}}})
    assert pv.is_present(1, 2)


@mock.patch('antakia.compute.dim_reduction.dim_reduction.compute_projection')
def test_compute(cpt_proj):  # probleme avec mock
    pv = ProjectedValues(np.empty([2, 2]), np.empty(2))

    X_red = pd.DataFrame([[4, 7, 10],
                          [5, 8, 11],
                          [6, 9, 12]],
                         index=[1, 2, 3],
                         columns=['a', 'b', 'c'])

    cpt_proj.return_value = X_red
    pv.compute(1, 2, lambda *args: None)
    a = 1
    # assert pv._projected_values[(1, 2)] == X_red
