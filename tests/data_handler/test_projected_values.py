import os
import numpy as np

from antakia.data_handler.projected_values import ProjectedValues

from antakia.compute.dim_reduction.dim_reduction import compute_projection, dim_reduc_factory


def test_init():
    pv = ProjectedValues(np.empty([2, 2]), np.empty(2))
    assert np.testing.assert_array_equal(pv.X, np.empty([2, 2]))
    assert np.testing.assert_array_equal(pv.y, np.empty(2))
    assert pv._projected_values == {}
    assert pv._kwargs == {}
    assert pv.current_proj == (int(os.environ.get('DEFAULT_VS_PROJECTION', 4)),
                               int(os.environ.get('DEFAULT_VS_DIMENSION', 2)))

def test_set_parameters():
    pass

def test_build_default_parameters():
    pv = ProjectedValues(np.empty([2, 2]), np.empty(2))
    pv.build_default_parameters(1,2)

    print((pv._kwargs))
    print((pv._kwargs.keys()))
    print((pv._kwargs.values()))
    # dict = {(1,2):'current': {}, 'previous': {}}

