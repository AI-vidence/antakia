import pytest

from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from tests.utils_fct import generate_df_series_callable


def test_init():
    with pytest.raises(ValueError):
        DimReducMethod(1, None, 2, None)

