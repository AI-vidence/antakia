import numpy as np
import pandas as pd

from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.compute.dim_reduction.dim_reduction import compute_projection


def test_dim_reduction():
    X = pd.DataFrame(np.random.random((100, 5)), index=np.random.choice(np.random.randint(1000, size=200), size=100))
    y = X.sum(axis=1)

    for dim_method in DimReducMethod.dimreduc_methods_as_list():
        print(DimReducMethod.dimreduc_method_as_str(dim_method))
        compute_projection(X, y, dim_method, 2, lambda x, y, z: None)
        compute_projection(X, y, dim_method, 3, lambda x, y, z: None)
