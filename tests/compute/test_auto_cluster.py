import numpy as np
import pandas as pd

from antakia.compute.auto_cluster.auto_cluster import auto_cluster_factory, AutoCluster


def test_auto_cluster():
    data = pd.DataFrame(np.random.random((100, 2)))
    for method in auto_cluster_factory:
        clusters = AutoCluster(data, lambda x, y, z: None, method).compute(data)
        assert (clusters.index == data.index).all()
        assert clusters.shape == (len(data),)
        AutoCluster(data, lambda x, y, z: None, method).compute(data, n_clusters=3)
