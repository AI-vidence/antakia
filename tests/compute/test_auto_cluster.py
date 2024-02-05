import numpy as np
import pandas as pd

from antakia.compute.auto_cluster.auto_cluster import auto_cluster_factory, AutoCluster


def test_auto_cluster():
    data = pd.DataFrame(np.random.random((200, 2)), columns=['col1', 'col2'])
    for method in auto_cluster_factory:
        cluster_regions = AutoCluster(data, lambda *args: None, method).compute(data.iloc[50:, :], data.iloc[50:, :])
        for mask in cluster_regions.get_masks():
            assert len(mask) == len(data)
        cluster_regions = AutoCluster(data, lambda *args: None, method).compute(data, data, n_clusters=3)
        assert len(cluster_regions) == 3
