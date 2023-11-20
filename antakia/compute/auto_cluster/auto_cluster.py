from typing import Callable

import pandas as pd

import warnings
from numba.core.errors import NumbaDeprecationWarning

from antakia.compute.auto_cluster.shap_based_kmeans import ShapBasedKmeans
from antakia.compute.auto_cluster.utils import reassign_clusters, _invert_list
from antakia.data import LongTask

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=UserWarning)


class AutoCluster(LongTask):

    def __init__(self, X: pd.DataFrame, progress_updated: Callable):
        super().__init__(X, progress_updated)
        self.cluster_algo = ShapBasedKmeans(X, progress_updated)

    def compute(self, shap_values: pd.DataFrame, n_clusters='auto') -> pd.Series:
        clusters = self.cluster_algo.process(shap_values, n_clusters)
        clusters = pd.Series(clusters, index=self.X.index, name='cluster')
        # clusters = rule_clusters(X, clusters, cluster_fct)
        clusters = reassign_clusters(clusters)
        return clusters
