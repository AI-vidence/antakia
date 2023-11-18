import pandas as pd

import warnings
from numba.core.errors import NumbaDeprecationWarning

from antakia.compute.auto_cluster.rule_cluster import rule_clusters
from antakia.compute.auto_cluster.shap_based_kmeans import shap_based_kmeans
from antakia.compute.auto_cluster.utils import reassign_clusters, _invert_list

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=UserWarning)


def auto_cluster(X: pd.DataFrame, shap_values: pd.DataFrame, n_clusters='auto') -> pd.Series:
    clusters = shap_based_kmeans(X, shap_values, n_clusters)
    clusters = pd.Series(clusters, index=X.index, name='cluster')
    # clusters = rule_clusters(X, clusters, cluster_fct)
    clusters = reassign_clusters(clusters)
    return clusters
