from typing import Callable

import pandas as pd

import warnings
from numba.core.errors import NumbaDeprecationWarning

from antakia.compute.auto_cluster.shap_based_kmeans import ShapBasedKmeans
from antakia.compute.auto_cluster.shap_based_hdbscan import ShapBasedHdbscan
from antakia.compute.auto_cluster.shap_based_tomaster import ShapBasedTomato
from antakia.compute.auto_cluster.utils import reassign_clusters, _invert_list
from antakia.utils.long_task import LongTask

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=UserWarning)

auto_cluster_factory = {
    'kmeans': ShapBasedKmeans,
    'hdbscan': ShapBasedHdbscan,
    'tomato': ShapBasedTomato,
}


class AutoCluster(LongTask):

    def __init__(self, X: pd.DataFrame, progress_updated: Callable, method='hdbscan'):
        super().__init__(X, progress_updated)
        assert len(X) > 50
        assert auto_cluster_factory[method]
        self.cluster_algo = auto_cluster_factory[method](X, progress_updated)

    def compute(self, shap_values: pd.DataFrame, n_clusters='auto') -> pd.Series:
        self.publish_progress(0)
        clusters = self.cluster_algo.compute(shap_values, n_clusters)
        clusters = pd.Series(clusters, index=self.X.index, name='cluster')
        # clusters = rule_clusters(X, clusters, cluster_fct)
        clusters = reassign_clusters(clusters)
        self.publish_progress(100)
        return clusters


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('../../../data/california_housing.csv').drop(['Unnamed: 0'], axis=1)
    # Remove outliers:
    df = df.loc[df['Population'] < 10000]
    df = df.loc[df['AveOccup'] < 6]
    df = df.loc[df['AveBedrms'] < 1.5]
    df = df.loc[df['HouseAge'] < 50]

    # # Only San Francisco :
    df = df.loc[(df['Latitude'] < 38.07) & (df['Latitude'] > 37.2)]
    df = df.loc[(df['Longitude'] > -122.5) & (df['Longitude'] < -121.75)]
    X = df.iloc[:, 0:8]  # the dataset
    y = df.iloc[:, 9]  # the target variable
    shapValues = df.iloc[:, [10, 11, 12, 13, 14, 15, 16, 17]]  # the SHAP values`
    shapValues.columns = [col.replace('_shap', '') for col in shapValues.columns]

    AutoCluster(X, lambda x, y, z: None).compute(shap_values=shapValues)
