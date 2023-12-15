from typing import Callable

import pandas as pd

import warnings
from numba.core.errors import NumbaDeprecationWarning

from antakia.compute.auto_cluster.shap_based_kmeans import ShapBasedKmeans
from antakia.compute.auto_cluster.shap_based_hdbscan import ShapBasedHdbscan
from antakia.compute.auto_cluster.shap_based_tomaster import ShapBasedTomato
from antakia.compute.skope_rule.skope_rule import skope_rules
from antakia.data_handler.region import RegionSet
from antakia.utils.long_task import LongTask
from antakia.utils.variable import DataVariables

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
        assert auto_cluster_factory[method]
        self.cluster_algo = auto_cluster_factory[method](X, progress_updated)

    def clusters_to_rules(
            self,
            clusters: pd.Series,
            variables: DataVariables = None,
            precision: float = 0.7,
            recall: float = 0.7,
            random_state=42
    ) -> RegionSet:

        new_clusters = clusters.unique()
        extended_clusters = clusters.reindex(self.X.index).fillna(-1)
        region_set = RegionSet(self.X)

        for cluster in new_clusters:
            if cluster != -1:
                cluster_mask = extended_clusters == cluster
                rules_list, _ = skope_rules(cluster_mask, self.X, variables, precision, recall, random_state)
                if len(rules_list) > 0:
                    r = region_set.add_region(rules=rules_list, auto_cluster=True)
                else:
                    region_set.add_region(mask=cluster_mask, auto_cluster=True)
        return region_set

    def compute(
            self,
            X: pd.DataFrame,
            shap_values: pd.DataFrame,
            n_clusters: int | str = 'auto',
            **kwargs
    ) -> RegionSet:
        assert len(X) > 50
        self.publish_progress(0)
        clusters = self.cluster_algo.compute(X, shap_values, n_clusters)
        clusters = pd.Series(clusters, index=X.index, name='cluster')
        self.publish_progress(90)
        new_regions = self.clusters_to_rules(clusters, **kwargs)
        self.publish_progress(100)
        return new_regions


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
