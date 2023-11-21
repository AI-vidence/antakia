import math

import mvlearn
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN,AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from antakia.data import LongTask


class ShapBasedHdbscan(LongTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, shap_values: pd.DataFrame, n_clusters='auto') -> list:
        x_scaled = self.scale(self.X, shap_values)
        clusters = self.cluster(x_scaled, shap_values, n_clusters=n_clusters)
        return clusters

    def cluster(self, x_scaled, shap, n_clusters):
        X2 = pd.concat([x_scaled, shap], axis=1)
        clusters = self.n_cluster(X2, n_clusters)
        score = self.score_cluster(X2, clusters)
        return clusters

    @staticmethod
    def scale(X, shap_values):
        return (X - X.mean()) / X.std() * np.abs(shap_values).mean()

    @staticmethod
    def score_cluster(X, clusters):
        return calinski_harabasz_score(X, clusters)  # adapté pour kmeans

    @staticmethod
    def n_cluster(X, n_clusters):
        if n_clusters=='auto':
            hdbscan = HDBSCAN(min_cluster_size=int(math.sqrt(len(X))))
            clusters = hdbscan.fit_predict(X)
        else:
            aggc = AgglomerativeClustering(n_clusters=n_clusters)
            clusters = aggc.fit_predict(X)
        return clusters


def cluster2(X, shap, n_clusters):
    m_kmeans = mvlearn.cluster.MultiviewKMeans(n_clusters=n_clusters, random_state=9)
    clusters = m_kmeans.fit_predict([X, shap])
    return clusters
