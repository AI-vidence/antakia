import math

import mvlearn
import numpy as np
import pandas as pd
from tomaster import tomato
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from antakia.utils.long_task import LongTask


class ShapBasedTomato(LongTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, X, shap_values: pd.DataFrame, n_clusters='auto') -> list:
        x_scaled = self.scale(X, shap_values)
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
        if n_clusters == 'auto':
            clusters = tomato(points=X.values, k=20)
        else:
            clusters = tomato(points=X.values, n_clusters=n_clusters, k=20)
        return clusters


def cluster2(X, shap, n_clusters):
    m_kmeans = mvlearn.cluster.MultiviewKMeans(n_clusters=n_clusters, random_state=9)
    clusters = m_kmeans.fit_predict([X, shap])
    return clusters
