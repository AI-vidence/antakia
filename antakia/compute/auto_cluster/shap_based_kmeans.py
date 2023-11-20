import mvlearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



def shap_based_kmeans(X: pd.DataFrame, shap_values: pd.DataFrame, n_clusters='auto') -> list:
    x_scaled = scale(X, shap_values)
    clusters = cluster(x_scaled, shap_values, n_clusters=n_clusters)
    return clusters


def cluster(X, shap, n_clusters):
    X2 = pd.concat([X, shap], axis=1)
    clusters = None
    score = 0
    if n_clusters=='auto':
        for k in range(2,30):
            k_clusters = n_cluster(X2, k)
            k_score = score_cluster(X2, k_clusters)
            if k_score > score:
                score = k_score
                clusters = k_clusters
    else:
        k_clusters = n_cluster(X2, n_clusters)
        k_score = score_cluster(X2, k_clusters)
        if k_score > score:
            score = k_score
            clusters = k_clusters
    return clusters


def scale(X, shap_values):
    return (X - X.mean()) / X.std() * np.abs(shap_values).mean()


def score_cluster(X, clusters):
    return silhouette_score(X, clusters)  # adapté pour kmeans


def n_cluster(X, n_clusters):
    km = KMeans(n_clusters, n_init=10)
    clusters = km.fit_predict(X)
    return clusters


def cluster2(X, shap, n_clusters):
    m_kmeans = mvlearn.cluster.MultiviewKMeans(n_clusters=n_clusters, random_state=9)
    clusters = m_kmeans.fit_predict([X, shap])
    return clusters
