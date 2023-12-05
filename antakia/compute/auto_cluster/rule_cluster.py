import warnings
from typing import Callable

import numpy as np
import pandas as pd
from skrules import SkopeRules

from antakia.compute.auto_cluster.utils import reassign_clusters, _reset_list, _invert_list


def rule_clusters(X, clusters, cluster_fct: Callable):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clusters = np.array(clusters)
        clusters = clusters - clusters.min()
        n_clusters = len(np.unique(clusters))
        if clusters.max() >= n_clusters:
            clusters = reassign_clusters(clusters, n_clusters)
        X_train = pd.DataFrame(X.copy())
        recall_min = 0.8
        precision_min = 0.8
        max_cluster = n_clusters
        for i in range(n_clusters):
            mask = clusters == i
            y_train = mask.astype(int)
            skope_rules_clf = SkopeRules(
                feature_names=X_train.columns,
                random_state=42,
                n_estimators=5,
                recall_min=recall_min,
                precision_min=precision_min,
                max_depth_duplication=0,
                max_samples=1.0,
                max_depth=3,
            )
            skope_rules_clf.fit(X_train, y_train)
            if len(skope_rules_clf.rules_) == 0 and mask.sum() > 50:
                sub_clusters = np.array(cluster_fct(X[mask])) + max_cluster
                clusters[mask] = sub_clusters
            a_list = _reset_list(a_list)
        a_list = list(np.array(a_list) - min(a_list))
        return _invert_list(a_list, max(a_list) + 1), a_list


def _find_best_k(X: pd.DataFrame, indices: list, recall_min: float, precision_min: float) -> int:
    new_X = X.iloc[indices]
    ind_f = 2
    for i in range(2, 9):
        # kmeans = sklearn.cluster.KMeans(n_clusters=i, random_state=9, n_init="auto")
        agglo = sklearn.cluster.AgglomerativeClustering(n_clusters=i)
        agglo.fit(new_X)
        labels = agglo.labels_
        a = True
        for j in range(max(labels) + 1):
            y = []
            for k in range(len(X)):
                if k in indices and labels[indices.index(k)] == j:
                    y.append(1)
                else:
                    y.append(0)
            skope_rules_clf = SkopeRules(
                feature_names=new_X.columns,
                random_state=42,
                n_estimators=5,
                recall_min=recall_min,
                precision_min=precision_min,
                max_depth_duplication=0,
                max_samples=1.0,
                max_depth=3,
            )
            skope_rules_clf.fit(X, y)
            if len(skope_rules_clf.rules_) == 0:
                ind_f = i - 1
                a = False
                break
        if not a:
            break
    return 2 if ind_f == 1 else ind_f