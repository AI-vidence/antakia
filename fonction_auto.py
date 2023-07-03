import numpy as np
import pandas as pd
import sklearn

import sklearn.cluster
from copy import deepcopy

import mvlearn

from skrules import SkopeRules

import sklearn.cluster


def create_list_invert(liste, taille):
    l = [[] for i in range(taille)]
    for i in range(len(liste)):
        l[liste[i]].append(i)
    return l


def reset_list(l):
    l = list(l)
    for i in range(max(l) + 1):
        if l.count(i) == 0:
            l = list(np.array(l) - 1)
    return l


def find_best_k(X, indices, recall_min, precision_min):
    recall_min = 0.5
    precision_min = 0.5
    new_X = X.iloc[indices]
    ind_f = 2
    for i in range(2, 9):
        kmeans = sklearn.cluster.KMeans(n_clusters=i, random_state=9, n_init="auto")
        kmeans = sklearn.cluster.AgglomerativeClustering(n_clusters=i)
        kmeans.fit(new_X)
        labels = kmeans.labels_
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
        if a == False:
            break
    if ind_f == 1:
        return 2
    return ind_f


def clustering_dyadique(X, SHAP, n_clusters, default):
    m_kmeans = mvlearn.cluster.MultiviewKMeans(n_clusters=n_clusters, random_state=9)
    l = m_kmeans.fit_predict([X, SHAP])
    if default == False:
        return create_list_invert(l, max(l) + 1), l
    else:
        X_train = pd.DataFrame(X.copy())
        recall_min = 0.8
        precision_min = 0.8
        max_ = max(l) + 1
        l_copy = deepcopy(l)
        for i in range(n_clusters):
            y_train = [1 if x == i else 0 for x in l_copy]
            indices = [i for i in range(len(y_train)) if y_train[i] == 1]
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
            if len(skope_rules_clf.rules_) == 0:
                k = find_best_k(X, indices, recall_min, precision_min)
                kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=9)
                kmeans.fit(X.iloc[indices])
                labels = np.array(kmeans.labels_) + max_
                max_ += k
                l[indices] = labels
        l = reset_list(l)
        return create_list_invert(l, max(l) + 1), l
