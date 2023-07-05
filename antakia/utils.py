"""
Utils module for the antakia package.

This module contains the following functions:
    - fonction_auto_clustering
    - create_save
    - load_save
    - from_rules
"""

import numpy as np
import pandas as pd
import sklearn

import sklearn.cluster
from copy import deepcopy

import mvlearn

from skrules import SkopeRules

import sklearn.cluster

import json


def _create_list_invert(liste, taille):
    l = [[] for _ in range(taille)]
    for i in range(len(liste)):
        l[liste[i]].append(i)
    return l


def _reset_list(l):
    l = list(l)
    for i in range(max(l) + 1):
        if l.count(i) == 0:
            l = list(np.array(l) - 1)
    return l


def _find_best_k(X, indices, recall_min, precision_min):
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
    return 2 if ind_f == 1 else ind_f


def _clustering_dyadique(X, SHAP, n_clusters, default):
    m_kmeans = mvlearn.cluster.MultiviewKMeans(n_clusters=n_clusters, random_state=9)
    l = m_kmeans.fit_predict([X, SHAP])
    if default != False:
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
                k = _find_best_k(X, indices, recall_min, precision_min)
                kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=9)
                kmeans.fit(X.iloc[indices])
                labels = np.array(kmeans.labels_) + max_
                max_ += k
                l[indices] = labels
        l = _reset_list(l)
    return _create_list_invert(l, max(l) + 1), l


def fonction_auto_clustering(X1, X2, n_clusters, default):
    """Return a clustering, generated a dyadic way.

    Function that allows to cluster the data in a dyadic way : the clusters are both in the X1 and X2 spaces.
    The clustering is done by the function antakia._clustering_dyadique from the module fonction_auto (see fonction_auto.py).

    Parameters
    ---------
    X1 : pandas dataframe
        The dataframe containing the first data to cluster.
    X2 : pandas dataframe
        The dataframe containing the second data to cluster.
    n_clusters : int
        The number of clusters to create.
    default : bool
        If False, the clustering will be done with a fixed number of cluster (n_clusters). If True, the clustering will be done with a variable number of clusters.
        The algorithm will then try to find the best number of clusters to use.

    Returns
    -------
    clusters : list
        A list of lists, each list being a cluster. Each cluster is a list of indices of the data in the dataframe X.
    clusters_axis : list
        A list of size len(X), each element being the axis of the cluster the corresponding data belongs to.
    """
    return _clustering_dyadique(X1, X2, n_clusters, default)


def create_save(liste=None, nom: str = "Default name", sub_models: list = None):
    """Return a save file from a list of pre-defined regions.

    Function that allows to create a save file from a list of pre-defined regions.
    The save file is a json file that contains a list of dictionaries, usually generated in the interface (see antakia.interface).

    Parameters
    ---------
    liste : list
        The list of pre-defined regions to save.
    nom : str
        The name of the save file.
    sub_models : list
        The list of sub_models used to generate the pre-defined regions.

    Returns
    -------
    retour : dict
        A dictionary containing the name of the save file, the list of pre-defined regions and the list of sub_models used to generate the pre-defined regions.
    """
    if sub_models is None:
        sub_models = []
    return {"nom": nom, "liste": liste, "sub_models": sub_models}


def load_save(local_path):
    """Return a save file from a json file.

    Function that allows to load a save file.
    The save file is a json file that contains a list of dictionaries, usually generated in the interface (see antakia.gui).

    Parameters
    ----------
    local_path :str
        The path to the save file. If None, the function will return a message saying that no save file was loaded.

    Returns
    ----------
    data : list
        A list of dictionaries, each dictionary being a save file. This list can directly be passed to the function antakia.interface so as to load the save file.
    """
    with open(local_path) as json_file:
        data = json.load(json_file)
    for dictio in data:
        dictio["sub_models"] = eval(dictio["sub_models"] + "()")
    return data


def from_rules(df, rules_list):
    """Return a dataframe with specific rules applied to it.

    The rules are meant to be generated by the function antakia.results after creating a list of regions.

    Parameters
    ---------
    df : pandas dataframe
        The dataframe to which the rules will be applied
    rules_list : list
        The list of rules to apply to the dataframe. The rules must be in the form of a list of lists, each list being a rule in the following format:
        [minimum, operator1, column, operator2, maximum].
        For example : [0.5, '<=', 'cool_feature', '<=', 0.7]

    Returns
    -------
    df : pandas dataframe
        The dataframe with the rules applied to it

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 2.5], 'b': [0.1, 0.2, 0.3, 0.4, 0.5, 0.33]})
    >>> rules_list = [[1.5, '<=', 'a', '<=', 3], [0.1, '<=', 'b', '<=', 0.4]]
    >>> df = from_rules(df, rules_list)
    >>> df
         a    b
    1  2.0  0.2
    2  3.0  0.3
    3  2.5  0.33

    """
    l = rules_list
    for i in range(len(l)):
        regle1 = "df.loc[" + str(l[i][0]) + l[i][1] + "df['" + l[i][2] + "']]"
        regle2 = "df.loc[" + "df['" + l[i][2] + "']" + l[i][3] + str(l[i][4]) + "]"
        df = eval(regle1)
        df = eval(regle2)
    return df
