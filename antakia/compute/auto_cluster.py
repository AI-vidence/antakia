from copy import deepcopy

import mvlearn
import math
import numpy as np
import pandas as pd
import sklearn
from skrules import SkopeRules

import warnings
from numba.core.errors import NumbaDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=UserWarning)

from antakia.rules import Rule
from antakia.data import Variable
import antakia.utils as utils


def skope_rules(df_indexes: list, base_space_df: pd.DataFrame, variables: list = None, precision: float = 0.7,
                recall: float = 0.7) -> list:
    """
    variables : list of Variables of the app
    df_indexes : list of (DataFrame) indexes for the points selected in the GUI
    base_space_df : the dataframe on which the rules will be computed / extracted. May be VS or ES values
    precision for SKR binary classifer : defaults to 0.7
    recall for SKR binary classifer : defaults to 0.7
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_train = np.zeros(len(base_space_df))
        y_train[utils.indexes_to_rows(base_space_df, df_indexes)] = 1  # our target

        if variables is None:
            variables = Variable.guess_variables(base_space_df)

        sk_classifier = SkopeRules(
            feature_names=Variable.vars_to_sym_list(variables),
            random_state=42,
            n_estimators=5,
            recall_min=recall,
            precision_min=precision,
            max_depth_duplication=0,
            max_samples=1.0,
            max_depth=3,
        )

        sk_classifier.fit(base_space_df, y_train)

    if sk_classifier.rules_ != []:
        rules_list, score_dict = Rule._extract_rules(sk_classifier.rules_, base_space_df, variables)

        if len(rules_list) >= 0:
            Rule._combine_rule_list(rules_list)

        # We remove infinity in rules : we convert in simple rule if inf present
        # We had to wait for _combine_rule_list to proceed
        for rule in rules_list:
            if rule.min == -math.inf:
                rule.min = None
                rule.operator_min = None
            if rule.max == math.inf:
                rule.max = None
                rule.operator_max = None
        return rules_list, score_dict

    else:
        return [], {}


def auto_cluster(X: pd.DataFrame, shap_values: pd.DataFrame, n_clusters: int, default: bool) -> list:
    """ Returns a list of regions, a region is a list of indexes of X
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_kmeans = mvlearn.cluster.MultiviewKMeans(n_clusters=n_clusters, random_state=9)
        a_list = m_kmeans.fit_predict([X, shap_values])
        clusters_number = 0
        if default is True:
            X_train = pd.DataFrame(X.copy())
            recall_min = 0.8
            precision_min = 0.8
            max_ = max(a_list) + 1
            l_copy = deepcopy(a_list)
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
                    clusters_number += k
                    # kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=9)
                    agglo = sklearn.cluster.AgglomerativeClustering(n_clusters=k)
                    agglo.fit(X.iloc[indices])
                    labels = np.array(agglo.labels_) + max_
                    max_ += k
                    a_list[indices] = labels
                else:
                    clusters_number += 1
            a_list = _reset_list(a_list)
        a_list = list(np.array(a_list) - min(a_list))
        return _invert_list(a_list, max(a_list) + 1), a_list


def _invert_list(a_list: list, size: int) -> list:
    newList = [[] for _ in range(size)]
    for i in range(len(a_list)):
        newList[a_list[i]].append(i)
    return newList


def _reset_list(a_list: list) -> list:
    with warnings.catch_warnings():
        a_list = list(a_list)
        for i in range(max(a_list) + 1):
            if a_list.count(i) == 0:
                a_list = list(np.array(a_list) - 1)
        return a_list


def _find_best_k(X: pd.DataFrame, indices: list, recall_min: float, precision_min: float) -> int:
    recall_min = 0.7
    precision_min = 0.7
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