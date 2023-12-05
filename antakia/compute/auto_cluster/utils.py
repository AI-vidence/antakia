import numpy as np


def reassign_clusters(clusters, n_clusters=None):
    if n_clusters is None:
        n_clusters = len(np.unique(clusters))
    for i in range(n_clusters):
        if (clusters == i).sum() == 0:
            clusters[clusters == clusters.max()] = i
    return clusters


def _invert_list(a_list: list, size: int) -> list:
    newList = [[] for _ in range(size)]
    for i in range(len(a_list)):
        newList[a_list[i]].append(i)
    return newList


def _reset_list(a_list: list) -> list:
    a_list = list(a_list)
    for i in range(max(a_list) + 1):
        if a_list.count(i) == 0:
            a_list = list(np.array(a_list) - 1)
    return a_list
