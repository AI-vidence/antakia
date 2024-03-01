from typing import Callable

from functools import partial

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, make_blobs
from scipy.stats import multivariate_normal


def generate_corner_dataset(
        num_samples: int, corner_position: str = "top_right", random_seed: int | None = None
) -> (np.ndarray, np.ndarray):
    """Generate a toy dataset with a corner of the feature space.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    corner_position : str, optional
        Position of the corner in the feature space, by default "top_right"

    Returns
    -------
    (np.ndarray, np.ndarray)
        Tuple of feature matrix and target vector.

    Raises
    ------
    ValueError
        If corner_position is not one of "top_left", "top_right", "bottom_left", "bottom_right".
    """
    np.random.seed(random_seed)
    X = np.random.uniform(0, 1, (num_samples, 2))

    if corner_position == "top_right":
        mask = (X[:, 0] > 0.5) & (X[:, 1] > 0.5)
    elif corner_position == "top_left":
        mask = (X[:, 0] < 0.5) & (X[:, 1] > 0.5)
    elif corner_position == "bottom_right":
        mask = (X[:, 0] > 0.5) & (X[:, 1] < 0.5)
    elif corner_position == "bottom_left":
        mask = (X[:, 0] < 0.5) & (X[:, 1] < 0.5)

    else:
        raise ValueError(
            "Invalid corner position must be one of: top_right, top_left, bottom_right, bottom_left."
        )

    y = mask.astype(int)
    return X, y


def get_data_from_mixture_distribution(
        num_samples, positive_component_rvs: Callable, negative_component_rvs: Callable
):
    d = positive_component_rvs(size=1).shape[0]
    Y = np.random.binomial(1, 0.5, num_samples)
    X = np.zeros((num_samples, d))
    for i in range(num_samples):
        if Y[i] == 1:
            X[i] = positive_component_rvs(size=1)
        else:
            X[i] = negative_component_rvs(size=1)
    return X, Y


def mixture_dataset(num_samples, positive_mean=[0, 0], negative_mean=[0, 1], positive_cov=[0.5, 0.5],
                    negative_cov=[0.5, 0.5], **kwargs):
    X, y = get_data_from_mixture_distribution(
        num_samples=num_samples,
        positive_component_rvs=partial(multivariate_normal.rvs, positive_mean, positive_cov),
        negative_component_rvs=partial(multivariate_normal.rvs, negative_mean, negative_cov),
    )
    return X, y


def blobs_dataset(num_samples, **kwargs):
    return make_blobs(
        n_samples=num_samples,
        centers=[[0, 0], [1, 0.1]],
        n_features=2,
        random_state=0,
        cluster_std=0.5,
    )


def breast_cancer_dataset(num_samples, **kwargs):
    breast_cancer = load_breast_cancer()
    idx = np.random.randint(0, 500, num_samples)
    X = breast_cancer.data[:, [7, 9]][idx]
    y = breast_cancer.target[idx]
    return X, y


def xor_dataset(num_samples, var=1, **kwargs):
    noise = np.random.normal(0, var, (num_samples, 2))
    X = np.random.randint(0, 2, (num_samples, 2))
    y = np.abs(X[:, 0] - X[:, 1])
    X = X + noise
    return X, y


def xor_proba(X, var):
    def proba_point(p_x, p_y, X, var):
        d_x = (X[:, 0] - p_x)
        d_y = (X[:, 1] - p_y)
        d_2 = (d_x ** 2 + d_y ** 2) / (var ** 2)
        return np.exp(-d_2 / 2) / (var * 2 * np.pi)

    proba = np.zeros((len(X), 2))
    for point in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        proba[:, point[2]] += proba_point(point[0], point[1], X, var)
    return proba[:, 1] / (proba.sum(axis=1))


DATASETS = {
    "Breast Cancer": breast_cancer_dataset,
    "Corner": generate_corner_dataset,
    "Blobs": blobs_dataset,
    "gaussian_mixture": mixture_dataset,
    "xor": xor_dataset
}


def load_dataset(
        dataset_name: str | None, num_samples: int = 100, **kwargs
) -> (pd.DataFrame, pd.Series):
    if dataset_name in DATASETS:
        return DATASETS[dataset_name](num_samples, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
