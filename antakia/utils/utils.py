"""
Utils module for the antakia package.
"""
import math
import time

import numpy as np
from functools import wraps

import pandas as pd


def overlap_handler(ens_potatoes, liste):
    # function that allows you to manage conflicts in the list of regions.
    # indeed, as soon as a region is added to the list of regions, the points it contains are removed from the other regions
    # TODO use np/pd to reimplement this
    gliste = [x.indexes for x in ens_potatoes]
    for i in range(len(gliste)):
        a = 0
        for j in range(len(gliste[i])):
            if gliste[i][j - a] in liste:
                gliste[i].pop(j - a)
                a += 1
    for i in range(len(ens_potatoes)):
        ens_potatoes[i].setIndexes(gliste[i])
    return ens_potatoes


def in_index(indexes: list, X: pd.DataFrame) -> bool:
    """
    Checks if a list of indexes is in the index of a DataFrame
    """
    try:
        X.loc[indexes]
        return True
    except KeyError:
        return False


def rows_to_mask(X: pd.DataFrame, rows_list: list) -> pd.Series:
    """
    Converts DataFrame row numbers to Index numbers
    """
    mask = pd.Series(np.zeros(len(X)), index=X.index)
    mask.iloc[rows_list] = 1
    return mask.astype(bool)


def indexes_to_rows(X: pd.DataFrame, indexes_list: list) -> list:
    """
    Converts DataFrame Index numbers to row numbers
    """
    index = pd.Series(np.arange(len(X)), index=X.index)
    return index.loc[indexes_list].tolist()


def mask_to_rows(mask: pd.Series) -> list:
    return mask_to_index(mask.reset_index(drop=True))


def mask_to_index(mask: pd.Series) -> list:
    return mask[mask].index.tolist()


def boolean_mask(X: pd.DataFrame, value: bool = True):
    return pd.Series([value] * len(X), index=X.index)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def debug(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args} and kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result

    return wrapper


def compute_step(min, max):
    step = (max - min) / 100
    round_value = round(math.log(step / 2) / math.log(10)) - 1
    min_ = np.round(min, -round_value)
    max_ = np.round(max, -round_value)
    step = np.round(step, -round_value)
    return min_, max_, step


# First color can't be blue, reserved for the rules - grey is reserved to background
colors = ["red", "blue", "green", "yellow", "orange", "pink", "brown", "cyan", "black"]
