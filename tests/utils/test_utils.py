import pandas as pd
import numpy as np
import pytest

from antakia.utils import utils


def test_overlap_handler():
    pass
    # print(utils.overlap_handler())
    # assert utils.overlap_handler(ens,liste) == 0


def test_in_index():
    df1 = pd.DataFrame([[4, 7, 10],
                        [5, 8, 11],
                        [6, 9, 12]],
                       index=[1, 2, 3],
                       columns=['a', 'b', 'c'])

    assert utils.in_index([1, 2], df1)
    assert not utils.in_index([0, 1, 2], df1)


def test_rows_to_mask_1():
    X = pd.DataFrame([[1]] * 5)
    row_list = [2, 4]
    res = pd.Series([False, False, True, False, True])
    mask = utils.rows_to_mask(X, row_list)

    assert (mask == res).all()

    assert utils.mask_to_rows(mask) == row_list

    assert utils.mask_to_index(mask) == row_list


def test_rows_to_mask():
    df1 = pd.DataFrame([[4, 7, 10],
                        [5, 8, 11],
                        [6, 9, 12]],
                       index=[1, 2, 3],
                       columns=['a', 'b', 'c'])

    np.testing.assert_array_equal(
        utils.rows_to_mask(df1, []).values,
        pd.Series([False, False, False], [1, 2, 3]).values
    )

    np.testing.assert_array_equal(
        utils.rows_to_mask(df1, [0, 1]).values,
        pd.Series([True, True, False], [1, 2, 3]).values
    )

    with pytest.raises(IndexError):
        np.testing.assert_array_equal(
            utils.rows_to_mask(df1, [0, 4]).values,
            pd.Series([True, True, False], [1, 2, 3]).values
        )


def test_indexes_to_rows():
    df1 = pd.DataFrame([[4, 7, 10],
                        [5, 8, 11],
                        [6, 9, 12]],
                       index=[1, 2, 3],
                       columns=['a', 'b', 'c'])

    assert utils.indexes_to_rows(df1, [1, 2]) == [0, 1]
    assert utils.indexes_to_rows(df1, [1, 2, 3]) == [0, 1, 2]
    assert utils.indexes_to_rows(df1, []) == []


def test_mask_to_rows():
    mask1 = pd.Series([True, True, False],
                      index=[1, 2, 3])

    mask2 = pd.Series([False, False, False],
                      index=[1, 2, 3])

    np.testing.assert_array_equal(
        utils.mask_to_rows(mask1),
        [0, 1]
    )
    np.testing.assert_array_equal(
        utils.mask_to_index(mask2),
        []
    )


def test_mask_to_index():
    mask1 = pd.Series([True, True, False],
                      index=[1, 2, 3])
    mask2 = pd.Series([False, False, False],
                      index=[1, 2, 3])

    np.testing.assert_array_equal(
        utils.mask_to_index(mask1),
        [1, 2]
    )
    np.testing.assert_array_equal(
        utils.mask_to_index(mask2),
        []
    )


def test_boolean_mask():
    df1 = pd.DataFrame([[4, 7, 10],
                        [5, 8, 11],
                        [6, 9, 12]],
                       index=[1, 2, 3],
                       columns=['a', 'b', 'c'])

    np.testing.assert_array_equal(
        utils.boolean_mask(df1, False),
        pd.Series([False, False, False])
    )

    np.testing.assert_array_equal(
        utils.boolean_mask(df1),
        pd.Series([True, True, True])
    )


def test_compute_step():
    assert utils.compute_step(0, 100) == (0, 100, 1.0)
    assert utils.compute_step(0, 50) == (0, 50, 0.5)
    assert utils.compute_step(0, 1) == (0, 1, 0.01)
    assert utils.compute_step(0, 1000) == (0, 1000, 10)
