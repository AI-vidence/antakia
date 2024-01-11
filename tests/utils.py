def compare_indexes(df1, df2) -> bool:
    return df1.index.equals(df2.index)


# Test Dim Reduction --------------------

def dr_callback(*args):
    pass


import pandas as pd

from antakia.utils.utils import rows_to_mask, mask_to_rows, mask_to_index


def test_rows_to_mask():
    X = pd.DataFrame([[1]] * 5)
    row_list = [2, 4]
    res = pd.Series([False, False, True, False, True])
    mask = rows_to_mask(X, row_list)

    assert (mask == res).all()

    assert mask_to_rows(mask) == row_list

    assert mask_to_index(mask) == row_list


test_rows_to_mask()
