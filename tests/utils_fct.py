import pandas as pd
import ipyvuetify as v
from pandas.api.types import is_bool_dtype

from antakia.gui.helpers.progress_bar import ProgressBar


def compare_indexes(df1, df2) -> bool:
    return df1.index.equals(df2.index)


# Test Dim Reduction --------------------


def dummy_callable(*args):
    pass


test_progress_bar = ProgressBar(v.ProgressLinear(), reset_at_end=False)


class DummyModel:

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return ((X.iloc[:, 0] > 0.5) & (X.iloc[:, 1] > 0.5)).astype(int)
        return ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)).astype(int)

    def fit(self, X, y):
        return self

    def score(self, *args):
        return 1


def is_mask_of_X(mask, X):
    return (isinstance(mask, pd.Series) and is_bool_dtype(mask)
            and pd.testing.assert_index_equal(mask.index, X.index))
