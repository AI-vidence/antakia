import pandas as pd
import ipyvuetify as v
from pandas.api.types import is_bool_dtype

from antakia_core.utils.utils import ProblemCategory
from antakia.explanation.explanation_method import ExplanationMethod

from antakia.gui.progress_bar import ProgressBar


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
        pass

    def score(self, *args):
        return 1

class EMPTYExplanation(ExplanationMethod):
    def __init__(self, X: pd.DataFrame, y: pd.Series, model, on_change_callback: callable):
        super().__init__(1, X, model, ProblemCategory.regression, on_change_callback)

    def compute(self):
        self.publish_progress(100)
        return 0


def is_mask_of_X(mask, X):
    return (
            isinstance(mask, pd.Series) and
            is_bool_dtype(mask) and
            pd.testing.assert_index_equal(mask.index, X.index)
    )
