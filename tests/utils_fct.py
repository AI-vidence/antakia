from antakia_core.utils.utils import ProblemCategory

from antakia.explanation.explanation_method import ExplanationMethod
from antakia.gui.explanation_values import ExplanationValues

import pandas as pd
import ipyvuetify as v

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


def generate_df_series_callable():
    test_progress_bar.reset_progress_bar()

    X = pd.DataFrame([[4, 7, 10],
                      [5, 8, 11],
                      [6, 9, 12]],
                     index=[1, 2, 3],
                     columns=['a', 'b', 'c'])
    y = pd.Series([1, 2, 3])

    return X, y, test_progress_bar.update


def generate_ExplanationValues(model=None, X_exp=None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame,
ExplanationValues, callable]:
    X, y, progress_callback = generate_df_series_callable()

    def on_change_callback(pv, progress=None):
        if progress is None:
            progress = test_progress_bar.update
        progress(100, 0)

    if model is None:
        model = DummyModel()

    exp_val = ExplanationValues(X, y, model, ProblemCategory.regression, on_change_callback, dummy_callable, X_exp)

    return X, y, X_exp, exp_val, on_change_callback


class EMPTYExplanation(ExplanationMethod):
    def __init__(self, X: pd.DataFrame, y: pd.Series, model, on_change_callback: callable):
        super().__init__(1, X, model, ProblemCategory.regression, on_change_callback)

    def compute(self):
        self.publish_progress(100)
        return 0
