from antakia.compute.explanation.explanation_method import ExplanationMethod
from antakia.gui.explanation_values import ExplanationValues

import pandas as pd
import ipyvuetify as v

from antakia.gui.progress_bar import ProgressBar


def compare_indexes(df1, df2) -> bool:
    return df1.index.equals(df2.index)


# Test Dim Reduction --------------------

def dr_callback(*args):
    pass


test_progress_bar = ProgressBar(v.ProgressLinear(), reset_at_end=False)


def generate_df_series_callable():
    test_progress_bar.reset_progress_bar()

    X = pd.DataFrame([[4, 7, 10],
                      [5, 8, 11],
                      [6, 9, 12]],
                     index=[1, 2, 3],
                     columns=['a', 'b', 'c'])
    y = pd.Series([1, 2, 3])

    return X, y, test_progress_bar.update


def generate_ExplanationValues(model=None, X_exp=None):
    X, y, progress_callback = generate_df_series_callable()

    def on_change_callback(pv, progress=None):
        if progress is None:
            progress = test_progress_bar.update
        progress(100, 0)

    if model is None:
        model = 'DT'

    exp_val = ExplanationValues(X, y, model, on_change_callback, X_exp)

    return X, y, X_exp, exp_val


class EMPTYExplanation(ExplanationMethod):
    def __init__(self, X: pd.DataFrame, y: pd.Series, model, on_change_callback: callable):
        super().__init__(1, X, model, on_change_callback)

    def compute(self):
        self.publish_progress(100)
        return 0
