from antakia.compute.explanation.explanation_method import ExplanationMethod
from antakia.gui.explanation_values import ExplanationValues

import pandas as pd


def compare_indexes(df1, df2) -> bool:
    return df1.index.equals(df2.index)


# Test Dim Reduction --------------------

def dr_callback(*args):
    pass


def generate_ExplanationValues(model=None, X_exp=None):
    X = pd.DataFrame([[4, 7, 10],
                      [5, 8, 11],
                      [6, 9, 12]],
                     index=[1, 2, 3],
                     columns=['a', 'b', 'c'])
    y = pd.Series([1, 2, 3])

    if model is None:
        model = 'DT'
    if X_exp is None:
        X_exp = pd.DataFrame([[1, 7, 10],
                              [5, 8, 11],
                              [6, 9, 12]],
                             index=[1, 2, 3],
                             columns=['a', 'b', 'c'])

    exp_val = ExplanationValues(X, y, model, lambda *args: None, X_exp)

    return X, y, X_exp, exp_val


class EMPTYExplanation(ExplanationMethod):
    def __init__(self, X: pd.DataFrame, y: pd.Series, model, on_change_callback: callable):
        super().__init__(1, X, model, on_change_callback)

    def compute(self):
        self.publish_progress(100)
        return 0
