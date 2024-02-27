import numpy as np
import pandas as pd
import pytest
from antakia_core.utils.utils import ProblemCategory

from antakia.explanation.explanations import compute_explanations
from antakia.utils.dummy_datasets import load_dataset
from sklearn.tree import DecisionTreeRegressor

from tests.utils_fct import test_progress_bar, dummy_callable


def test_compute_explanations():
    X, y = load_dataset('Corner', 100, random_seed=42)
    X = pd.DataFrame(X, columns=['X1', 'X2'])
    X['X3'] = np.random.random(len(X))
    y = pd.Series(y)
    model_DT = DecisionTreeRegressor().fit(X, y)
    X = X.sample(100)  # randomize index order
    for i in range(4):
        if i in (1, 2):
            X_exp = compute_explanations(X, model_DT, i, ProblemCategory.regression, test_progress_bar.update)
            pd.testing.assert_index_equal(X_exp.index, X.index)
            pd.testing.assert_index_equal(X_exp.columns, X.columns)
        else:
            with pytest.raises(ValueError):
                compute_explanations(X, model_DT, ProblemCategory.regression, i, test_progress_bar.update)
