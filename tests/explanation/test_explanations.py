from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from antakia_core.utils.utils import ProblemCategory

from antakia.explanation.explanations import compute_explanations
from antakia.utils.dummy_datasets import load_dataset
from sklearn.tree import DecisionTreeRegressor

from tests.utils_fct import test_progress_bar, dummy_callable, DummyModel


class TestComputeExplanation(TestCase):
    def setUp(self):
        self.X, y = load_dataset('Corner', 100, random_seed=42)
        self.X = pd.DataFrame(self.X, columns=['X1', 'X2'],
                              index=np.random.choice(np.arange(2 * len(self.X)), len(self.X)))
        self.X['X3'] = np.random.random(len(self.X))
        y = pd.Series(y)
        self.model_DT = DecisionTreeRegressor().fit(self.X, y)
        self.model_any = DummyModel().fit(self.X, y)
        self.X = self.X.sample(100)  # randomize index order

    def test_compute_explanations_DT(self):
        """
        run compute explanation with all explanation methods and check output format
        Returns
        -------

        """
        for i in range(4):
            if i in (1, 2):
                X_exp = compute_explanations(self.X, self.model_DT, i, ProblemCategory.regression,
                                             test_progress_bar.update)
                pd.testing.assert_index_equal(X_exp.index, self.X.index)
                pd.testing.assert_index_equal(X_exp.columns, self.X.columns)
            else:
                with pytest.raises(ValueError):
                    compute_explanations(self.X, self.model_DT, ProblemCategory.regression, i, test_progress_bar.update)

    def test_compute_explanations_dummy(self):
        """
        run compute explanation with all explanation methods and check output format
        Returns
        -------

        """
        for i in range(4):
            if i in (1, 2):
                X_exp = compute_explanations(self.X, self.model_any, i, ProblemCategory.regression,
                                             test_progress_bar.update)
                pd.testing.assert_index_equal(X_exp.index, self.X.index)
                pd.testing.assert_index_equal(X_exp.columns, self.X.columns)
            else:
                with pytest.raises(ValueError):
                    compute_explanations(self.X, self.model_any, ProblemCategory.regression, i,
                                         test_progress_bar.update)
