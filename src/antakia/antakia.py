from __future__ import annotations

from typing import List, Dict, Any, Callable

import numpy as np
import pandas as pd
from antakia_core.utils import DataVariables

from antakia_core.utils.utils import ProblemCategory, timeit

from antakia.config import AppConfig
from antakia.gui.helpers.data import DataStore
from antakia.utils.logging_utils import Log
from antakia.utils.stats import stats_logger, log_errors
from antakia.utils.checks import is_valid_model
from antakia.gui.gui import GUI


class AntakIA:
    """AntakIA class. 

    Antakia instances provide data and methods to explain a ML model.

    Instance attributes
    -------------------
    X : pd.DataFrame the training dataset
    y : pd.Series the target value
    model : Model
        the model to explain
    variables : a list of Variables, describing X_list[0]
    X_test : pd.DataFrame the test dataset
    y_test : pd.Series the test target value
    score : reference scoring function
    """

    @log_errors
    @timeit
    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 model,
                 variables: DataVariables | List[Dict[str, Any]] | pd.DataFrame
                 | None = None,
                 X_test: pd.DataFrame | None = None,
                 y_test: pd.Series | None = None,
                 X_exp: pd.DataFrame | None = None,
                 score: Callable | str = 'auto',
                 problem_category: str = 'auto',
                 verbose=0):
        """
        AntakiIA constructor.

        Parameters:
            X : pd.DataFrame the training dataset
            y : pd.Series the target value
            model : Model
                the model to explain
            variables : a list of Variables, describing X_list[0]
            X_test : pd.DataFrame the test dataset
            y_test : pd.Series the test target value
            score : reference scoring function
        """
        stats_logger.log('launched', {})
        AppConfig.verbose = verbose

        if not is_valid_model(model):
            raise ValueError(model,
                             " should implement predict and score methods")
        self.data_store = self._get_data_store(
            X=X,
            X_exp=X_exp,
            X_test=X_test,
            model=model,
            problem_category=problem_category,
            score=score,
            variables=variables,
            y=y,
            y_test=y_test)

        with Log('building GUI', 1):
            self.gui = GUI(self.data_store)
        stats_logger.log(
            'launch_info', {
                'data_dim': str(self.data_store.X.shape),
                'category': str(self.data_store.problem_category),
                'provided_exp': self.data_store.X_exp is not None,
                'test_dataset': self.data_store.X_test is not None
            })

    @classmethod
    def _get_data_store(cls,
                        X: pd.DataFrame,
                        y: pd.Series,
                        model,
                        X_exp: pd.DataFrame = None,
                        X_test: pd.DataFrame = None,
                        problem_category: str = 'auto',
                        score: Callable | str = 'auto',
                        variables: DataVariables | List[Dict[str, Any]]
                        | pd.DataFrame | None = None,
                        y_test=None):
        with Log('cleaning data', 2):
            X, y, X_exp = cls._preprocess_data(X, y, X_exp)
            if X_test is not None:
                X_test, y_test, _ = cls._preprocess_data(X_test, y_test, None)
        if y.ndim > 1:
            y = y.squeeze()  # type:ignore
        y = y.astype(float)
        if y_test is not None and y_test.ndim > 1:
            y_test = y_test.squeeze()  # type:ignore
        problem_category = cls._preprocess_problem_category(
            problem_category, model, X)
        score = cls._preprocess_score(score, problem_category)
        with Log('building variables', 2):
            variables = cls.preprocess_variables(X, variables)
        return DataStore(X=X,
                         y=y,
                         variables=variables,
                         X_exp=X_exp,
                         X_test=X_test,
                         y_test=y_test,
                         model=model,
                         problem_category=problem_category,
                         score=score)

    @classmethod
    @timeit
    def preprocess_variables(cls, X, variables) -> DataVariables:
        """
        Set variables attribute according to variable input and X
        """
        if variables is not None and not isinstance(variables,
                                                    (list, pd.DataFrame)):
            raise ValueError(
                "Provided variable list must be a list or a pandas DataFrame")
        return DataVariables.build_variables(X, variables)

    @timeit
    def start_gui(self) -> GUI:
        return self.gui.initialize()

    @timeit
    def export_regions(self):
        """
        get region set from modeling
        """
        return self.data_store.region_set

    @classmethod
    @timeit
    def _preprocess_data(cls, X: pd.DataFrame, y, X_exp: pd.DataFrame | None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(X_exp, np.ndarray):
            X_exp = pd.DataFrame(X_exp)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        X.columns = [str(col) for col in X.columns]  # type:ignore
        if X_exp is not None:
            X_exp.columns = X.columns

        if X_exp is not None:
            pd.testing.assert_index_equal(X.index,
                                          X_exp.index,
                                          check_names=False)  # type:ignore
            if X.reindex(X_exp.index).iloc[:, 0].isna().sum(
            ) != X.iloc[:, 0].isna().sum():
                raise IndexError('X and X_exp must share the same index')
        pd.testing.assert_index_equal(X.index, y.index,
                                      check_names=False)  # type:ignore
        return X, y, X_exp

    @classmethod
    @timeit
    def _preprocess_problem_category(cls, problem_category: str, model,
                                     X: pd.DataFrame) -> ProblemCategory:
        if problem_category not in [e.name for e in ProblemCategory]:
            raise ValueError('Invalid problem category')
        with Log('Preprocessing problem category', 2):
            if problem_category == 'auto':
                if hasattr(model, 'predict_proba'):
                    return ProblemCategory['classification_with_proba']
                pred = model.predict(X.sample(min(100, len(X))))
                if len(pred.shape) > 1 and pred.shape[1] > 1:
                    return ProblemCategory['classification_proba']
                return ProblemCategory['regression']
            if problem_category == 'classification':
                if hasattr(model, 'prodict_proba'):
                    return ProblemCategory['classification_with_proba']
                pred = model.predict(X.sample(min(100, len(X))))
                if len(pred.shape) > 1 and pred.shape[1] > 1:
                    return ProblemCategory['classification_proba']
                return ProblemCategory['classification_label_only']
            return ProblemCategory[problem_category]

    @classmethod
    @timeit
    def _preprocess_score(cls, score, problem_category):
        """
        preprocess score to imput default score if not provided
        """
        if callable(score):
            return score
        if score != 'auto':
            return score
        if problem_category == ProblemCategory.regression:
            return 'mse'
        return 'accuracy'

    @timeit
    def predict(self, X):
        """
        predict the result using the region_set
        """
        return self.gui.data_store.region_set.predict(X)
