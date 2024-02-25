from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import pandas as pd

from dotenv import load_dotenv

from antakia_core.utils.utils import ProblemCategory

load_dotenv()

from antakia.utils.checks import is_valid_model
from antakia_core.utils.variable import Variable, DataVariables
from antakia.gui.gui import GUI

import importlib.metadata

__version__ = importlib.metadata.version("antakia")
__author__ = "AI-vidence"


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

    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            model,
            variables: DataVariables | List[Dict[str, Any]] | pd.DataFrame | None = None,
            X_test: pd.DataFrame = None,
            y_test: pd.Series = None,
            X_exp: pd.DataFrame | None = None,
            score: callable | str = 'auto',
            problem_category: str = 'auto'
    ):
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

        load_dotenv()

        if not is_valid_model(model):
            raise ValueError(model, " should implement predict and score methods")
        X, y, X_exp = self._preprocess_data(X, y, X_exp)

        self.X = X
        self.X_test = X_test
        if y.ndim > 1:
            y = y.squeeze()
        self.y = y.astype(float)
        if y_test is not None and y_test.ndim > 1:
            y_test = y_test.squeeze()
        self.y_test = y_test
        self.model = model
        self.X_exp = X_exp
        self.problem_category = self._preprocess_problem_category(problem_category, model, X)
        self.score = self._preprocess_score(score, self.problem_category)

        self.set_variables(X, variables)

        self.gui = GUI(
            self.X,
            self.y,
            self.model,
            self.variables,
            self.X_test,
            self.y_test,
            self.X_exp,
            self.score,
            self.problem_category
        )

    def set_variables(self, X, variables):
        if variables is not None:
            if isinstance(variables, list):
                self.variables: DataVariables = Variable.import_variable_list(variables)
                if len(self.variables) != len(X.columns):
                    raise ValueError("Provided variable list must be the same length of the dataframe")
            elif isinstance(variables, pd.DataFrame):
                self.variables = Variable.import_variable_df(variables)
            else:
                raise ValueError("Provided variable list must be a list or a pandas DataFrame")
        else:
            self.variables = Variable.guess_variables(X)

    def start_gui(self) -> GUI:
        return self.gui.show_splash_screen()

    def export_regions(self):
        return self.gui.region_set

    def _preprocess_data(self, X: pd.DataFrame, y, X_exp: pd.DataFrame):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(X_exp, np.ndarray):
            X_exp = pd.DataFrame(X_exp)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        X.columns = [str(col) for col in X.columns]
        if X_exp is not None:
            X_exp.columns = X.columns

        if X_exp is not None:
            pd.testing.assert_index_equal(X.index, X_exp.index, check_names=False)
            if X.reindex(X_exp.index).iloc[:, 0].isna().sum() != X.iloc[:, 0].isna().sum():
                raise IndexError('X and X_exp must share the same index')
        pd.testing.assert_index_equal(X.index, y.index, check_names=False)
        return X, y, X_exp

    def _preprocess_problem_category(self, problem_category: str, model, X: pd.DataFrame) -> ProblemCategory:
        if problem_category not in [e.name for e in ProblemCategory]:
            raise ValueError('Invalid problem category')
        if problem_category == 'auto':
            if hasattr(model, 'predict_proba'):
                return ProblemCategory['classification_with_proba']
            pred = self.model.predict(self.X.sample(min(100, len(self.X))))
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

    def _preprocess_score(self, score, problem_category):
        if callable(score):
            return score
        if score != 'auto':
            return score
        if problem_category == ProblemCategory.regression:
            return 'mse'
        return 'accuracy'
