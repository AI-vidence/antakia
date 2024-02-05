from __future__ import annotations

from typing import List, Dict, Any

import pandas as pd

from dotenv import load_dotenv

load_dotenv()

from antakia.utils.checks import is_valid_model
from antakia.utils.variable import Variable, DataVariables
from antakia.gui.gui import GUI


class AntakIA:
    """
    AntakIA class. 

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
            score: callable | str = 'mse'
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

        self.X = X
        self.X_test = X_test
        if y.ndim > 1:
            y = y.squeeze()
        self.y = y
        if y_test is not None and y_test.ndim > 1:
            y_test = y_test.squeeze()
        self.y_test = y_test
        self.model = model
        self.score = score

        if X_exp is not None:
            # It's common to have column names ending with _shap, so we remove them
            X_exp.columns = X_exp.columns.astype(str)
            X_exp.columns = X_exp.columns.str.replace('_shap', '')
        self.X_exp = X_exp

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

        self.gui = GUI(
            self.X,
            self.y,
            self.model,
            self.variables,
            self.X_test,
            self.y_test,
            self.X_exp,
            self.score
        )

    def start_gui(self) -> GUI:
        return self.gui.show_splash_screen()

    def export_regions(self):
        return self.gui.region_set
