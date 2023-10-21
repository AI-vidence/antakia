import pandas as pd

from antakia.data import ExplanationMethod, Variable, is_valid_model
from antakia.gui import GUI
from antakia.utils import confLogger

import logging
from logging import getLogger
logger = logging.getLogger(__name__)
handler = confLogger(logger)
handler.clear_logs()
handler.show_logs()

class AntakIA():
    """
    AntakIA class.

    Antakia instances provide data and methods to explain a ML model.

    Instance attributes
    -------------------
    X_list : a list of one or more pd.DataFrame 
    X_method_list : a list starting with ExplanationMethod.NONE, followed by one or more ExplanationMethod
    y : a pd.Series
    Y_pred : a pd.Series
    variables : a list of Variables, describing X_list[0]
    model : Model
        the model to explain

    regions : List of Selection objects


    """

    def __init__(self, X, y:pd.Series, model, method=None):
        """
        AntakiIA constructor.

        Parameters:
        X : can be a pd.DataFrame or a list of pd.DataFrame
        """

        if y is None or not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series")
        if not is_valid_model(model):
            raise ValueError(model, " should implement predict and score methods")
        
        if isinstance(X, list):
            self.X_list = X
            if isinstance(method, list) and (len(method) == len(X)):
                self.X_method_list = method
                for i in range(len(X)):
                    if not isinstance(X[i], pd.DataFrame):
                        raise ValueError("X must be a list of pandas DataFrame")
                    if not ExplanationMethod.is_valid_explanation_method(method[i]):
                        raise ValueError(method[i], " is not a valide ExplanationMethod code")
            else:
                raise ValueError("Since your provided a list of X, you must provide a list of methods of the same size")
            
        else:
            self.X_list = [X]
            if isinstance(method, int) and method == ExplanationMethod.NONE:
                self.X_method_list = [method]
            else:
                raise ValueError("bad explain method provided")

        self.variables = Variable.guess_variables(X[0])
        self.model = model
        self.y = y
        self.Y_pred = model.predict(X[0])


        # if csv_file_name is not None :
        #     with open(csv_file_name) as csv_file:
        #         self.X_exp = pd.read_csv(csv_file)
        #         # TODO : we should check coherecne between X and X_exp

        self.regions = []

    def startGUI(self)-> GUI:
        return GUI(self.X_list, self.X_method_list, self.y, self.model).show_splash_screen()