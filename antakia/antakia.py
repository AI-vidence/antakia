import pandas as pd

from antakia.data import Dataset, ExplanationDataset, ExplanationMethod, Model, Variable
from antakia.potato import Potato
from antakia.gui import GUI
from antakia.utils import confLogger

import logging
from logging import getLogger
import csv

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
    X : a pd.DataFrame 
    variables : a list of Variables
    X_exp : a pd.DataFrame
    exp_method : an int, ExplanationMethod
    Y : a pd.Series
    model : Model
        the model to explain

    regions : List of Selection objects


    """

    def __init__(self, X: pd.DataFrame, y:pd.Series, model: Model, csv_file_name:str=None, explain_method: int = None):

        if X is None or not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if y is None or not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series")
        if model is None or not isinstance(model, Model):
            raise ValueError("model must be a Model object")
        
        self.X = X
        self.variables = Variable.guess_variables(X)
        self.y = y
        self.model = model

        self.X_exp = None
        self.exp_method = explain_method
        if csv_file_name is not None :
            with open(csv_file_name) as csv_file:
                self.X_exp = pd.read_csv(csv_file)
                # TODO : we should check coherecne between X and X_exp

        self.regions = []

    def startGUI(self)-> GUI:
        return GUI(self)