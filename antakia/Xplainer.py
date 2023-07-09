import pandas as pd
import numpy as np

from antakia.Gui import Gui

class Xplainer():
    """
    Xplainer object.
    This object is the main object of the package antakia. It contains all the data and variables needed to run the interface (see antakia.interface).

    Attributes
    -------
    X : pandas dataframe
        The dataframe containing the data to explain.
    Y : pandas series
        The series containing the target variable of the data to explain.
    model : object
        The model used to explain the data.
    selection : list
        The list of the indices of the data currently selected by the user.
    """

    def __init__(self, X: pd.DataFrame, Y: pd.Series = None, model: object = None):
        """
        Constructor of the class Xplainer.

        Parameters
        ---------
        X : pandas dataframe
            The dataframe containing the data to explain.
        Y : pandas series
            The series containing the target variable of the data to explain.
        model : object
            The model used to explain the data.
            The only thing necessary for the model is that it has a method model.fit(X,Y) and model.predict(X) that takes as input a dataframe X and returns a series of predictions.

        Returns
        -------
        Xplainer object
            An Xplainer object.
        """
        self.X = X
        self.Y = Y
        self.model = model
        self.gui = None

    def __str__(self):
        """
        Function that allows to print the Xplainer object.
        """
        print("Xplainer object")

    def startGui(self,
                explanation: str = "SHAP",
                exp_val: pd.DataFrame = None,
                X_all: pd.DataFrame = None,
                default_projection: str = "PaCMAP",
                sub_models: list = None,
                save_regions: list = None,
                display = True,):
        """
        Function that starts the interface.
        """
        self.gui = Gui(self, explanation, exp_val, X_all, default_projection, sub_models, save_regions)
        if display:
            self.gui.display()