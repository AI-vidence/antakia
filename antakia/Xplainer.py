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
    gui : Gui object
        The Gui object that contains the interface.
    imported_values : pandas dataframe
        The dataframe containing the explanatory data imported by the user.
    SHAP_values : pandas dataframe
        The dataframe containing the SHAP values of the data.
    LIME_values : pandas dataframe
        The dataframe containing the LIME values of the data.
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

        self.imported_values = None
        self.SHAP_values = None
        self.LIME_values = None

    def __str__(self):
        """
        Function that allows to print the Xplainer object.
        """
        print("Xplainer object")

    def startGui(self,
                explanation: str | pd.DataFrame = "SHAP",
                X_all: pd.DataFrame = None,
                default_projection: str = "PaCMAP",
                sub_models: list = None,
                save_regions: list = None,
                display = True):
        """
        Function that starts the interface.

        Parameters
        ---------
        explanation : str or pandas dataframe
            The type of explanation to display.
            If not computed : string ("SHAP" or "LIME", default : "SHAP")
            If already computed : pandas dataframe containing the explanations
        X_all : pandas dataframe
            The dataframe containing the data to explain.
        default_projection : str
            The default projection method used to display the data.
            The possible values are "PaCMAP" and "UMAP".
        sub_models : list
            The list of the submodels used to explain the data.
        save_regions : list
            The list of the regions saved by the user.
        display : bool
            If True, the interface is displayed.
            If False, the interface is not displayed.

        Returns
        -------
        Gui object
            The Gui object that contains the interface. Displayed if display = True.
        """
        if type(explanation) != str:
            self.imported_values = explanation
        self.gui = Gui(self, explanation, X_all, default_projection, sub_models, save_regions)
        if display:
            self.gui.display()