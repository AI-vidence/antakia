import pandas as pd
import numpy as np
import json

import antakia._gui as _gui


class Xplainer(_gui.Mixin):
    """
    Class that allows to create an Xplainer object.
    This object is the main object of the package antakia. It contains all the data and variables needed to run the interface (see antakia.interface).

    Variables
    ---------
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

        Parameter
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
        self.selection = []

    def __str__(self):
        """
        Function that allows to print the Xplainer object.
        """
        print("Xplainer object")
