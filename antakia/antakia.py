import pandas as pd

from antakia.data import Dataset, ExplanationDataset, ExplanationMethod, Model, DimReducMethod
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
    _X : a pd.DataFrame 
    _variables : a list of Variables
    _X_exp : a pd.DataFrame
    _Y : a pd.Series
    _model : Model
        the model to explain
    _gui : an instance of the GUI class
        In charge of the user interface
    _regions : List of Selection objects


    """

    def __init__(self, X: pd.DataFrame, y:pd.Series, model: Model, csv_file_name:str=None, explain_method: int = None):

        if X is None or not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if y is None or not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series")
        if model is None or not isinstance(model, Model):
            raise ValueError("model must be a Model object")
        
        self.X = X

        self.y = y
        self._model = model

        self._X_exp = None
        if csv_file_name is not None :
            with open(csv_file_name) as csv_file:
                self._X_exp = pd.read_csv(csv_file)


        
        self._regions = []
        self._gui = None

  

    def startGUI(self, ) -> GUI :
        self._gui = GUI(self._dataset, self._model, self._explainDataset)

        return self._gui

# ========= Getters  ===========

    def getGUI(self) -> GUI:
        """
        Function that returns the GUI object.

        Returns
        -------
        GUI object
            The GUI object.
        """
        return self._gui

    def getRegions(self) -> list:
        """
        Function that returns the list of the regions computed by the user.

        Returns
        -------
        list
            The list of the regions computed. A region is a list of AntakIA objects, named `Potato`.
        """
        return self._regions
    
    def resetRegions(self):
        """
        Function that resets the list of the regions computed by the user.
        """
        self._regions = []
    
    def getBackups(self) -> list:
        """
        Function that returns the list of the saves.

        Returns
        -------
        list
            The list of the saves. A save is a list of regions.
        """
        return self._backups

    
    def getDataset(self) -> Dataset:
        """
        Returns the Dataset object containing the data and their projected values.

        Returns
        -------
        Dataset object
            The Dataset object.
        """
        return self._dataset
    
    def getExplainationDataset(self) -> ExplanationDataset:
        """
        Returns the ExplanationDataset object containing the explained data and their projected values.

        Returns
        -------
        ExplanationDataset object
            The ExplanationDataset object.
        """
        return self._explainDataset

    def newRegion(self, potato: Potato):
        """
        Function that adds a region to the list of regions.

        Parameters
        ---------
        potato : Potato object
            The Potato object to add to the list of regions.
        """
        self.regions.append(potato)

    def getModel(self) -> Model :
        return self._model

    def getSubModels(self) -> list:
        return self._sub_models 


