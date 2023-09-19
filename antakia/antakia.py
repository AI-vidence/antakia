import pandas as pd
# import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import ensemble


from antakia.data import Dataset, ExplanationsDataset, ExplanationMethod, Model, DimReducMethod
from antakia.potato import Potato
from antakia.gui import GUI

import ipywidgets as widgets
from IPython.display import display
import ipyvuetify as v

import logging
from log_utils import OutputWidgetHandler
logger = logging.getLogger(__name__)
handler = OutputWidgetHandler()
handler.setFormatter(logging.Formatter('antakia.py [%(levelname)s] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
handler.clear_logs()
handler.show_logs()

class AntakIA():
    """
    AntakIA class.

    Antakia instances provide data and methods to explain a ML model.

    Instance attributes
    -------------------
    __dataset : Dataset 
        The Datase t object containing data and projected data
    __explainDataset : ExplanationsDataset
        The ExplanationsDataset object containing explained values and projected explained values
    __model : Model
        the model to explain
    __gui : an instance of the GUI class
        In charge of the user interface
    __regions : List
        The list of the regions defined by the user. Regions are of type Potato.
    __backups: dict
        A list of saved regions # TODO to be explained. Rather unclear for me for now
    __backups_path: str
    __sub_models: list

    """


    def __init__(self, dataset: Dataset, model : Model, explainDataset : ExplanationsDataset = None, backups: dict = None, backups_path: str = None):
        '''
        Constructor of the class AntakIA.

        Parameters
        ----------
        dataset : Dataset object
        model : Model object
        explainDataset : ExplanationsDataset object, defauly to None (no computation yet)
        backups: dict of backups
        backups_path: str TODO : expliquer
        '''

        self.__dataset = dataset
        self.__model = model
        self. __explainDataset = explainDataset # Defaults to None
        self.__regions = [] #empty list of Potatoes
        self.__gui = None
        self.__backups_path = backups_path

        # TODO : understand ths saves thing
        if backups is not None:
            self.__backups = backups 
        elif backups_path is not None:
            self.__backups = utils.loadBackup(self, backups_path)
        else:
            self.__backups = []

        # TODO : compute Y_pred here

        self.__sub_models =  [linear_model.LinearRegression(), RandomForestRegressor(random_state=9), ensemble.GradientBoostingRegressor(random_state=9)]




    def startGUI(self, defaultProjection: int = DimReducMethod.PaCMAP) -> GUI :
        """
        Function that instantiates the GUI and calls its display() function.
        For more information, please see the documentation of the class GUI.

        Parameters
        ---------
        defaultProjection : int
            The default projection to use. The possible values are DimensionalityReduction.PacMAP, PCA, t-SNE and UMAP.
        sub_models : list #TODO why should they be passed to create the GUI ?
            The list of the sub_models to choose from for each region. The only constraint is that sub_models must have a predict method.
        display : bool
            If True, the interface is displayed. Else, You can access the interface with the attribute gui of the class.
        """
        self.__gui = GUI(self.__dataset, self.__model, self.__explainDataset)

        logger.debug("Just created GUI")

        return self.__gui

# ========= Getters  ===========

    def getGUI(self) -> GUI:
        """
        Function that returns the GUI object.

        Returns
        -------
        GUI object
            The GUI object.
        """
        return self.__gui

    def getRegions(self) -> list:
        """
        Function that returns the list of the regions computed by the user.

        Returns
        -------
        list
            The list of the regions computed. A region is a list of AntakIA objects, named `Potato`.
        """
        return self.__regions
    
    def resetRegions(self):
        """
        Function that resets the list of the regions computed by the user.
        """
        self.__regions = []
    
    def getBackups(self) -> list:
        """
        Function that returns the list of the saves.

        Returns
        -------
        list
            The list of the saves. A save is a list of regions.
        """
        return self.__backups

    
    def getDataset(self) -> Dataset:
        """
        Returns the Dataset object containing the data and their projected values.

        Returns
        -------
        Dataset object
            The Dataset object.
        """
        return self.__dataset
    
    def getExplainationDataset(self) -> ExplanationsDataset:
        """
        Returns the ExplanationsDataset object containing the explained data and their projected values.

        Returns
        -------
        ExplanationDataset object
            The ExplanationsDataset object.
        """
        return self.__explainDataset

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
        return self.__model

    def getSubModels(self) -> list:
        return self.__sub_models 


