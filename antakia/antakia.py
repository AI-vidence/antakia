import pandas as pd
# import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import ensemble


from antakia.gui import GUI
from antakia.data import Dataset
from antakia.compute import DimensionalityReduction, ExplainationMethod
from antakia.potato import Potato
from antakia.model import Model


import ipywidgets as widgets
from IPython.display import display
import ipyvuetify as v

class AntakIA():
    """
    AntakIA class.

    Antakia instances provide data and methods to explain a ML model.

    Instance attributes
    -------------------
    __dataset : Dataset 
        The Dataset object containing data and explained values
    __model :Model
            the model to explain
    __gui : an instance of the GUI class
        In charge of the user interface
    __regions : List
        The list of the regions defined by the user. Regions are of type Potato.
    __backups: dict
        A list of saved regions # TODO to be explained. Rather unclear for me for now
    __backups_path: str
        #TODO : to be understand
    __sub_models: list
        #TODO : to be understand
    """

    # Class constants 
    # TODO : we could use a config file ?
    DEFAULT_EXPLANATION_METHOD = ExplainationMethod.SHAP



    def __init__(self, dataset: Dataset, model : Model, backups: dict = None, backups_path: str = None):
        '''
        Constructor of the class AntakIA.

        Parameters
        ----------
        dataset : Dataset object
        backups: dict TODO : expliquer 
        backups_path: str TODO : expliquer
        '''

        self.__dataset = dataset
        self.__model = model
        self.__regions = [] #empty list of Potatoes
        self.__gui = None
        self.__backups_path = backups_path

        # TODO : understand ths saves thing
        if saves is not None:
            self.__backups = backups 
        elif backups_path is not None:
            self.__backups = utils.load_save(self, backups_path)
        else:
            self.__saves = []

        self.__sub_models =  [linear_model.LinearRegression(), RandomForestRegressor(random_state=9), ensemble.GradientBoostingRegressor(random_state=9)]




    def startGUI(self, defaultProjection: int = DimensionalityReduction.PacMAP, display = True):
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
        self.gui = GUI(self, defaultProjection) 

        if display:
            self.gui.display()

# ========= Getters  ===========

    def getGUI(self) -> GUI:
        """
        Function that returns the GUI object.

        Returns
        -------
        GUI object
            The GUI object.
        """
        return self.gui

    def getRegions(self) -> list:
        """
        Function that returns the list of the regions computed by the user.

        Returns
        -------
        list
            The list of the regions computed. A region is a list of AntakIA objects, named `Potato`.
        """
        return self.regions
    
    def resetRegions(self):
        """
        Function that resets the list of the regions computed by the user.
        """
        self.regions = []
    
    def getSaves(self) -> list:
        """
        Function that returns the list of the saves.

        Returns
        -------
        list
            The list of the saves. A save is a list of regions.
        """
        return self.saves

    
    def getDataset(self) -> Dataset:
        """
        Function that returns the Dataset object containing the data, the ML medel to explain
        and the explanatory values.

        Returns
        -------
        Dataset object
            The Dataset object.
        """
        return self.dataset
    
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
    

# A Protéger !!

