import pandas as pd
# import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import ensemble


from antakia.gui import GUI
from antakia.data import Dataset
from antakia.compute import DimensionalityReduction
from antakia.potato import Potato


import ipywidgets as widgets
from IPython.display import display
import ipyvuetify as v

class AntakIA():
    """
    AntakIA class.

    Antakia instances provide data and methods to explain a ML model.

    Instance attributes
    -------------------
    dataset : Dataset 
        The Dataset object containing data and the ML model to explain
    gui : an instance of the GUI class
        In charge of the user interface
    regions : List
        The list of the regions defined by the user. Regions are of type Potato.
    saves: dict
        A list of saved regions # TODO to be explained. Rather unclear for me for now
    saves_path: str
        #TODO : to be understand
    sub_models: list
        #TODO : to be understand
    widget : IPyWidget TODO : why ???
    """


    def __init__(self, dataset: Dataset, saves: dict = None, saves_path: str = None):
        '''
        Constructor of the class AntakIA.

        Parameters
        ----------
        dataset : Dataset object
        saves: dict TODO : expliquer 
        saves_path: str TODO : expliquer
        '''

        self.dataset = dataset
        self.regions = [] #empty list of Potatoes
        self.gui = None

        # TODO : understand ths saves thing
        if saves is not None:
            self.saves = saves 
        elif saves_path is not None:
            self.saves = utils.load_save(self, saves_path)
        else:
            self.saves = []

        self.sub_models =  [linear_model.LinearRegression(), RandomForestRegressor(random_state=9), ensemble.GradientBoostingRegressor(random_state=9)]

        # TODO : this should be in the GUI !!
        self.widget = None 


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



# A Protéger !!

