import pandas as pd
import numpy as np

# TODO : these references to IPython should be removed in favor of a new scheme (see Wiki)
import ipyvuetify as v
import ipywidgets as widgets
from IPython.display import display 

from sklearn.preprocessing import StandardScaler

from abc import ABC, abstractmethod
from typing import Tuple
import time
from copy import deepcopy
from pubsub import pub


import logging
from log_utils import OutputWidgetHandler
logger = logging.getLogger(__name__)
handler = OutputWidgetHandler()
handler.setFormatter(logging.Formatter('data.py [%(levelname)s] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
handler.clear_logs()
handler.show_logs()

class Model(ABC) :

    @abstractmethod
    def predict(self, x:pd.DataFrame ) -> pd.Series:
        pass

    @abstractmethod
    def score(self, X:pd.DataFrame, y : pd.Series ) -> float :
        pass


class LongTask(ABC):
    '''
    Abstract class to compute long tasks, often in a separate thread.

    Attributes      
    ----------
    _longTaskType : int
        Can be LongTask.EXPLAINATION or LongTask.DIMENSIONALITY_REDUCTION   
    _X : dataframe
    _X_all : dataframe
    _progress : int
        The progress of the long task, between 0 and 100.
    '''

    # Class attributes : LongTask types
    EXPLAINATION = 1
    DIMENSIONALITY_REDUCTION = 2

    def __init__(self, longTaskType : int, X : pd.DataFrame, X_all: pd.DataFrame = None) :
        if not LongTask.isValidLongTaskType(longTaskType) :
            raise ValueError(longTaskType, " is a bad long task type")
        if X is None :
                raise ValueError("You must provide a dataframe for a LongTask")
        self._longTaskType = longTaskType
        self._X = X
        self._X_all = X_all
        self._progress = 0


    def getX(self) -> pd.DataFrame :
        return self._X

    def getX_all(self) -> pd.DataFrame :
        return self._X_all
    
    def getProgress(self) -> int:
        return self._progress

    def setProgress(self, prog:int) :
        self._progress = prog

    @staticmethod
    def isValidLongTaskType(type: int) -> bool:
        """
        Returns True if the type is a valid LongTask type.
        """
        return type == LongTask.EXPLAINATION or type == LongTask.DIMENSIONALITY_REDUCTION

    @abstractmethod
    def getTopic(self) -> str:
        """Required by pubssub.
        Identifier for the long task being computed.

        Returns:
            str: a Topic for pubsub
        """
        pass
    
    def getLongTaskType(self) -> int:
        return self._longTaskType
    
    @abstractmethod
    def compute(self) -> pd.DataFrame:
        """
        Method to compute the long task and update listener with the progress.
        """
        pass

    def sendProgress(self) :
        """
        Method to send the progress of the long task.

        Parameters
        ----------
        progress : int
            An integer between 0 and 100.
        """
        pub.sendMessage(self.getTopic(), progress=self._progress)


class ExplanationMethod(LongTask):
    """
    Abstract class (see Long Task) to compute explaination values for the Explanation Space (ES)

    Attributes
    _model : Model to explain
    _explainationType : SHAP or LIME
    """

    # Class attributes types
    NONE=-1
    SHAP = 0
    LIME = 1
    OTHER = 2 

    # TODO : we could use a config file ?
    DEFAULT_EXPLANATION_METHOD = SHAP

    def __init__(self, explainationType : int, X  : pd.DataFrame, X_all : pd.DataFrame, model : Model = None, userComputedValuesForProjection : bool = True) :
        # TODO : do wee need X_all ?
        super().__init__(LongTask.EXPLAINATION,  X, X_all)
        self._model = model
        self._explainationType = explainationType
        
    def getModel(self) -> Model:
        return self._model

    @staticmethod
    def isValidExplanationType(type:int) -> bool:
        """
        Returns True if the type is a valid explanation type.
        """
        return type == ExplanationMethod.SHAP or type == ExplanationMethod.LIME or type == ExplanationMethod.OTHER

    def getTopic(self) -> str:
        """Required by pubssub.
        Identifier for the long task being computed.

        Returns:
            str: a Topic for pubsub
        """
        return str(LongTask.EXPLAINATION) + "/" + str(self._explainationType)


    @staticmethod
    def getExplanationMethodsAsList() -> list :
        return [ExplanationMethod.SHAP, ExplanationMethod.LIME]
    
    @staticmethod
    def getExplanationMethodAsStr(type : int) -> str :
        if type == ExplanationMethod.SHAP :
            return "SHAP"
        elif type == ExplanationMethod.LIME :
            return "LIME"
        elif type == ExplanationMethod.OTHER :
            return "OTHER"
        elif type == ExplanationMethod.NONE :
            return "NONE"
        else :
            raise ValueError(type," is a bad explaination type")

class DimReducMethod(LongTask):
    # TODO : do we need XAll ?
    """
    Class that allows to reduce the dimensionality of the data.

    Attributes
    ----------
    _dimReducType : int, can be PCA, TSNE etc.
    _dimension : int
        Dimension reduction methods require a dimension parameter
        We store it in the abstract class
    _X_all : pd.DataFrame
    """

    NONE = 0
    PCA = 1
    TSNE = 2
    UMAP = 3
    PaCMAP = 4

    DIM_ALL = -1
    DIM_TWO = 2
    DIM_THREE = 3

    def __init__(self, dimReducType : int , dimension : int, X : pd.DataFrame, X_all : pd.DataFrame) :
        """
        Constructor for the DimReducMethod class.

        Parameters
        ----------
        longTaskType : int
            need by LongType consturctor
        X, X_all : pd.DataFrame
            idem
        dimension : int
            Dimension reduction methods require a dimension parameter
            We store it in the abstract class
        
    """
        self._dimReducType = dimReducType
        self._dimension = dimension

        LongTask.__init__(self, LongTask.DIMENSIONALITY_REDUCTION, X, X_all)

    def getDimReducType(self) -> int:
        return self._dimReducType

    def getDimension(self) -> int:
        return self._dimension

    @staticmethod
    def getDimensionAsStr(dim) -> str:
        if dim == DimReducMethod.DIM_ALL :
            return "ALL"
        elif dim == DimReducMethod.DIM_TWO :
            return "2D"
        elif dim == DimReducMethod.DIM_THREE :
            return "3D"
        else :
            return None
    
    def getTopic(self) -> str:
        """Required by pubssub.
        Identifier for the long task being computed.

        Returns:
            str: a Topic for pubsub
        """
        return str(LongTask.DIMENSIONALITY_REDUCTION) + "/" + str(self._dimReducType)+ "/" + str(self._dimension)

    @staticmethod
    def getDimReducMehtodAsStr(type : int) -> str :
        if type == DimReducMethod.PCA :
            return "PCA"
        elif type == DimReducMethod.TSNE :
            return "t-SNE"
        elif type == DimReducMethod.UMAP :
            return "UMAP"
        elif type == DimReducMethod.PaCMAP :
            return "PaCMAP"
        else :
            return None

    @staticmethod
    def getDimReducMhdsAsList() -> list :
        return [DimReducMethod.PCA, DimReducMethod.TSNE, DimReducMethod.UMAP, DimReducMethod.PaCMAP]

    @staticmethod
    def getDimReducMhdsAsStrList() -> list :
        list = []
        for i in range(5):
            item = DimReducMethod.getDimReducMehtodAsStr(i)
            if item is not None :
                list.append(item)    
        return list

    @staticmethod
    def isValidDimReducType(type:int) -> bool:
        """
        Returns True if the type is a valid dimensionality reduction type.
        """
        return type == DimReducMethod.PCA or type == DimReducMethod.TSNE or type == DimReducMethod.UMAP or type == DimReducMethod.PaCMAP or type == DimReducMethod.NONE

    @staticmethod
    def isValidDimNumber(type:int) -> bool:
        """
        Returns True if the type is a valid dimension number.
        """
        return type == DimReducMethod.DIM_ALL or type == DimReducMethod.DIM_TWO or type == DimReducMethod.DIM_THREE


# ================================================


class Dataset():
    """
    Dataset class.
    
    Instance attributes
    ------------------
    _X  : pandas.Dataframe
        The dataframe to be used by AntakIA
    _X_proj : dict :
        - key DimReducMethod.PCA : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PCA-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PCA-projected X values
        - key DimReducMethod.TSNE : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D TSNE-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D TSNE-projected X values
        - key DimReducMethod.UMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D UMAP-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D UMAP-projected X values
        -  key DimReducMethod.PaCMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PaCMAP-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PaCMAP-projected X values
    _X_all: pandas.Dataframe
        The entire dataframe. X may be smaller than Xall if the frac method has been used.
    _X_scaled : pandas.Dataframe
        The dataframe with normalized (scaled) values.
    _y : pandas.Series
        Target values
    _y_pred : pandas.Series
        The Serie containing the predictions of the model. Computed at construction time.
    _comments : List of str
        The comments associated to each variable in X
    _sensible : List of bool
        If True, a warning will be displayed when the feature is used in the explanations. More to come in the future.
    _lat : str
        The name of the latitude column if it exists.
        #TODO use a specific object for lat/long ?
    _long : str
        The name of the longitude column if it exists.
        #TODO idem
    """

    # Class attributes for X values
    CURRENT = 1
    ALL = 2
    SCALED = 3

    # Class attributes for Y values
    TARGET = 0
    PREDICTED = 1


    def __init__(self, X:pd.DataFrame = None, csv:str = None, y:pd.Series = None):
        """
        Constructor of the class Dataset.
        
        Parameters
        ----------
        X : pandas.Dataframe
            The dataframe containing the dataset provided by the user.
        csv : str (optional)
            The path to the csv file containing the dataset.
            #TODO : we shoudl use a Path or File object
        y : pandas.series (optional)
            The series containing the target values.
            TODO : is it compulsory ?
        """

        if X is None and csv is None :
            raise ValueError("You must provide a dataframe or a CSV file")
        if X is not None and csv is not None :
            raise ValueError("You must provide either a dataframe or a CSV file, not both")
        if X is not None :
            self._X = X
        else :
            self._X = pd.read_csv(csv)

        self._X_proj = {} # Empty dict
        self._explanations = None

        # We remove spaces in the column names
        self._X.columns = [self._X.columns[i].replace(" ", "_") for i in range(len(self._X.columns))]
        self._X = self._X.reset_index(drop=True)

        self._X_all = X
        self._y = y
        self._y_pred = None # TODO : could be included in the dataset ?
        self._X_scaled = pd.DataFrame(StandardScaler().fit_transform(X))
        self._X_scaled.columns = X.columns

        self._comments = [""]*len(self._X.columns) 
        self._sensible = [False]*len(self._X.columns)

        self._fraction = 1 # TODO : what is this ?
        self._frac_indexes = self._X.index #

        # # TODO : should be handled with a GeoData object ?
        self._long, self.__lat = None, None # TODO : shoudl only  be used if needed

        for name in ['longitude', 'Longitude', 'Long', 'long']:
            if name in self._X.columns:
                self._long = name

        for name in ['latitude', 'Latitude', 'Lat', 'lat']:
            if name in self._X.columns:
                self._lat = name


    def __str__(self):
        text = "Dataset object :\n"
        text += "------------------\n"
        text += "- Number of observations:"  + str(self._X.shape[0]) + "\n"
        text += "- Number of variables: " + str(self._X.shape[1]) + "\n"
        return text
    
    # TODO : is it useful ?
    def __len__(self):
        return self._X.shape[0]

    def getXValues(self, flavour : int = CURRENT) -> pd.DataFrame:
        """
        Access X values for the dataset as a DataFrame.

        Parameters
        ----------
        flavour : int
            The flavour of the X values to return. Must be Dataset.CURRENT, Dataset.ALL or Dataset.SCALED.

        Returns
        -------
        pd.DataFrame :
            The X values for the given flavour
        """    
        if flavour == Dataset.CURRENT :
            return self._X
        elif flavour == Dataset.ALL :
            return self._X_all
        elif flavour == Dataset.SCALED :
            return self._X_scaled
        else :
            raise ValueError("Bad flavour value")
        
    def isValidXFlavour (flavour : int) -> bool:
        """
        Returns True if the flavour is valid, False otherwise.
        """
        return flavour == Dataset.CURRENT or flavour == Dataset.ALL or flavour == Dataset.SCALED


    def isValidYFlavour(flavour : int) -> bool:
        """
        Returns True if the flavour is valid, False otherwise.
        """
        return flavour == Dataset.TARGET or flavour == Dataset.PREDICTED

    def getXProjValues(self, dimReducMethodType:int, dimension:int = DimReducMethod.DIM_TWO) -> pd.DataFrame:
        """Returns de projected X values using a dimensionality reduction method and target dimension (2 or 3)

        Args:
            dimReducMethodType (int): the type of dimensionality reduction method
            dimension (int, optional): Defaults to DimReducMethod.DIM_TWO.

        Returns:
            pd.DataFrame: the projected X values. May be None 
        """
        if not DimReducMethod.isValidDimReducType(dimReducMethodType) :
                raise ValueError("Bad dimensionality reduction type")
        
        if not DimReducMethod.isValidDimNumber(dimension) :
                raise ValueError("Bad dimension number")

        return self._X_proj[dimReducMethodType][dimension]
    

    def setXProjValues(self, dimReducMethodType:int, dimension:int, values: pd.DataFrame) :
        """Set X_proj alues for this dimensionality reduction and  dimension."""

        # TODO we may want to check values.shape and raise value error if it does not match
        if not DimReducMethod.isValidDimReducType(dimReducMethodType) :
                raise ValueError("Bad dimensionality reduction type")
        if not DimReducMethod.isValidDimNumber(dimension) :
                raise ValueError("Bad dimension number")
        
        if dimReducMethodType not in self._X_proj :
            self._X_proj[dimReducMethodType] = {} # We create a new dict for this dimReducMethodType
        if dimension not in self._X_proj[dimReducMethodType] :
            self._X_proj[dimReducMethodType][dimension] = {} # We create a new dict for this dimension
        self._X_proj[dimReducMethodType][dimension] = values
        #  TODO We could log this assignment


    def getYValues(self, flavour : int = TARGET) -> pd.Series:
        """
        Returns the y values of the dataset as a Series, depending on the flavour.
        """
        if flavour == Dataset.TARGET :
            return self._y
        elif flavour == Dataset.PREDICTED :
            return self._y_pred
        else :
            raise ValueError("Bad flavour value for Y")


    def getShape(self):
        """ Returns the shape of the used dataset"""
        return self._X.shape
    
    # def frac(self, p:float):
    #     """
    #     Reduces the dataset to a fraction of its size.

    #     Parameters
    #     ---------
    #     p : float
    #         The fraction (%) of the dataset to keep.
    #     """

    #     self.X = self.X_all.sample(frac=p, random_state=9)
    #     self.frac_indexes = deepcopy(self.X.index)
    #     self.X_scaled = self.X_scaled.iloc[self.frac_indexes].reset_index(drop=True)
    #     self.y_pred = self.y_pred.iloc[self.frac_indexes].reset_index(drop=True)
    #     if self.y is not None:
    #         self.y = self.y.iloc[self.frac_indexes].reset_index(drop=True)
    #     self.fraction = p
    #     self.X.reset_index(drop=True, inplace=True)

    def setLatLon(self, lat:str, long:str) :
        """
        Sets the longitude and latitude columns of the dataset.

        Parameters
        ---------
        long : str
            The name of the longitude column.
        lat : str
            The name of the latitude column.
        """
        self._lat = lat
        self._long = long

    def getLatLon(self) -> (float, float):
        """
        Returns the longitude and latitude columns of the dataset.

        Returns
        -------
        lat : str
            The name of the latitude column.
        long : str
            The name of the longitude column.
        """
        return (self._lat, self._long)

    # def improve(self):
    #     """
    #     Improves the dataset. 

    #     # TODO : shoudl be in the gui module

    #     Displays a widget to modify the dataset. For each feature, you can change its name, its type, its comment and if it is sensible or not.

    #     You also have the access to the general informations of the dataset.
    #     """
    #     general_infos = v.Row(class_="ma-2", children=[
    #         v.Icon(children=["mdi-database"], size="30px"),
    #         v.Html(tag="h3", class_="mb-3 mt-3 ml-4", children=[
    #             str(self.X.shape[0]) + " observations, " + str(self.X.shape[1]) + " features"
    #             ])])
    #     liste_slides = []
    #     for i in range(self.X.shape[1]):
    #         infos = [min(self.X.iloc[:,i]), max(self.X.iloc[:,i]), np.mean(self.X.iloc[:,i]), np.std(self.X.iloc[:,i])]
    #         infos = [round(infos[j], 3) for j in range(len(infos))]
    #         liste_slides.append(guiFactory.create_slide_dataset(self.X.columns[i], i+1, self.X.dtypes[i], len(self.X.columns), self.comments[i], self.sensible[i], infos))

    #     slidegroup = v.SlideGroup(
    #         v_model=None,
    #         class_="ma-3 pa-3",
    #         elevation=4,
    #         center_active=True,
    #         show_arrows=True,
    #         children=liste_slides,
    #     )

    #     def changement_sensible(widget, event, data):
    #         i = int(widget.class_)-1
    #         if widget.v_model :
    #             liste_slides[i].children[0].color = "red lighten-5"
    #             self.sensible[i] = True
    #         else:
    #             liste_slides[i].children[0].color = "white"
    #             self.sensible[i] = False

    #     def changement_names(widget, event, data):
    #         i = widget.value-1
    #         self.X = self.X.rename(columns={self.X.columns[i]: widget.v_model})

    #     def changement_type(widget, event, data):
    #         i = widget.value-1
    #         widget2 = liste_slides[i].children[0].children[-1].children[1].children[0]
    #         try :
    #             self.X = self.X.astype({self.X.columns[i]: widget2.v_model})
    #         except:
    #             print("The type of the column " + self.X.columns[i] + " cannot be changed to " + widget2.v_model)
    #             widget.color = "error"
    #             time.sleep(2)
    #             widget.color = ""
    #         else:
    #             widget.color = "success"
    #             time.sleep(2)
    #             widget.color = ""

    #     def changement_comment(widget, event, data):
    #         i = widget.value-1
    #         self.comments[i] = widget.v_model

    #     for i in range(len(liste_slides)):
    #         liste_slides[i].children[0].children[-1].children[2].on_event("change", changement_sensible)
    #         liste_slides[i].children[0].children[-1].children[3].on_event("change", changement_comment)
    #         liste_slides[i].children[0].children[0].children[0].on_event("change", changement_names)
    #         liste_slides[i].children[0].children[-1].children[1].children[-1].on_event("click", changement_type)

    #     widget = v.Col(children=[
    #         general_infos,
    #         slidegroup,
    #     ])
    #     display(widget)

# =============================================================================


class ExplanationsDataset():
    """
    ExplanationsDataset class.

    An Explanations object holds the explanations values of the model.
    It is the "explained version" of a Dataset object.

    Instance Attributes
    --------------------
    _shapValues : dict
        - key IMPORTED : a pandas Dataframe containing the SHAP values provided by the user.
        - key COMPUTED : a pandas Dataframe containing the SHAP values computed by AntakIA.
        - key DimReducMethod.PCA : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PCA-projected SHAP values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PCA-projected SHAP values
        - key DimReducMethod.TSNE : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D TSNE-projected SHAP values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D TSNE-projected SHAP values
        - key DimReducMethod.UMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D UMAP-projected SHAP values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D UMAP-projected SHAP values
        -  key DimReducMethod.PaCMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PaCMAP-projected SHAP values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PaCMAP-projected SHAP values
    _limeValues : dict
        - key IMPORTED : a pandas Dataframe containing the LIME values provided by the user.
        - key COMPUTED : a pandas Dataframe containing the LIME values computed by AntakIA.
        - key DimReducMethod.PCA : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PCA-projected LIME values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PCA-projected LIME values
        - key DimReducMethod.TSNE : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D TSNE-projected LIME values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D TSNE-projected LIME values
        - key DimReducMethod.UMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D UMAP-projected LIME values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D UMAP-projected LIME values
        -  key DimReducMethod.PaCMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PaCMAP-projected LIME values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PaCMAP-projected LIME values

        """
    
    
    # Class attributes
    # Characterizes an AntaKIA-computed Dataframe of explainations
    COMPUTED = -1
    # Characterizes an user-provided Dataframe of explainations
    IMPORTED = -2 
    # We have both computed and imported values
    BOTH = -3


    def __init__(self, values : pd.DataFrame, explanationType:int):
        """
        Constructor of the class ExplanationsDataset.
        
        Parameters :
        ------------
        values : pandas.Dataframe
            The dataframe containing explanations values provided by the user (IMPORTED)
        explanationType : int
            Must be ExplainedValues.SHAP or ExplainedValues.LIME
        """


        if explanationType==ExplanationMethod.SHAP :
            self._shapValues = {self.IMPORTED : values}
            self._limeValues = {self.IMPORTED : None}
        elif explanationType==ExplanationMethod.LIME :
            self._shapValues = {self.IMPORTED : None}
            self._limeValues = {self.IMPORTED : values}
        else :
            raise ValueError("explanationType must be ExplainedValues.SHAP or ExplainedValues.LIME")

    def getValues(self, explainationMethodType:int, dimReducMethodType:int=DimReducMethod.NONE, dimension:int = DimReducMethod.DIM_ALL, onlyImported : bool = False) -> pd.DataFrame:
        """
        Returns the explanations values of the ExplanationsDataset.

        Parameters
        ----------
        explainationMethodType : int
            Must be ExplanationMethod.SHAP or ExplanationMethod.LIME
        DimReducMethodType : int
            Must be DimReducMethod.PCA, DimReducMethod.TSNE, DimReducMethod.UMAP or DimReducMethod.PaCMAP. Can be none if we want the whole dataframe.
        dimension : int 
            The dimension of the projection (2 or 3). Can be None or DIM_ALL

        Returns
        -------
        values : pandas.Dataframe
            The values for this explainationMethodType, DimReducMethodType and dimension.
            Returns None if not computed yet.
        """

        if not DimReducMethod.isValidDimReducType(dimReducMethodType) :
            raise ValueError(dimReducMethodType," is a bad dimensionality reduction type")
        if not DimReducMethod.isValidDimNumber(dimension) :
            raise ValueError(dimension, " is a bad dimension number")
        

        if explainationMethodType==ExplanationMethod.SHAP :
            if (dimension == DimReducMethod.DIM_ALL or dimension is None) or dimReducMethodType is None :
                # Our caller is not interested in SHAP projected values
                if ExplanationsDataset.COMPUTED in self._shapValues and not onlyImported :
                    return self._shapValues[ExplanationsDataset.COMPUTED] # We return computed values first
                elif ExplanationsDataset.IMPORTED in self._shapValues : 
                    return self._shapValues[ExplanationsDataset.IMPORTED] # we return user-provided values if no computed values
                else :
                    return None  # We have not SHAP (computed or imported) values yet for all dimensions
            else : # Our caller wants SHAP projected values
                if dimReducMethodType not in self._shapValues :
                    return None # No SHAP projection computed yet for this DimReducMethodType (2 or 3D)
                else : # We do have projected values for SHAP explainations. Let's see if we have the right dimension
                    if dimension == DimReducMethod.DIM_TWO :
                        if DimReducMethod.DIM_TWO in self._shapValues[dimReducMethodType] and self._shapValues[dimReducMethodType][DimReducMethod.DIM_TWO] is not None :
                            return self._shapValues[dimReducMethodType][DimReducMethod.DIM_TWO]
                        else :
                            # No 2D projection computed yet for these SHAP values, and this DimReducMethodType
                            return None
                    else : # So dimension == DimReducMethod.DIM_THREE
                        if DimReducMethod.DIM_THREE in self._shapValues[dimReducMethodType] and self._shapValues[dimReducMethodType][DimReducMethod.DIM_THREE] is not None :
                            return self._shapValues[dimReducMethodType][DimReducMethod.DIM_THREE]
                        else :
                            # No 3D projection computed yet for these SHAP values, and this DimReducMethodType
                            return None
        else : # So our caller wants LIME values
             if (dimension == DimReducMethod.DIM_ALL or dimension is None) or dimReducMethodType is None :
                # Our caller is not interested in LIME projected values
                if ExplanationsDataset.COMPUTED in self._limeValues and not onlyImported:
                    return self._limeValues[ExplanationsDataset.COMPUTED] # We return computed values first
                elif ExplanationsDataset.IMPORTED in self._limeValues : 
                    return self._limeValues[ExplanationsDataset.IMPORTED] # we return user-provided values if no computed values
                else :
                    return None  # We have not LIME (computed or imported) values yet for all dimensions



        if explainationMethodType==ExplanationMethod.LIME :# Our caller wants LIME projected values
                if dimReducMethodType not in self._limeValues :
                    return None # No LIME projection computed yet for this DimReducMethodType
                else : # We do have projected values for LIME explainations. Let's see if we have the right dimension
                    if dimension == DimReducMethod.DIM_TWO :
                        if DimReducMethod.DIM_TWO in self._limeValues[dimReducMethodType] and self._limeValues[dimReducMethodType][DimReducMethod.DIM_TWO] is not None :
                            return self._limeValues[dimReducMethodType][DimReducMethod.DIM_TWO]
                        else :
                            # No 2D projection computed yet for these LIME values, and this DimReducMethodType
                            return None
                    else :  # So dimension == DimReducMethod.DIM_THREE
                        if  DimReducMethod.DIM_THREE in self._limeValues[dimReducMethodType] and self._limeValues[dimReducMethodType][DimReducMethod.DIM_THREE] is not None :
                            return self._limeValues[dimReducMethodType][DimReducMethod.DIM_THREE]
                        else :
                            # No 3D projection computed yet for these SHAP values, and this DimReducMethodType
                            return None


    def setValues(self, explainationMethodType:int, values: pd.DataFrame, dimReducMethodType:int=DimReducMethod.NONE, dimension:int=DimReducMethod.DIM_ALL) :
        """Set values for this ExplanationsDataset, given an explanation method, a dimensionality reduction and a dimension.
        Indeed, the values in explainations has been CUMPUTED by Antakia. It can't be IMPORTED.

        Args:
            explainationMethodType : int
                SHAP, LIME or NONE (see ExplanationMethod class in compute.py)
            values : pd.DataFrame
                The values to set.
            dimReducMethodTypev : int
                Type of dimensuion reduction. Can be None if values are not projected.
            dimension (int, optional): dimension of projection. Can be None or DIM_ALL if values are not projected.
        """

        if not DimReducMethod.isValidDimReducType(dimReducMethodType) :
                raise ValueError(dimReducMethodType," is a bad dimensionality reduction type")
        
        if not DimReducMethod.isValidDimNumber(dimension) :
                raise ValueError(dimension, " is a bad dimension number")

        if explainationMethodType==ExplanationMethod.SHAP :
            if (dimension == DimReducMethod.DIM_ALL or dimension is None) or dimReducMethodType is None: 
                # Our caller wants to set SHAP values without any projection
                if ExplanationsDataset.COMPUTED not in self._shapValues :
                    self._shapValues[ExplanationsDataset.COMPUTED] = values
                self._shapValues[ExplanationsDataset.COMPUTED] = values
            # Our caller want to set projected SHAP values
            else :
                # We need to initialize the dict of dict if needed :
                if dimReducMethodType not in self._shapValues :
                    self._shapValues[dimReducMethodType] = {}
                if dimension not in self._shapValues[dimReducMethodType] :
                    self._shapValues[dimReducMethodType][dimension] = {}              
                self._shapValues[dimReducMethodType][dimension] = values
        elif explainationMethodType==ExplanationMethod.LIME :
            if (dimension == DimReducMethod.DIM_ALL or dimension is None) or dimReducMethodType is None: 
                # Our caller wants to set LIME values without any projection
                if ExplanationsDataset.COMPUTED not in self._limeValues :
                    self._limeValues[ExplanationsDataset.COMPUTED] = values
                self._limeValues[ExplanationsDataset.COMPUTED] = values
            # Our caller want to set projected LIME values
            else :
                # We need to initialize the dict of dict if needed :
                if dimReducMethodType not in self._limeValues :
                    self._limeValues[dimReducMethodType] = {}
                if dimension not in self._limeValues[dimReducMethodType] :
                    self._limeValues[dimReducMethodType][dimension] = {}              
                self._limeValues[dimReducMethodType][dimension] = values
        else :
            raise ValueError(explainationMethodType, " is bad explanantion method type")


    def isExplanationAvailable(self, explainationMethodType:int) ->  (bool, int) :
        """ Tells wether we have this explanation method values or not, and if it is imported or computed or both.s
            
        """
        if not ExplanationMethod.isValidExplanationType(explainationMethodType) :
            raise ValueError(explainationMethodType," is a bad explaination method type")

        if explainationMethodType==ExplanationMethod.SHAP :
            imported, computed = False, False

            if ExplanationsDataset.IMPORTED in self._shapValues and self._shapValues[ExplanationsDataset.IMPORTED] is not None :
                imported = True
            if ExplanationsDataset.COMPUTED in self._shapValues and self._shapValues[ExplanationsDataset.IMCOMPUTEDPORTED] is not None :
                computed = True
            
            returnValue = None

            if imported and computed :
                returnValue = ExplanationsDataset.BOTH
            elif imported :
                returnValue = ExplanationsDataset.IMPORTED
            elif computed :
                returnValue = ExplanationsDataset.COMPUTED
            else :
                returnValue = None

            return (True, returnValue)

        elif explainationMethodType==ExplanationMethod.LIME :
            imported, computed = False, False

            if ExplanationsDataset.IMPORTED in self._limeValues and self._limeValues[ExplanationsDataset.IMPORTED] is not None :
                imported = True
            if ExplanationsDataset.COMPUTED in self._limeValues and self._limeValues[ExplanationsDataset.IMCOMPUTEDPORTED] is not None :
                computed = True
            
            returnValue = None

            if imported and computed :
                returnValue = ExplanationsDataset.BOTH
            elif imported :
                returnValue = ExplanationsDataset.IMPORTED
            elif computed :
                returnValue = ExplanationsDataset.COMPUTED
            else :
                returnValue = None

            return (True, returnValue)
        else :
            return (False, None)

    def _descProjHelper(self, text : str, explainationMethodType:int, dimReducMethodType : int) -> str :
        if explainationMethodType==ExplanationMethod.SHAP :
            explainDict = self._shapValues
        elif explainationMethodType==ExplanationMethod.LIME :
            explainDict = self._limeValues
        else :
            raise ValueError(explainationMethodType," is a bad explaination method type")
        
        if not DimReducMethod.isValidDimReducType(dimReducMethodType) : 
            raise ValueError(dimReducMethodType," is a bad dimensionality reduction type")
        
        text += "OK , je regarde la proj " + DimReducMethod.getDimReducMehtodAsStr(dimReducMethodType) + "\n"

        if dimReducMethodType in explainDict and explainDict[dimReducMethodType] is not None :
                text +=  ExplanationMethod.getExplanationMethodAsStr(explainationMethodType) +" values dict has " + DimReducMethod.getDimReducMehtodAsStr(dimReducMethodType) + " projected values :\n"
                if DimReducMethod.DIM_TWO in explainDict[DimReducMethod.PCA] and explainDict[DimReducMethod.PCA][DimReducMethod.DIM_TWO] is not None :
                    text += "    2D :"  + str(explainDict[dimReducMethodType][DimReducMethod.DIM_TWO].shape[0]) + " observations, " + str(self._shapValues[dimReducMethodType][DimReducMethod.DIM_TWO].shape[1]) + " features\n"
                if DimReducMethod.DIM_THREE in explainDict[dimReducMethodType] and explainDict[dimReducMethodType][DimReducMethod.DIM_THREE] is not None :
                    text += "    3D :"  + str(explainDict[dimReducMethodType][DimReducMethod.DIM_THREE].shape[0]) + " observations, " + str(explainDict[dimReducMethodType][DimReducMethod.DIM_THREE].shape[1]) + " features\n" 

        return text

    def _descExplainHelper(self, text : str, explainationMethodType:int) -> str :
        if explainationMethodType==ExplanationMethod.SHAP :
            explainDict = self._shapValues
        elif explainationMethodType==ExplanationMethod.LIME :
            explainDict = self._limeValues
        else :
            raise ValueError(explainationMethodType," is a bad explaination method type")
        
        
        if explainDict is None or len(explainDict) == 0 :
            text += ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)+ " values dict is empty\n"
        else : 
            if ExplanationsDataset.IMPORTED in explainDict and explainDict[ExplanationsDataset.IMPORTED] is not None :
                text += ExplanationMethod.getExplanationMethodAsStr(explainationMethodType) + " values dict has imported values :\n"
                text += "    "  + str(explainDict[ExplanationsDataset.IMPORTED].shape[0]) + " observations, " + str(explainDict[ExplanationsDataset.IMPORTED].shape[1]) + " features\n"

            if ExplanationsDataset.COMPUTED in explainDict and explainDict[ExplanationsDataset.COMPUTED] is not None :
                text += ExplanationMethod.getExplanationMethodAsStr(explainationMethodType) + " values dict has computed values :\n"
                text += "    "  + str(explainDict[ExplanationsDataset.COMPUTED].shape[0]) + " observations, " +  + str(explainDict[ExplanationsDataset.COMPUTED].shape[1]) + " features\n"
        
        for projType in DimReducMethod.getDimReducMhdsAsList() :
            text +=  self._descProjHelper(text, explainationMethodType, projType)

        return text

    def __str__(self) -> str:  
        text = "ExplanationDataset object :\n"
        text += "---------------------------\n"
        
        list = ExplanationMethod.getExplanationMethodsAsList()
        text += list
        for explainType in ExplanationMethod.getExplanationMethodsAsList() :
            text += "J'appelle _descExplainHelper pour "+ ExplanationMethod.getExplanationMethodAsStr(explainType) + "\n"
            text += self._descExplainHelper(text, explainType)

        text += "Fin desc\n"

        return text