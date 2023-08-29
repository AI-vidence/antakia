import pandas as pd
import numpy as np

# TODO : these references to IPython should be removed in favor of a new scheme (see Wiki)
import ipyvuetify as v
import ipywidgets as widgets
from IPython.display import display 

from sklearn.preprocessing import StandardScaler

from antakia.compute import ExplainationMethod, DimensionalityReduction
from antakia import gui_elements # TODO : why ?


import time

from copy import deepcopy

class Dataset():
    """
    Dataset class.
    
    Instance attributes
    ------------------
    X  : pandas.Dataframe
        The dataframe to be used by AntakIA
    Xall: pandas.Dataframe
        The entire dataframe. X may be smaller than Xall if the frac method has been used.
    X_scaled : pandas.Dataframe
        The dataframe with normalized (scaled) values.
    Xproj : dict
        The dictionnary containing Dataframes with the projected values of the dataset
        #TODO Understand how this dict works
    model : ????
        The "black-box" ML model to explain.
    y_pred : pandas.Series
        The Serie containing the predictions of the model. Computed at construction time.
    explanations : ExplanationDataset
        See the class ExplanationDataset below
    comments : List of str
        The comments associated to each variable in X
    sensible : List of bool
        If True, a warning will be displayed when the feature is used in the explanations. More to come in the future.
    lat : str
        The name of the latitude column if it exists.
        #TODO use a specific object for lat/long ?
    long : str
        The name of the longitude column if it exists.
        #TODO idem
    """

    def __init__(self, X:pd.DataFrame = None, csv:str = None, y:pd.Series = None, model = None, user_explanations:pd.DataFrame = None, user_explanationa_type:int = None):
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
        model : (optional) TODO wich type is it ? TODO : how is it it's optional ?
            The "black-box" ML model to explain. The model must have a predict method.
            TODO : write a Model class ?
        user_explanations: dict (optional)
            Explanations provided by the user, as a dictionary.
        """

        if X is None and csv is None :
            raise ValueError("You must provide a dataframe or a CSV file")
        if X is not None and csv is not None :
            raise ValueError("You must provide either a dataframe or a CSV file, not both")
        if X is not None :
            self.X = X
        else :
            self.X = pd.read_csv(csv)

        self.explanations = None

        # We remove spaces in the column names
        X.columns = [X.columns[i].replace(" ", "_") for i in range(len(X.columns))]
        X = X.reset_index(drop=True)

        self.X_all = X
        self.model = model
        self.y = y
        self.X_scaled = pd.DataFrame(StandardScaler().fit_transform(X))
        self.X_scaled.columns = X.columns

        # We compute the predictions of the model
        self.y_pred = pd.Series(self.model.predict(self.X))

        self.verbose = None # TODO shoulb be setttable through **kwargs ?
        self.widget = None # TODO what is this ?

        self.comments = [""]*len(self.X.columns) 
        self.sensible = [False]*len(self.X.columns)

        self.fraction = 1 # TODO : what is this ?
        self.frac_indexes = self.X.index #

        # TODO : should be handled with a GeoData object ?
        self.long, self.lat = None, None # TODO : shoudl only  be used if needed

        for name in ['longitude', 'Longitude', 'Long', 'long']:
            if name in self.X.columns:
                self.long = name

        for name in ['latitude', 'Latitude', 'Lat', 'lat']:
            if name in self.X.columns:
                self.lat = name

        if user_explanations is not None:
            if user_explanationa_type is None:
                raise ValueError("You must provide the type of the explanations")
            self.explanations = ExplanationsDataset(self, user_explanations.values, user_explanations_type)

    def __str__(self):
        text = "Dataset object :\n"
        text += "------------------\n"
        text += "- Number of observations:"  + str(self.X.shape[0]) + "\n"
        text += "- Number of variables: " + str(self.X.shape[1]) + "\n"
        if self.explanations is not None:
            text += "Includes :\n"
            text += self.explanations.__str__()
        return text
    
    # TODO : is it useful ?
    def __len__(self):
        return self.X.shape[0]


    def getShape()-> tuple:
        """ Returns the shape of the used dataset"""
        return self.X.shape
    
    def frac(self, p:float):
        """
        Reduces the dataset to a fraction of its size.

        Parameters
        ---------
        p : float
            The fraction (%) of the dataset to keep.
        """

        self.X = self.X_all.sample(frac=p, random_state=9)
        self.frac_indexes = deepcopy(self.X.index)
        self.X_scaled = self.X_scaled.iloc[self.frac_indexes].reset_index(drop=True)
        self.y_pred = self.y_pred.iloc[self.frac_indexes].reset_index(drop=True)
        if self.y is not None:
            self.y = self.y.iloc[self.frac_indexes].reset_index(drop=True)
        self.fraction = p
        self.X.reset_index(drop=True, inplace=True)

    def setLongLat(self, long:str, lat:str):
        """
        Sets the longitude and latitude columns of the dataset.

        Parameters
        ---------
        long : str
            The name of the longitude column.
        lat : str
            The name of the latitude column.
        """
        self.long = long
        self.lat = lat

    def getLongLat(self) -> tuple:
        """
        Returns the longitude and latitude columns of the dataset.

        Returns
        -------
        long : str
            The name of the longitude column.
        lat : str
            The name of the latitude column.
        """
        return self.long, self.lat

    def improve(self):
        """
        Improves the dataset. 

        # TODO : shoudl be in the gui module

        Displays a widget to modify the dataset. For each feature, you can change its name, its type, its comment and if it is sensible or not.

        You also have the access to the general informations of the dataset.
        """
        general_infos = v.Row(class_="ma-2", children=[
            v.Icon(children=["mdi-database"], size="30px"),
            v.Html(tag="h3", class_="mb-3 mt-3 ml-4", children=[
                str(self.X.shape[0]) + " observations, " + str(self.X.shape[1]) + " features"
                ])])
        liste_slides = []
        for i in range(self.X.shape[1]):
            infos = [min(self.X.iloc[:,i]), max(self.X.iloc[:,i]), np.mean(self.X.iloc[:,i]), np.std(self.X.iloc[:,i])]
            infos = [round(infos[j], 3) for j in range(len(infos))]
            liste_slides.append(gui_elements.create_slide_dataset(self.X.columns[i], i+1, self.X.dtypes[i], len(self.X.columns), self.comments[i], self.sensible[i], infos))

        slidegroup = v.SlideGroup(
            v_model=None,
            class_="ma-3 pa-3",
            elevation=4,
            center_active=True,
            show_arrows=True,
            children=liste_slides,
        )

        def changement_sensible(widget, event, data):
            i = int(widget.class_)-1
            if widget.v_model :
                liste_slides[i].children[0].color = "red lighten-5"
                self.sensible[i] = True
            else:
                liste_slides[i].children[0].color = "white"
                self.sensible[i] = False

        def changement_names(widget, event, data):
            i = widget.value-1
            self.X = self.X.rename(columns={self.X.columns[i]: widget.v_model})

        def changement_type(widget, event, data):
            i = widget.value-1
            widget2 = liste_slides[i].children[0].children[-1].children[1].children[0]
            try :
                self.X = self.X.astype({self.X.columns[i]: widget2.v_model})
            except:
                print("The type of the column " + self.X.columns[i] + " cannot be changed to " + widget2.v_model)
                widget.color = "error"
                time.sleep(2)
                widget.color = ""
            else:
                widget.color = "success"
                time.sleep(2)
                widget.color = ""

        def changement_comment(widget, event, data):
            i = widget.value-1
            self.comments[i] = widget.v_model

        for i in range(len(liste_slides)):
            liste_slides[i].children[0].children[-1].children[2].on_event("change", changement_sensible)
            liste_slides[i].children[0].children[-1].children[3].on_event("change", changement_comment)
            liste_slides[i].children[0].children[0].children[0].on_event("change", changement_names)
            liste_slides[i].children[0].children[-1].children[1].children[-1].on_event("click", changement_type)

        widget = v.Col(children=[
            general_infos,
            slidegroup,
        ])
        display(widget)

# =============================================================================


class ExplanationsDataset():
    """
    ExplanationsDataset class.

    An Explanations object holds the explanations values of the model for the dataset.

    Instance Attributes
    --------------------
    parent : Dataset
        The parent dataset.
    shapValues : dict
        - key IMPORTED : a pandas Dataframe containing the SHAP values provided by the user.
        - key COMPUTED : a pandas Dataframe containing the SHAP values computed by AntakIA.
        - key DimensionalityReduction.PCA : a dict with :
            - key DimensionalityReduction.DIM_TWO : a pandas Dataframe with 2D PCA-projected SHAP values
            - key DimensionalityReduction.DIM_THREE :  a pandas Dataframe with 3D PCA-projected SHAP values
        - key DimensionalityReduction.TSNE : a dict with :
            - key DimensionalityReduction.DIM_TWO : a pandas Dataframe with 2D TSNE-projected SHAP values
            - key DimensionalityReduction.DIM_THREE :  a pandas Dataframe with 3D TSNE-projected SHAP values
        - key DimensionalityReduction.UMAP : a dict with :
            - key DimensionalityReduction.DIM_TWO : a pandas Dataframe with 2D UMAP-projected SHAP values
            - key DimensionalityReduction.DIM_THREE :  a pandas Dataframe with 3D UMAP-projected SHAP values
        -  key DimensionalityReduction.PacMAP : a dict with :
            - key DimensionalityReduction.DIM_TWO : a pandas Dataframe with 2D PacMAP-projected SHAP values
            - key DimensionalityReduction.DIM_THREE :  a pandas Dataframe with 3D PacMAP-projected SHAP values
    limeValues : dict
        - key IMPORTED : a pandas Dataframe containing the LIME values provided by the user.
        - key COMPUTED : a pandas Dataframe containing the LIME values computed by AntakIA.
        - key DimensionalityReduction.PCA : a dict with :
            - key DimensionalityReduction.DIM_TWO : a pandas Dataframe with 2D PCA-projected LIME values
            - key DimensionalityReduction.DIM_THREE :  a pandas Dataframe with 3D PCA-projected LIME values
        - key DimensionalityReduction.TSNE : a dict with :
            - key DimensionalityReduction.DIM_TWO : a pandas Dataframe with 2D TSNE-projected LIME values
            - key DimensionalityReduction.DIM_THREE :  a pandas Dataframe with 3D TSNE-projected LIME values
        - key DimensionalityReduction.UMAP : a dict with :
            - key DimensionalityReduction.DIM_TWO : a pandas Dataframe with 2D UMAP-projected LIME values
            - key DimensionalityReduction.DIM_THREE :  a pandas Dataframe with 3D UMAP-projected LIME values
        -  key DimensionalityReduction.PacMAP : a dict with :
            - key DimensionalityReduction.DIM_TWO : a pandas Dataframe with 2D PacMAP-projected LIME values
            - key DimensionalityReduction.DIM_THREE :  a pandas Dataframe with 3D PacMAP-projected LIME values
        """
    
    # Class attributes
    # Characterizes an user-provided Dataframe of explainations
    IMPORTED = -1 
    # Characterizes an AntaKIA-computed Dataframe of explainations
    COMPUTED = 0

    def __init__(self, parent : Dataset, values : pd.DataFrame, explanationType:int):
        """
        Constructor of the class Dataset.
        
        Parameters :
        ------------
        parent : Dataset
            The parent dataset.
        values : pandas.Dataframe
            The dataframe containing explanations values. Must match parent.shape()
        explanationType : int
            Must be ExplainedValues.SHAP or ExplainedValues.LIME
        """
        if values.shape != parent.X.shape :
            raise ValueError("The shape of the explanations dataframe must match the shape of the parent dataset")
        self.parent = parent

        if explanationType==ExplainedValues.SHAP :
            self.shapValues = {self.IMPORTED : values}
            self.limeValues = {self.IMPORTED : None}
        elif explanationType==ExplainedValues.LIME :
            self.shapValues = {self.IMPORTED : None}
            self.limeValues = {self.IMPORTED : values}
        else :
            raise ValueError("explanationType must be ExplainedValues.SHAP or ExplainedValues.LIME")


    def getValues(self, explainationMethodType:int, dimensionalityReductionType:int = None, dimension:int = None) -> pd.DataFrame:
        """
        Returns the explanations values of the dataset.

        Parameters
        ----------
        explainationMethodType : int
            Must be ExplainationMethod.SHAP or ExplainationMethod.LIME
        dimensionalityReductionType : int
            Must be DimensionalityReduction.PCA, DimensionalityReduction.TSNE, DimensionalityReduction.UMAP or DimensionalityReduction.PacMAP   
        dimension : int 
            The dimension of the projection (2 or 3).

        Returns
        -------
        values : pandas.Dataframe
            The values for this explainationMethodType, dimensionalityReductionType and dimension.
            Returns None if not computed yet.
        """
        if explainationMethodType==ExplainationMethod.SHAP :
            if dimension == DimensionalityReduction.DIM_ALL :
                if ExplanationsDataset.COMPUTED in self.shapValues :
                    return self.shapValues[ExplanationsDataset.COMPUTED] # We return computed values first
                else :
                    if ExplanationsDataset.IMPORTED in self.shapValues : 
                        return self.shapValues[ExplanationsDataset.IMPORTED] # we return user-provided values if no cimputed values
                    else :
                        return None      
            else :
                if dimension == DimensionalityReduction.DIM_TWO :
                    if self.shapValues[dimensionalityReductionType][DimensionalityReduction.DIM_TWO] is not None :
                        return self.shapValues[dimensionalityReductionType][DimensionalityReduction.DIM_TWO]
                    else :
                        # TODO log that there is no such projection computed yet for this explanation method and dimension
                        return None
                else :
                    if dimension == DimensionalityReduction.DIM_THREE :
                        if self.shapValues[dimensionalityReductionType][DimensionalityReduction.DIM_THREE] is not None :
                            return self.shapValues[dimensionalityReductionType][DimensionalityReduction.DIM_THREE]
                        else :
                            return None
                    else :
                        raise ValueError(dimension, " is not a proper dimension")
        else :
            if explainationMethodType==ExplainationMethod.LIME :
                if dimension == DimensionalityReduction.DIM_ALL :
                    if ExplanationsDataset.COMPUTED in self.limeValues :
                        return self.limeValues[ExplanationsDataset.COMPUTED]
                    else :
                        if ExplanationsDataset.IMPORTED in self.limeValues : 
                            return self.limeValues[ExplanationsDataset.IMPORTED]
                        else :
                            raise ValueError(dimension, " is not a proper dimension")
                else :
                    if dimension == DimensionalityReduction.DIM_TWO :
                        if self.limeValues[dimensionalityReductionType][DimensionalityReduction.DIM_TWO] is not None :
                            return self.limeValues[dimensionalityReductionType][DimensionalityReduction.DIM_TWO]
                        else :
                            # TODO log that there is no such projection computed yet for this explanation method and dimension
                            return None
                    else :
                        if dimension == DimensionalityReduction.DIM_THREE :
                            if self.limeValues[dimensionalityReductionType][DimensionalityReduction.DIM_THREE] is not None :
                                return self.limeValues[dimensionalityReductionType][DimensionalityReduction.DIM_THREE]
                            else :
                                return None
                        else :
                            raise ValueError(dimension, " is not a proper dimension")
            else :
                 raise ValueError("Bad explanantion method type")
        
    def userProvidedData(self, explainationMethodType:int) -> bool:
        """
        Returns True if the user provided explanations values, False otherwise.
        """
        
        if explainationMethodType==ExplainationMethod.SHAP :
            return self.shapValues[ExplanationsDataset.IMPORTED] is not None
        else :
            return self.limeValues[ExplanationsDataset.IMPORTED] is not None
        

    def antakIAComputedData(self, explainationMethodType:int) -> bool:
        """
        Returns True if AntakIA computed explanations values, False otherwise.
        """
        
        if explainationMethodType==ExplainationMethod.SHAP :
            return self.shapValues[ExplanationsDataset.COMPUTED] is not None
        else :
            return self.limeValues[ExplanationsDataset.COMPUTED] is not None


    # TOOD : should be refactored
    def __str__(self) -> str:  
        text = "Explanation object :\n"
        text += "------------------\n"
        shap = False
        if self.userProvidedData(ExplainationMethod.SHAP) :
            text += "- SHAP values imported : YES\n"
            shap = True
        if self.antakIAComputedData(ExplainationMethod.SHAP) :
            text += "- SHAP values computed : YES\n"
            shap = True
        if shap :
            if self.getValues(ExplainationMethod.SHAP, DimensionalityReduction.PCA, DimensionalityReduction.DIM_TWO) is not None :
                text += "   - Projection PCA 2D : YES\n"
            if self.getValues(ExplainationMethod.SHAP, DimensionalityReduction.PCA, DimensionalityReduction.DIM_THREE) is not None :
                text += "   - Projection PCA 3D : YES\n"
            if self.getValues(ExplainationMethod.SHAP, DimensionalityReduction.TSNE, DimensionalityReduction.DIM_TWO) is not None :
                text += "   - Projection TSNE 2D : YES\n"
            if self.getValues(ExplainationMethod.SHAP, DimensionalityReduction.TSNE, DimensionalityReduction.DIM_THREE) is not None :
                text += "   - Projection TSNE 3D : YES\n"
            if self.getValues(ExplainationMethod.SHAP, DimensionalityReduction.UMAP, DimensionalityReduction.DIM_TWO) is not None :
                text += "   - Projection UMAP 2D : YES\n"
            if self.getValues(ExplainationMethod.SHAP, DimensionalityReduction.UMAP, DimensionalityReduction.DIM_THREE) is not None :   
                text += "   - Projection UMAP 3D : YES\n"
            if self.getValues(ExplainationMethod.SHAP, DimensionalityReduction.PacMAP, DimensionalityReduction.DIM_TWO) is not None :
                text += "   - Projection PacMAP 2D : YES\n"
            if self.getValues(ExplainationMethod.SHAP, DimensionalityReduction.PacMAP, DimensionalityReduction.DIM_THREE) is not None :   
                text += "   - Projection PacMAP 3D : YES\n"   

        lime = False
        if self.userProvidedData(ExplainationMethod.LIME) :
            text += "- LIME values imported : YES\n"
            lime = True
        if self.antakIAComputedData(ExplainationMethod.LIME) :
            text += "- LIME values computed : YES\n"
            lime = True
        if lime :
            if self.getValues(ExplainationMethod.LIME, DimensionalityReduction.PCA, DimensionalityReduction.DIM_TWO) is not None :
                text += "   - Projection PCA 2D : YES\n"
            if self.getValues(ExplainationMethod.LIME, DimensionalityReduction.PCA, DimensionalityReduction.DIM_THREE) is not None :
                text += "   - Projection PCA 3D : YES\n"
            if self.getValues(ExplainationMethod.LIME, DimensionalityReduction.TSNE, DimensionalityReduction.DIM_TWO) is not None :
                text += "   - Projection TSNE 2D : YES\n"
            if self.getValues(ExplainationMethod.LIME, DimensionalityReduction.TSNE, DimensionalityReduction.DIM_THREE) is not None :
                text += "   - Projection TSNE 3D : YES\n"
            if self.getValues(ExplainationMethod.LIME, DimensionalityReduction.UMAP, DimensionalityReduction.DIM_TWO) is not None :
                text += "   - Projection UMAP 2D : YES\n"
            if self.getValues(ExplainationMethod.LIME, DimensionalityReduction.UMAP, DimensionalityReduction.DIM_THREE) is not None :   
                text += "   - Projection UMAP 3D : YES\n"
            if self.getValues(ExplainationMethod.LIME, DimensionalityReduction.PacMAP, DimensionalityReduction.DIM_TWO) is not None :
                text += "   - Projection PacMAP 2D : YES\n"
            if self.getValues(ExplainationMethod.LIME, DimensionalityReduction.PacMAP, DimensionalityReduction.DIM_THREE) is not None :   
                text += "   - Projection PacMAP 3D : YES\n"   

        return text
