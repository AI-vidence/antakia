import pandas as pd
import numpy as np

from antakia.gui import GUI
from antakia.dataset import Dataset

import antakia.utils as utils
from antakia.utils import function_auto_clustering

import ipywidgets as widgets

from antakia import compute

from IPython.display import display

from antakia.potato import Potato

import ipyvuetify as v

class AntakIA():
    """AntakIA object.
    This main class of the antakia package. It contains all the data and variables needed to run the interface (see antakia.GUI).

    Attributes
    -------
    dataset : Dataset object
        The Dataset object containing the data to explain. For more information, please see the documentation of the class Dataset.
    explain : dict
        The dictionary containing the explanations. The keys are the names of the explanations (for example "SHAP" or "LIME"). The explanations are pandas dataframes.
        You can import your own explanations using `import_explanation`.
    regions : list
        The list of the regions computed by the user. A region is an AntakIA object, named Potato. For more information, please see the documentation of the class Potato.
    saves : list
        The list of the saves. A save is a list of regions.
    gui : GUI object
        The GUI object is in charge of the interface. For more information, please see the documentation of the class GUI.
    """

    # TODO : il faudrait un constructeur __init__(self, dataset) tout court non ?
    def __init__(self, dataset: Dataset, import_explanation: pd.DataFrame = None, saves: dict = None, saves_path: str = None):
        """
        Constructor of the class AntakIA.

        Parameters
        ---------
        dataset : Dataset object
            The Dataset object containing the data to explain.
        import_explanation : pandas dataframe
            The dataframe containing the explanations. The dataframe must have the same number of rows as the dataset.
            The GUI can compute other types of explanations using different methods.
        """
        self.dataset = dataset
        self.regions = []
        self.gui = None

        self.explain = dict()
        if import_explanation is not None:
            self.explain["Imported"] = import_explanation.iloc[dataset.frac_indexes].reset_index(drop=True)
        else:
            self.explain["Imported"] = None
        self.explain["SHAP"] = None
        self.explain["LIME"] = None

        if saves is not None:
            self.saves = saves
        elif saves_path is not None:
            self.saves = utils.load_save(self, saves_path)
        else:
            self.saves = []

        self.widget = None

        self.gui = GUI(self)

    def getGUI(self) -> GUI:
        """
        Function that returns the GUI object.
        For more information, please see the documentation of the class GUI.

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
    
    def getExplanations(self, method=None) -> dict:
        """
        Function that returns the dictionary containing the explanations.
        The keys are the names of the explanations (for example "SHAP" or "LIME"). The explanations are pandas dataframes.

        Returns
        -------
        dict or pandas dataframe
            The dictionary containing the explanations or the explanation corresponding to the key explanation.
        """
        if method is None:
            return self.explain
        try :
            return self.explain[method]
        except KeyError:
            raise KeyError("The method " + method + " is not a valid method. The possible methods are " + str(list(self.explain.keys())) + ".")
    
    def getDataset(self) -> Dataset:
        """
        Function that returns the Dataset object containing the data to explain.
        For more information, please see the documentation of the class Dataset.

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

    def startGUI(self,
                explanation: str = None,
                projection: str = "PaCMAP",
                sub_models: list = None,
                display = True) -> GUI:
        """
        Function that instantiates the GUI and calls its display() function.
        For more information, please see the documentation of the class GUI.

        Parameters
        ---------
        explanation : str
            The type of explanation to use. If an explanation is already computed (see antakia.dataset), it is used by default. If not, the user must choose between "SHAP" and "LIME".
            The explanatory values caen be computed directly in the interface!
        projection : str
            The default projection to use. The possible values are "PaCMAP", "PCA", "t-SNE" and "UMAP".
        sub_models : list
            The list of the sub_models to choose from for each region. The only constraint is that sub_models must have a predict method.
        display : bool
            If True, the interface is displayed. Else, You can access the interface with the attribute gui of the class.
        """
        self.gui = GUI(self, explanation, projection, sub_models)
        if display:
            self.gui.display()

    def computeDyadicClustering(self, explanation:str = "Imported", min_clusters:int = 3, automatic:bool = True, sub_models:bool = False):
        """
        Function that computes the dyadic-clustering.
        Our dyadic-clustering (sometimes found as co-clusetring or bi-clustering), uses `mvlearn` and `skope-rules` to compute the clusters.

        Parameters
        ---------
        explanation : str
            The type of explanation to use.
            The possible values are "Imported", "SHAP" and "LIME".
        min_clusters : int
            The minimum number of clusters to compute.
        automatic : bool
            If True, the number of clusters is computed automatically, respecting the minimum number of clusters.
        sub_models : bool
            If True, the best model for each region is computed. The possible models are the ones in the list sub_models.
        """
        if self.explain[explanation] is None:
            raise ValueError("You must compute the explanations before computing the dyadic-clustering!")
        if min_clusters <2 or min_clusters > len(self.dataset.X):
            raise ValueError("The minimum number of clusters must be between 2 and the number of observations!")
        clusters, clusters_axis = function_auto_clustering(self.dataset.X, self.explain[explanation], min_clusters, automatic)
        self.regions = []
        for i in range(len(clusters)):
            self.regions.append(Potato(self, clusters[i]))
            if sub_models:
                self.regions[i].sub_model["model"], self.regions[i].sub_model["score"] = self.__find_best_model(self.regions[i].data, self.regions[i].y, self.gui.sub_models)

    def __find_best_model(self, X:pd.DataFrame, y:pd.Series, sub_models:list):
        best_model = None
        best_score = 0
        for model in sub_models:
            model.fit(X, y)
            score = model.score(X, y)
            if score > best_score:
                best_score = score
                best_model = model
        return best_model.__class__.__name__, round(best_score, 4)

    def __create_progress(self, titre:str):
        widget = v.Col(
            class_="d-flex flex-column align-center",
            children=[
                    v.Html(
                        tag="h3",
                        class_="mb-3",
                        children=["Compute " + titre + " values"],
                ),
                v.ProgressLinear(
                    style_="width: 80%",
                    v_model=0,
                    color="primary",
                    height="15",
                    striped=True,
                ),
                v.TextField(
                    class_="w-100",
                    style_="width: 100%",
                    v_model = "0.00% [0/?] - 0m0s (estimated time : /min /s)",
                    readonly=True,
                ),
            ],
        )
        self.widget = widget

    
    def computeSHAP(self, verbose:bool = True):
        """
        Computes the SHAP values of the dataset.

        Parameters
        ---------
        verbose : bool
            If True, a progress bar is displayed.

        See also:
        ---------
        The Shap library on GitHub : https://github.com/shap/shap/tree/master
        """
        shap = compute.computationSHAP(self.dataset.X, self.dataset.X_all, self.dataset.model)
        if verbose:
            self.verbose = self.__create_progress("SHAP")
            widgets.jslink((self.widget.children[1], "v_model"), (shap.progress_widget, "v_model"))
            widgets.jslink((self.widget.children[2], "v_model"), (shap.text_widget, "v_model"))
            display(self.widget)
        self.explain["SHAP"] = shap.compute()

    def computeLIME(self, verbose:bool = True):
        """
        Computes the LIME values of the dataset.

        Parameters
        ---------
        verbose : bool
            If True, a progress bar is displayed.

        See also:
        ---------
        The Lime library on GitHub : https://github.com/marcotcr/lime/tree/master
        """
        lime = compute.computationSHAP(self.dataset.X, self.dataset.X_all, self.dataset.model)
        if verbose:
            self.verbose = self.__create_progress("LIME")
            widgets.jslink((self.widget.children[1], "v_model"), (lime.progress_widget, "v_model"))
            widgets.jslink((self.widget.children[2], "v_model"), (lime.text_widget, "v_model"))
            display(self.widget)
        self.explain["LIME"] = lime.compute()