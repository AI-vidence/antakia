import pandas as pd
import numpy as np

from antakia.gui import GUI
from antakia.dataset import Dataset

from antakia.utils import fonction_auto_clustering

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
        The dictionary containing the explanations. The keys are the names of the explanations (for example "SHAP" or "LIME"). The values are the explanations. The explanations are pandas dataframes.
        You can import your own explanations using `import_explanation`.
    regions : list
        The list of the regions computed by the user. A region is an AntakIA object, named Potato. For more information, please see the documentation of the class Potato.
    saves : list
        The list of the saves. A save is a list of regions.
    gui : GUI object
        The GUI object is in charge of the interface. For more information, please see the documentation of the class GUI.
    """

    # TODO : il faudrait un constructeur __init__(self, dataset) tout court non ?
    def __init__(self, dataset: Dataset, import_explanation: pd.DataFrame = None):
        """
        Constructor of the class AntakIA.

        Parameters
        ---------
        dataset : Dataset object
            The Dataset object containing the data to explain.
        import_explanation : pandas dataframe
            The dataframe containing the explanations. The dataframe must have the same number of rows as the dataset.
            The GUI can compute other types of explanations using different methods.

        Returns
        -------
        AntakIA object
            The AntakIA object.
        """
        self.dataset = dataset
        self.regions = []
        self.gui = None
        self.saves = []

        self.widget = None

        self.explain = dict()
        self.explain["Imported"] = import_explanation.sample(frac=dataset.fraction, random_state=9).reset_index(drop=True)
        self.explain["SHAP"] = None
        self.explain["LIME"] = None

        self.gui = GUI(self)

    def __str__(self):
        print("Xplainer object")

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
            The list of the regions computed.
        """
        return self.regions
    
    def newRegion(self, potato: Potato):
        """
        Function that adds a region to the list of regions.

        Parameters
        ---------
        name : str
            The name of the region.
        """
        self.regions.append(potato)

    def startGUI(self,
                explanation: str = None,
                projection: str = "PaCMAP",
                sub_models: list = None,
                display = True) -> GUI:
        """
        Function that instantiates the GUI and calls it display() function

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

        Returns
        -------
        list
            The list of the regions computed.
        """
        if self.explain[explanation] is None:
            raise ValueError("You must compute the explanations before computing the dyadic-clustering!")
        if min_clusters <2 or min_clusters > len(self.dataset.X):
            raise ValueError("The minimum number of clusters must be between 2 and the number of observations!")
        clusters, clusters_axis = fonction_auto_clustering(self.dataset.X, self.explain[explanation], min_clusters, automatic)
        self.regions = []
        for i in range(len(clusters)):
            self.regions.append(Potato(self, clusters[i]))
            if sub_models:
                self.regions[i].sub_model["model"], self.regions[i].sub_model["score"] = self.__find_best_model(self.regions[i].data, self.regions[i].y, self.gui.sub_models)

    def __find_best_model(self, X:pd.DataFrame, y:pd.Series, sub_models:list):
        """
        Function that finds the best model for a region.

        Parameters
        ---------
        X : pandas dataframe
            The data of the region.
        y : pandas series
            The target of the region.
        sub_models : list
            The list of the sub_models to choose from for each region. The only constraint is that sub_models must have a predict method.

        Returns
        -------
        sklearn model
            The best model for the region.
        """
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

    
    def compute_SHAP(self, verbose:bool = True):
        """
        Computes the SHAP values of the dataset.

        Parameters
        ---------
        verbose : bool
            If True, a progress bar is displayed.

        See also:
        ---------
        The Shap library.
        """
        shap = compute.SHAP_computation(self.dataset.X, self.dataset.X_all, self.dataset.model)
        if verbose:
            self.verbose = self.__create_progress("SHAP")
            widgets.jslink((self.widget.children[1], "v_model"), (shap.progress_widget, "v_model"))
            widgets.jslink((self.widget.children[2], "v_model"), (shap.text_widget, "v_model"))
            display(self.widget)
        self.explain["SHAP"] = shap.compute()

        
    # TODO : ça devrait subclasser LongTask
    def compute_LIME(self, verbose:bool = True):
        """
        Computes the LIME values of the dataset.

        Parameters
        ---------
        verbose : bool
            If True, a progress bar is displayed.

        See also:
        ---------
        The Lime library.
        """
        lime = compute.SHAP_computation(self.dataset.X, self.dataset.X_all, self.dataset.model)
        if verbose:
            self.verbose = self.__create_progress("LIME")
            widgets.jslink((self.widget.children[1], "v_model"), (lime.progress_widget, "v_model"))
            widgets.jslink((self.widget.children[2], "v_model"), (lime.text_widget, "v_model"))
            display(self.widget)
        self.explain["LIME"] = lime.compute()