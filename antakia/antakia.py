import pandas as pd
import numpy as np

from antakia.gui import GUI
from antakia.dataset import Dataset

# from antakia.utils import fonction_auto_clustering

import ipywidgets as widgets

# import antakia.longtask as LongTask

from IPython.display import display

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
        self.regions = [] #TODO : regions peut être une liste de N listes de régions. Le liste N est en cours, les N-1 précédentes sont "saved"
        self.gui = None
        self.saves = [] #TODO : voir intule. regions

        self.explain = dict()
        self.explain["Imported"] = import_explanation
        self.explain["SHAP"] = None
        self.explain["LIME"] = None

    def __str__(self):
        print("Xplainer object")

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
        #TODO : soit retourner gui, soit changer la doc / "returns"

    def dyadic_clustering(self, explanation:str = "Imported", min_clusters:int = 3, automatic:bool = True):
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
        if self.dataset.explain[explanation] is None:
            raise ValueError("You must compute the explanations before computing the dyadic-clustering!")
        if min_clusters <2 or min_clusters > len(self.dataset.X):
            raise ValueError("The minimum number of clusters must be between 2 and the number of observations!")
        return fonction_auto_clustering(self.dataset.X, self.dataset.explain[explanation], min_clusters, automatic)
    
    # TODO : ça devrait subclasser LongTask
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
        shap = LongTask.compute_SHAP(self.X, self.X_all, self.model)
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
        lime = LongTask.compute_LIME(self.X, self.X_all, self.model)
        if verbose:
            self.verbose = self.__create_progress("SHAP")
            widgets.jslink((self.widget.children[1], "v_model"), (lime.progress_widget, "v_model"))
            widgets.jslink((self.widget.children[2], "v_model"), (lime.text_widget, "v_model"))
            display(self.widget)
        self.explain["LIME"] = lime.compute()