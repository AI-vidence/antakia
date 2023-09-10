import pandas as pd
import numpy as np
import threading
import time
from abc import ABC, abstractmethod
from pubsub import pub
from typing import Tuple

# Imports for the dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import pacmap

# Imports for the explanations
import lime
import lime.lime_tabular
import shap

from antakia.model import Model

class LongTask(ABC):
    '''
    Abstract class to compute long tasks, often in a separate thread.

    Attributes
    ----------
    __longTaskType : Tuple[int,int]
        For instance,can be (LongTask.EXPLAINATION, ExplainationMethod.SHAP)
        or (LongTask.DIMENSIONALITY_REDUCTION, DimensionalityReduction.PCA)
    __X : pandas dataframe
        The dataframe containing the data to explain.
    __X_all : pandas dataframe
        The dataframe containing the entire dataset, in order for the explanations to be computed.
    __model : Model object as defined in antakia/model.py
        The "black-box" model to explain.
    __progress : int
        The progress of the long task, between 0 and 100.
    '''

    # Class attributes : LongTask types
    EXPLAINATION = 1
    DIMENSIONALITY_REDUCTION = 2

    def __init__(self, longTaskType : Tuple[int,int], X  : pd.DataFrame, X_all : pd.DataFrame, model : Model = None) :
        self.__longTaskType = longTaskType
        if not LongTask.isValidLongTaskType(longTaskType) :
            return ValueError("The long task type is not valid!")
        self.__X = X
        self.__X_all = X_all
        if longTaskType[0] == LongTask.EXPLAINATION and model is None :
            raise ValueError("You must provide a model to compute the explaination!")
        self.__model = model
        self.__progress = 0

    @staticmethod
    def isValidLongTaskType(type: Tuple[int,int]) -> bool:
        """
        Returns True if the type is a valid LongTask type.
        """
        if type[0] == LongTask.EXPLAINATION and ExplainationMethod.isValidExplanationType(type[1]) :
            return True
        elif type[0] == LongTask.DIMENSIONALITY_REDUCTION and DimReducMethod.isValidDimReducType(type[1]) :
            return True
        else : return False

    def getTopic(self) -> str:
        """Returns a string describing the long task type.
        We concatenate the two integers of the tuple with a slash.

        Returns:
            str: "{longTaskType{0]}/{longTaskType[1]}"
        """
        return str(self.__longTaskType[0]) + "/" + str(self.__longTaskType[1])
    
    def getLongTaskType(self) -> Tuple[int,int]:
        return self.__longTaskType
    
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
        pub.sendMessage(self.getTopic(), progress=self.__progress)



# ===========================================================
#                   Explanations
# ===========================================================

class ExplainationMethod(LongTask):
    """
    Abstract class (Long Task) to compute explaination values for the Explanation Space (ES)
    """

    # Class attributes : ExplainationMethod types
    NONE=-1
    SHAP = 0
    LIME = 1
    OTHER = 2 


    @staticmethod
    def isValidExplanationType(type:int) -> bool:
        """
        Returns True if the type is a valid explanation type.
        """
        return type == ExplainationMethod.SHAP or type == ExplainationMethod.LIME or type == ExplainationMethod.OTHER

class SHAPExplaination(ExplainationMethod):
    """
    SHAP computation class.
    """

    def __init__(self, X:pd.DataFrame, X_all:pd.DataFrame, model:Model):
        super().__init__((LongTask.EXPLAINATION, ExplainationMethod.SHAP), X, X_all, model)

    def compute(self) -> pd.DataFrame :
        time_init = time.time()
        explainer = shap.Explainer(self.__model.predict, self.__X_all) # TODO : why is it X_all ?

        valuesSHAP = pd.DataFrame().reindex_like(self.__X)

        colNames = list(self.__X.columns)
        for i in range(len(colNames)):
            colNames[i] = colNames[i] + "_shap"

        for i in range(len(self.__X)):
            shap_value = explainer(self.__X[i : i + 1], max_evals=1400) #TOTDO : why 1400 ?
            valuesSHAP.iloc[i] = shap_value.values
            self.__progress += 100 / len(self.__X)
            self.sendProgress()

        valuesSHAP.columns = colNames
        return valuesSHAP
    
    @staticmethod
    def getExplanationType() -> int:
        return ExplainationMethod.SHAP

class LIMExplaination(ExplainationMethod):
    """
    LIME computation class.
    """
    
    def __init__(self, X:pd.DataFrame, X_all:pd.DataFrame, model:Model):
        super().__init__((LongTask.EXPLAINATION, ExplainationMethod.LIME), X, X_all, model)

    def compute(self) -> pd.DataFrame :
        time_init = time.time()

        # TODO : It seems we defined class_name in order to workl with California housing dataset. We should find a way to generalize this.
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(self.__X_all), feature_names=self.__X.columns, class_names=['price'], verbose=False, mode='regression')

        N = len(self.__X.shape)
        valuesLIME = pd.DataFrame(np.zeros((N, self.__X.shape[-1])))
        
        for j in range(N):
            l = []
            exp = explainer.explain_instance(
                self.__X.values[j], self.__model.predict
            )
            l = []
            taille = self.__X.shape[-1]
            for ii in range(taille):
                exp_map = exp.as_map()[0]
                l.extend(exp_map[ii][1] for jj in range(taille) if ii == exp_map[jj][0])
            
            valuesLIME.iloc[j] = pd.Series(l)
            self.__progress += 100 / len(self.__X)
            self.sendProgress()
        j = list(self.__X.columns)
        for i in range(len(j)): 
            j[i] = j[i] + "_lime"
        valuesLIME.columns = j
        return valuesLIME

    @staticmethod
    def getExplanationType() -> int:
        return ExplainationMethod.LIME

# ===========================================================
#             Projections / Dim Reductions
# ===========================================================

class DimReducMethod(LongTask):
    """
    Class that allows to reduce the dimensionality of the data.

    Attributes
    ----------
    __dimension : int
        Dimension reduction methods require a dimension parameter
        We store it in the abstract class


    """

    NONE = 0
    PCA = 1
    TSNE = 2
    UMAP = 3
    PaCMAP = 4

    DIM_ALL = -1
    DIM_TWO = 2
    DIM_THREE = 3

    """
    Constructor for the DimensionalityReductionMethod class.

    Parameters
    ----------
    longTaskType : Tuple[int,int]
        need by LongType consturctor
    X, X_all : pd.DataFrame
        idem
    dimension : int
        Dimension reduction methods require a dimension parameter
        We store it in the abstract class
    
    """

    def __init__(self, longTaskType : Tuple[int,int], X  : pd.DataFrame, X_all : pd.DataFrame, dimension : int) :
        self.__dimension = dimension
        super().__init__(longTaskType, X, X_all)

    @staticmethod
    def getDimReducMehtodAsStr(type : int) -> str :
        if type == DimReducMethod.PCA :
            return "PCA"
        elif type == DimReducMethod.TSNE :
            return "t-SNE"
        elif type == DimReducMethod.UMAP :
            return "UMAP"
        elif type == DimReducMethod.PacMAP :
            return "PaCMAP"
        else :
            return None

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
        return type == DimReducMethod.PCA or type == DimReducMethod.TSNE or type == DimReducMethod.UMAP or type == DimReducMethod.PacMAP

    
        
class PCADimReduc(DimReducMethod):
    """
    PCA computation class.
    """
    def __init__(self, X:pd.DataFrame, X_all:pd.DataFrame, dimension : int = 2):
        super().__init__((LongTask.DIMENSIONALITY_REDUCTION, DimReducMethod.PCA), X, X_all, dimension=dimension)

    def compute(self) -> pd.DataFrame:
        __progress = 0
        pca = PCA(n_components=super().__dimension)
        pca.fit(super().__X)
        X_pca = pca.transform(super().__X)
        X_pca = pd.DataFrame(X_pca)
        # TODO : we need to iterated over the dataset in order to sendProgress messages to the GUI
        __progress = 100
        self.sendProgress()
        return X_pca

    
class TSNEDimReduc(DimReducMethod):
    """
    T-SNE computation class.
    """

    def __init__(self, X:pd.DataFrame, X_all:pd.DataFrame, dimension : int = 2):
        super().__init__((LongTask.DIMENSIONALITY_REDUCTION, DimReducMethod.TSNE), X, X_all, dimension=dimension)
    
    def compute(self) -> pd.DataFrame:
        __progress = 0
        tsne = TSNE(n_components=super().__dimension)
        X_tsne = tsne.fit_transform(super().__X)
        X_tsne = pd.DataFrame(X_tsne)
        __progress = 100
        self.sendProgress()
        return X_tsne
    
class UMAPDimReduc(DimReducMethod):
    """
    UMAP computation class.
    """
    def __init__(self, X:pd.DataFrame, X_all:pd.DataFrame, dimension : int = 2):
        super().__init__((LongTask.DIMENSIONALITY_REDUCTION, DimReducMethod.UMAP), X, X_all, dimension=dimension)

    def compute(self) -> pd.DataFrame:
        __progress = 0
        reducer = umap.UMAP(n_components=super().__dimension)
        embedding = reducer.fit_transform(super().__X)
        embedding = pd.DataFrame(embedding)
        __progress = 100
        self.sendProgress()
        return embedding

class PaCMAPDimReduc(DimReducMethod):
    """
    PaCMAP computation class.
    """


    def __init__(self, X:pd.DataFrame, X_all:pd.DataFrame, dimension : int = 2, **kwargs):
        super().__init__((LongTask.DIMENSIONALITY_REDUCTION, DimReducMethod.PacMAP), X, X_all, dimension=dimension)
        self.__paramDict = dict()
        # TODO : below, can I call setParam() while in the constructor ?
        if "neighbours" in kwargs:
                self.__paramDict.update({"neighbours": kwargs["neighbours"]})
        if "MNratio" in kwargs :
                self.__paramDict.update({"MN_ratio": kwargs["MN_ratio"]})
        if "FP_ratio" in kwargs :
                self.__paramDict.update({"FP_ratio": kwargs["FP_ratio"]})


    
    def compute(self, *args):
        reducer = pacmap.PaCMAP(
                n_components=super().__dimension,
                n_neighbors=args[0],
                MN_ratio=args[1],
                FP_ratio=args[2],
                random_state=9,
            )
        embedding = reducer.fit_transform(super().__X, init="pca")
        embedding = pd.DataFrame(embedding)
        return embedding


# ===========================================================

def computeProjections(X, dimReducMethod:int, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes projected values for VS or ES in 2 and 3D given a dimensionality reduction method.
    Returns a Tuple with values for 2D and 3D projections.
    """

    #TODO why do we need X-all here ?
    if dimReducMethod == DimReducMethod.PCA:
        return (PCADimReduc(X, None, DimReducMethod.DIM_TWO).compute(),
            PCADimReduc(X, None, DimReducMethod.DIM_THREE).compute())
    elif dimReducMethod == DimReducMethod.TSNE:
        return (TSNEDimReduc(X, None, DimReducMethod.DIM_TWO).compute(),
            TSNEDimReduc(X, None, DimReducMethod.DIM_THREE).compute())
    elif dimReducMethod == DimReducMethod.UMAP:
        return (UMAPDimReduc(X, None, DimReducMethod.DIM_TWO).compute(),
            UMAPDimReduc(X, None, DimReducMethod.DIM_THREE).compute())
    elif dimReducMethod == DimReducMethod.PacMAP:
        if kwargs is None:
            return (PaCMAPDimReduc(X, None, DimReducMethod.DIM_TWO).compute(),
                PaCMAPDimReduc(X, None, DimReducMethod.DIM_THREE).compute())
        else : 
            return (PaCMAPDimReduc(X, None, DimReducMethod.DIM_TWO, kwargs).compute(),
                PaCMAPDimReduc(X, None, DimReducMethod.DIM_THREE,kwargs).compute())
    else :
        raise ValueError("This default projection is not valid!")
        

        

def function_score(y, y_chap):
    y = np.array(y)
    y_chap = np.array(y_chap)
    return round(np.sqrt(sum((y - y_chap) ** 2) / len(y)), 3)

def function_beeswarm_shap(gui, exp, nom_colonne):
    X = gui.atk.dataset.X
    Exp = gui.atk.explain[exp]
    y = gui.atk.dataset.y_pred
    
    # redefinition de la figure beeswarm de shap
    def positions_ordre_croissant(lst):
        positions = list(range(len(lst)))  # Create a list of initial positions
        positions.sort(key=lambda x: lst[x])
        l = []
        for i in range(len(positions)):
            l.append(positions.index(i))  # Sort positions by list items
        return l
    
    nom_colonne_shap = nom_colonne + "_shap"
    y_histo_shap = [0] * len(Exp)
    nombre_div = 60
    garde_indice = []
    garde_valeur_y = []
    for i in range(nombre_div):
        garde_indice.append([])
        garde_valeur_y.append([])
    liste_scale = np.linspace(
        min(Exp[nom_colonne_shap]), max(Exp[nom_colonne_shap]), nombre_div + 1
    )
    for i in range(len(Exp)):
        for j in range(nombre_div):
            if (
                Exp[nom_colonne_shap][i] >= liste_scale[j]
                and Exp[nom_colonne_shap][i] <= liste_scale[j + 1]
            ):
                garde_indice[j].append(i)
                garde_valeur_y[j].append(y[i])
                break
    for i in range(nombre_div):
        l = positions_ordre_croissant(garde_valeur_y[i])
        for j in range(len(garde_indice[i])):
            ii = garde_indice[i][j]
            if l[j] % 2 == 0:
                y_histo_shap[ii] = l[j]
            else:
                y_histo_shap[ii] = -l[j]
    marker_shap = dict(
        size=4,
        opacity=0.6,
        color=X[nom_colonne],
        colorscale="Bluered_r",
        colorbar=dict(thickness=20, title=nom_colonne),
    )
    return [y_histo_shap, marker_shap]




```python
def compute(self, X, explanation, projection, sub_models, display=True):
    """
    Function that computes the regions and starts the GUI.

    Parameters
    ----------
    X : array-like
        The data to compute the regions on.
    explanation : int
        The index of the sub_model to use for the explanation.
    projection : int
        The index of the sub_model to use for the projection.
    sub_models : list
        The list of the sub_models to choose from for each region. The only constraint is that sub_models must have a predict method.
    display : bool, optional
        If True, the interface is displayed. Else, You can access the interface with the attribute gui of the class. The default is True.
    """
    self.resetRegions()
    self.saves = []
    self.widget = None

    self.computeRegions(X, explanation, projection, sub_models)
    self.gui = GUI(self, explanation, projection, sub_models)
    if display:
        self.gui.display()  projection, sub_models)
    self.gui = GUI(self, explanation, projection, sub_models)
    if display:
        self.gui.display()index o sub_mod [] explanation,
    self.widget = None
    self.computeRegions(X,
els mu def .
    self.saves =    """Regions()

    self.resetis Trueault th in Thete .classrfac attribute gu thei of thee withes   accesscandispl  YouEls,eayed. pred is
        If True, the interfaceict me.
    display : bool, optionalthodt have a the sub_m ch only c is thatonstraintoose  ea Thech. regionfrom for toodelsf the sub_mod for projec.
        The list of    sub_models : list
tion theel to use to compute the regi use  explana.
    pro Thejection : int
       tion theforons on. to_model
        The index of the sub    explanation : int
```python explanat
        The dataion=True): st.
-like
    X : array    Parameters
    ----------
arts the GUI
    """ regions and
    Function that computes the, pr displayoje s_model,sub,ction
def compute(self, X,
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
        if self.explainations[explanation] is None:
            raise ValueError("You must compute the explanations before computing the dyadic-clustering!")
        if min_clusters <2 or min_clusters > len(self.dataset.X):
            raise ValueError("The minimum number of clusters must be between 2 and the number of observations!")
        clusters, clusters_axis = function_auto_clustering(self.dataset.X, self.explainations[explanation], min_clusters, automatic)
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
        self.explainations["SHAP"] = shap.compute()

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
        self.explainations["LIME"] = lime.compute()