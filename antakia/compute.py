import pandas as pd
import numpy as np
import threading
import time
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
import logging

from ipywidgets.widgets.widget import Widget

from antakia.data import Dataset, ExplanationsDataset, LongTask, ExplanationMethod, DimReducMethod, Model

import logging
from log_utils import OutputWidgetHandler
logger = logging.getLogger(__name__)
handler = OutputWidgetHandler()
handler.setFormatter(logging.Formatter('compute.py [%(levelname)s] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
handler.clear_logs()
handler.show_logs()


# ===========================================================
#              Explanations implementations
# ===========================================================

class SHAPExplanation(ExplanationMethod):
    """
    SHAP computation class.
    """

    def __init__(self, X:pd.DataFrame, X_all:pd.DataFrame, model:Model):
        super().__init__(ExplanationMethod.SHAP, X, X_all, model)


    def compute(self) -> pd.DataFrame :
        time_init = time.time()
    
        explainer = shap.Explainer(self.getModel().predict, self.getX_all())
        valuesSHAP = pd.DataFrame().reindex_like(self.getX())

        colNames = list(self.getX().columns)
        for i in range(len(colNames)):
            colNames[i] = colNames[i] + "_shap"

        for i in range(len(self.getX())):
            shap_value = explainer(self.getX()[i : i + 1], max_evals=1400) #TOTDO : why 1400 ?
            valuesSHAP.iloc[i] = shap_value.values
            self.setProgress(int(self.getProgress() + 100 / len(self.getX())))
            self.sendProgress()

        valuesSHAP.columns = colNames
        return valuesSHAP
    
    @staticmethod
    def getExplanationType() -> int:
        return ExplanationMethod.SHAP

class LIMExplanation(ExplanationMethod):
    """
    LIME computation class.
    """
    
    def __init__(self, X:pd.DataFrame, X_all:pd.DataFrame, model:Model):
        super().__init__(ExplanationMethod.LIME, X, X_all, model)

    def compute(self) -> pd.DataFrame :
        time_init = time.time()

        # TODO : It seems we defined class_name in order to workl with California housing dataset. We should find a way to generalize this.
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(_X_all), feature_names=_X.columns, class_names=['price'], verbose=False, mode='regression')

        N = len(_X.shape)
        valuesLIME = pd.DataFrame(np.zeros((N, _X.shape[-1])))
        
        for j in range(N):
            l = []
            exp = explainer.explain_instance(
                _X.values[j], _model.predict
            )
            l = []
            taille = _X.shape[-1]
            for ii in range(taille):
                exp_map = exp.as_map()[0]
                l.extend(exp_map[ii][1] for jj in range(taille) if ii == exp_map[jj][0])
            
            valuesLIME.iloc[j] = pd.Series(l)
            _progress += 100 / len(_X)
            self.sendProgress()
        j = list(_X.columns)
        for i in range(len(j)): 
            j[i] = j[i] + "_lime"
        valuesLIME.columns = j
        return valuesLIME

    @staticmethod
    def getExplanationType() -> int:
        return ExplanationMethod.LIME

# ===========================================================
#         Projections / Dim Reductions implementations
# ===========================================================


class PCADimReduc(DimReducMethod):
    """
    PCA computation class.
    """
    def __init__(self, X:pd.DataFrame, X_all:pd.DataFrame, dimension : int = 2):
        DimReducMethod.__init__(self, DimReducMethod.PCA, dimension, X, X_all)

    def compute(self) -> pd.DataFrame:
        self.setProgres(0)
        pca = PCA(n_components=self.getDimension())
        pca.fit(self.getX())
        X_pca = pca.transform(self.getX())
        X_pca = pd.DataFrame(X_pca)
        # TODO : we need to iterated over the dataset in order to sendProgress messages to the GUI
        self.setProgres(100)
        self.sendProgress()
        return X_pca

    
class TSNEDimReduc(DimReducMethod):
    """
    T-SNE computation class.
    """

    def __init__(self, X:pd.DataFrame, X_all:pd.DataFrame, dimension : int = 2):
        DimReducMethod.__init__(self, DimReducMethod.TSNE, dimension, X, X_all)
    
    def compute(self) -> pd.DataFrame:
        self.setProgress(0)
        tsne = TSNE(n_components=self.getDimension())
        X_tsne = tsne.fit_transform(self.getX())
        X_tsne = pd.DataFrame(X_tsne)
        self.setProgress(100)
        self.sendProgress()
        return X_tsne
    
class UMAPDimReduc(DimReducMethod):
    """
    UMAP computation class.
    """
    def __init__(self, X:pd.DataFrame, X_all:pd.DataFrame, dimension : int = 2):
        DimReducMethod.__init__(self,DimReducMethod.UMAP, dimension, X, X_all)

    def compute(self) -> pd.DataFrame:
        self.setProgress(0)
        reducer = umap.UMAP(n_components=self.getDimension())
        embedding = reducer.fit_transform(set.getX())
        embedding = pd.DataFrame(embedding)
        self.setProgress(100)
        self.sendProgress()
        return embedding

class PaCMAPDimReduc(DimReducMethod):
    """
    PaCMAP computation class.

    __paramDict : dict, 
        Paramters for the PaCMAP algorithm
        Keys : "neighbours", "MN_ratio", "FP_ratio"


    """

    def __init__(self, X:pd.DataFrame, X_all:pd.DataFrame, dimension : int = 2, **kwargs):
        
        _paramDict = dict()
        if kwargs is not None :
            if "neighbours" in kwargs:
                    _paramDict.update({"neighbours": kwargs["neighbours"]})
            if "MN_ratio" in kwargs :
                    _paramDict.update({"MN_ratio": kwargs["MN_ratio"]})
            if "FP_ratio" in kwargs :
                    _paramDict.update({"FP_ratio": kwargs["FP_ratio"]})
        
        DimReducMethod.__init__(self, DimReducMethod.PaCMAP, dimension, X, X_all)


    
    def compute(self, *args):
        # compute fonction allows to define parmameters for the PaCMAP algorithm
        if len(args) == 3 :
                _paramDict.update({"neighbours": args[0]})
                _paramDict.update({"MN_ratio": args[1]})
                _paramDict.update({"FP_ratio": args[2]})
                reducer = pacmap.PaCMAP(
                    n_components=self.getDimension(),
                    n_neighbors=_paramDict["neighbours"],
                    MN_ratio=_paramDict["MN_ratio"],
                    FP_ratio=_paramDict["FP_ratio"],
                    random_state=9,
                )
        else :
            reducer = pacmap.PaCMAP(
                    n_components=self.getDimension(),  
                    random_state=9,
                )
        embedding = reducer.fit_transform(self.getX(), init="pca")
        embedding = pd.DataFrame(embedding)
        return embedding


def computeProjection(X:pd.DataFrame, dimReducMethod:int, dimension : int, **kwargs) -> pd.DataFrame:
    
    if not DimReducMethod.isValidDimReducType(dimReducMethod) or not DimReducMethod.isValidDimNumber(dimension):
        raise ValueError("This projection is not valid!")
    
    if dimReducMethod == DimReducMethod.PCA :
        return PCADimReduc(X, None, dimension).compute()
    elif dimReducMethod == DimReducMethod.TSNE :
        return TSNEDimReduc(X, None, dimension).compute()
    elif dimReducMethod == DimReducMethod.UMAP :
        return UMAPDimReduc(X, None, dimension).compute()
    elif dimReducMethod == DimReducMethod.PaCMAP :
        if kwargs is None or len(kwargs) == 0 :
            return PaCMAPDimReduc(X, None, dimension).compute()
        else : 
            return PaCMAPDimReduc(X, None, dimension, **kwargs).compute()
    else :
        raise ValueError("This default projection is not valid!")


def compute2D3DProjections(X, dimReducMethod:int, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes projected values for VS or ES in 2 and 3D given a dimensionality reduction method.
    Returns a Tuple with values for 2D and 3D projections.
    """
    return (computeProjection(X, dimReducMethod, DimReducMethod.DIM_TWO, **kwargs), computeProjection(X, dimReducMethod, DimReducMethod.DIM_THREE, **kwargs))
  
        
        

def function_score(y, y_chap):
    y = np.array(y)
    y_chap = np.array(y_chap)
    return round(np.sqrt(sum((y - y_chap) ** 2) / len(y)), 3)

@staticmethod
def createBeeswarm(ds : Dataset, xds : ExplanationsDataset, explainationMth : int, variableIndex : int) -> Tuple: #255
    X = ds.getXValues(Dataset.CURRENT)
    y = ds.getYValues(Dataset.PREDICTED)
    XP = xds.getValues(explainationMth)

    def orderAscending(lst : list):
        positions = list(range(len(lst)))  # Create a list of initial positions
        positions.sort(key=lambda x: lst[x])
        l = []
        for i in range(len(positions)):
            l.append(positions.index(i))  # Sort positions by list items
        return l
    
    expValuesHistogram = [0] * len(XP)
    binNumES = 60
    garde_indice = []
    garde_valeur_y = []
    for i in range(binNumES):
        keepIndex.append([])
        keepYValue.append([])

    liste_scale = np.linspace(
        min(XP[variableIndex]), max(XP[variableIndex]), binNumES + 1
    )
    for i in range(len(Exp)):
        for j in range(binNumES):
            if (
                XP[variableIndex][i] >= liste_scale[j]
                and XP[variableIndex][i] <= liste_scale[j + 1]
            ):
                keepIndex[j].append(i)
                keepYValue[j].append(y[i])
                break
    for i in range(binNumES):
        l = orderAscending(keepYValue[i])
        for j in range(len(keepIndex[i])):
            ii = keepIndex[i][j]
            if l[j] % 2 == 0:
                expValuesHistogram[ii] = l[j]
            else:
                expValuesHistogram[ii] = -l[j]
    explainMarker = dict(
        size=4,
        opacity=0.6,
        color=X[nom_colonne],
        colorscale="Bluered_r",
        colorbar=dict(thickness=20, title=nom_colonne),
    )
    return [expValuesHistogram, explainMarker]




# def compute(self, X, explanation, projection, sub_models, display=True):
#     """
#     Function that computes the regions and starts the GUI.

#     Parameters
#     ----------
#     X : array-like
#         The data to compute the regions on.
#     explanation : int
#         The index of the sub_model to use for the explanation.
#     projection : int
#         The index of the sub_model to use for the projection.
#     sub_models : list
#         The list of the sub_models to choose from for each region. The only constraint is that sub_models must have a predict method.
#     display : bool, optional
#         If True, the interface is displayed. Else, You can access the interface with the attribute gui of the class. The default is True.
#     """
#     self.resetRegions()
#     self.saves = []
#     self.widget = None

#     self.computeRegions(X, explanation, projection, sub_models)
#     self.gui = GUI(self, explanation, projection, sub_models)
#     if display:
#         self.gui.display()  projection, sub_models)
#     self.gui = GUI(self, explanation, projection, sub_models)
#     if display:
#         self.gui.display()index o sub_mod [] explanation,
#     self.widget = None
#     self.computeRegions(X,
# els mu def .
#     self.saves =    """Regions()

#     self.resetis Trueault th in Thete .classrfac attribute gu thei of thee withes   accesscandispl  YouEls,eayed. pred is
#         If True, the interfaceict me.
#     display : bool, optionalthodt have a the sub_m ch only c is thatonstraintoose  ea Thech. regionfrom for toodelsf the sub_mod for projec.
#         The list of    sub_models : list
# tion theel to use to compute the regi use  explana.
#     pro Thejection : int
#        tion theforons on. to_model
#         The index of the sub    explanation : int
# ```python explanat
#         The dataion=True): st.
# -like
#     X : array    Parameters
#     ----------
# arts the GUI
#     """ regions and
#     Function that computes the, pr displayoje s_model,sub,ction
# def compute(self, X,
#     def computeDyadicClustering(self, explanation:str = "Imported", min_clusters:int = 3, automatic:bool = True, sub_models:bool = False):
#         """
#         Function that computes the dyadic-clustering.
#         Our dyadic-clustering (sometimes found as co-clusetring or bi-clustering), uses `mvlearn` and `skope-rules` to compute the clusters.

#         Parameters
#         ---------
#         explanation : str
#             The type of explanation to use.
#             The possible values are "Imported", "SHAP" and "LIME".
#         min_clusters : int
#             The minimum number of clusters to compute.
#         automatic : bool
#             If True, the number of clusters is computed automatically, respecting the minimum number of clusters.
#         sub_models : bool
#             If True, the best model for each region is computed. The possible models are the ones in the list sub_models.
#         """
#         if self.explainations[explanation] is None:
#             raise ValueError("You must compute the explanations before computing the dyadic-clustering!")
#         if min_clusters <2 or min_clusters > len(self.dataset.X):
#             raise ValueError("The minimum number of clusters must be between 2 and the number of observations!")
#         clusters, clusters_axis = function_auto_clustering(self.dataset.X, self.explainations[explanation], min_clusters, automatic)
#         self.regions = []
#         for i in range(len(clusters)):
#             self.regions.append(Potato(self, clusters[i]))
#             if sub_models:
#                 self.regions[i].sub_model["model"], self.regions[i].sub_model["score"] = _find_best_model(self.regions[i].data, self.regions[i].y, self.gui.sub_models)

#     def __find_best_model(self, X:pd.DataFrame, y:pd.Series, sub_models:list):
#         best_model = None
#         best_score = 0
#         for model in sub_models:
#             model.fit(X, y)
#             score = model.score(X, y)
#             if score > best_score:
#                 best_score = score
#                 best_model = model
#         return best_model.__class__.__name__, round(best_score, 4)

#     def __create_progress(self, titre:str):
#         widget = v.Col(
#             class_="d-flex flex-column align-center",
#             children=[
#                     v.Html(
#                         tag="h3",
#                         class_="mb-3",
#                         children=["Compute " + titre + " values"],
#                 ),
#                 v.ProgressLinear(
#                     style_="width: 80%",
#                     v_model=0,
#                     color="primary",
#                     height="15",
#                     striped=True,
#                 ),
#                 v.TextField(
#                     class_="w-100",
#                     style_="width: 100%",
#                     v_model = "0.00% [0/?] - 0m0s (estimated time : /min /s)",
#                     readonly=True,
#                 ),
#             ],
#         )
#         self.widget = widget

    
#     def computeSHAP(self, verbose:bool = True):
#         """
#         Computes the SHAP values of the dataset.

#         Parameters
#         ---------
#         verbose : bool
#             If True, a progress bar is displayed.

#         See also:
#         ---------
#         The Shap library on GitHub : https://github.com/shap/shap/tree/master
#         """
#         shap = compute.computationSHAP(self.dataset.X, self.dataset.X_all, self.dataset.model)
#         if verbose:
#             self.verbose = _create_progress("SHAP")
#             widgets.jslink((self.widget.children[1], "v_model"), (shap.progress_widget, "v_model"))
#             widgets.jslink((self.widget.children[2], "v_model"), (shap.text_widget, "v_model"))
#             display(self.widget)
#         self.explainations["SHAP"] = shap.compute()

#     def computeLIME(self, verbose:bool = True):
#         """
#         Computes the LIME values of the dataset.

#         Parameters
#         ---------
#         verbose : bool
#             If True, a progress bar is displayed.

#         See also:
#         ---------
#         The Lime library on GitHub : https://github.com/marcotcr/lime/tree/master
#         """
#         lime = compute.computationSHAP(self.dataset.X, self.dataset.X_all, self.dataset.model)
#         if verbose:
#             self.verbose = _create_progress("LIME")
#             widgets.jslink((self.widget.children[1], "v_model"), (lime.progress_widget, "v_model"))
#             widgets.jslink((self.widget.children[2], "v_model"), (lime.text_widget, "v_model"))
#             display(self.widget)
#         self.explainations["LIME"] = lime.compute()


# --------------- update figues (we're still in display method---------------------

def updateFigures(ds : Dataset, xds : ExplanationsDataset, explanationMethod : int, projectionVS : tuple, leftVSFigure :Widget, leftVSFigure3D : Widget, projectionES : tuple, rightESFigure : Widget, rightESFigure3D : Widget) :

    projValues = compute2D3DProjections(ds.getXValues(Dataset.CURRENT), projectionVS[0])

    if leftVSFigure is not None :
        with leftVSFigure.batch_update():
            leftVSFigure.data[0].x, leftVSFigure.data[0].y  = projValues[0][0], projValues[0][1]
    
    if leftVSFigure3D is not None :
        with leftVSFigure3D.batch_update():
            leftVSFigure3D.data[0].x, leftVSFigure3D.data[0].y, leftVSFigure3D.data[0].z = projValues[1][0], projValues[1][1], projValues[1][2]
                
    projValues = compute2D3DProjections(xds.getValues(explanationMethod), projectionES[0])

    if rightESFigure is not None :
        with rightESFigure.batch_update():
            rightESFigure.data[0].x, rightESFigure.data[0].y = projValues[0][0], projValues[0][1]

    if rightESFigure3D is not None :    
        with rightESFigure3D.batch_update():
            rightESFigure3D.data[0].x, rightESFigure3D.data[0].y, rightESFigure3D.data[0].z = projValues[1][0], projValues[1][1], projValues[1][2]

