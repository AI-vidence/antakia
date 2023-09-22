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
from logging import getLogger

from ipywidgets.widgets.widget import Widget

from antakia.data import Dataset, ExplanationsDataset, LongTask, ExplanationMethod, DimReducMethod, Model
from antakia.utils import confLogger

logger = logging.getLogger(__name__)
handler = confLogger(logger)
handler.clear_logs()
handler.show_logs()


# ===========================================================
#              Explanations implementations
# ===========================================================

class SHAPExplanation(ExplanationMethod):
    """
    SHAP computation class.
    """

    def __init__(self, X:pd.DataFrame, model:Model):
        super().__init__(ExplanationMethod.SHAP, X, model)

    def compute(self) -> pd.DataFrame :
        time_init = time.time()
    
        # explainer = shap.Explainer(self.getModel().predict)
        explainer = shap.Explainer(self.getModel().predict, self.getX())
        # valuesSHAP = pd.DataFrame().reindex_like(self.getX())
        self.setProgress(0)
        logger.debug(f"SHAPExplanation.compute : set progress = 0")
        valuesSHAP = explainer(self.getX())
        self.setProgress(100)
        logger.debug(f"SHAPExplanation.compute : set progress = 100")

        # colNames = list(self.getX().columns)
        # for i in range(len(colNames)):
        #     colNames[i] = colNames[i] + "_shap"

        # X = self.getX()

        # for i in range(len(X)):
        #     shap_value = explainer(X[i : i + 1], max_evals=1400) 
        #     valuesSHAP.iloc[i] = shap_value.values
        #     p = int(100*(i/len(X)))
        #     logger.debug("SHAPExplanation.compute : progress is {p}%")
        #     self.setProgress(p)

        # valuesSHAP.columns = colNames
        return valuesSHAP
    
    @staticmethod
    def getExplanationType() -> int:
        return ExplanationMethod.SHAP

class LIMExplanation(ExplanationMethod):
    """
    LIME computation class.
    """
    
    def __init__(self, X:pd.DataFrame, model:Model):
        super().__init__(ExplanationMethod.LIME, X, model)

    def compute(self) -> pd.DataFrame :
        time_init = time.time()

        # TODO : It seems we defined class_name in order to work with California housing dataset. We should find a way to generalize this.
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(self.getX()), feature_names=self.getX().columns, class_names=['price'], verbose=False, mode='regression')

        N = len(self.getX().shape[0])
        valuesLIME = pd.DataFrame(np.zeros((N, self.getX().shape[-1])))
        
        for j in range(N):
            l = []
            exp = explainer.explain_instance(
                self.getX().values[j], _model.predict
            )
            l = []
            taille = self.getX().shape[-1]
            for ii in range(taille):
                exp_map = exp.as_map()[0]
                l.extend(exp_map[ii][1] for jj in range(taille) if ii == exp_map[jj][0])
            
            valuesLIME.iloc[j] = pd.Series(l)
            _progress += 100 / len(self.getX())
            self.setProgress()
        j = list(self.getX().columns)
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
    def __init__(self, baseSpace : int, X:pd.DataFrame, X_all:pd.DataFrame, dimension : int = 2):
        DimReducMethod.__init__(self, baseSpace, DimReducMethod.PCA, dimension, X)

    def compute(self) -> pd.DataFrame:
        self.setProgress(0)
        pca = PCA(n_components=self.getDimension())
        pca.fit(self.getX())
        X_pca = pca.transform(self.getX())
        X_pca = pd.DataFrame(X_pca)
        # TODO : we need to iterated over the dataset in order to setProgress messages to the GUI
        self.setProgress(100)
        return X_pca

    
class TSNEDimReduc(DimReducMethod):
    """
    T-SNE computation class.
    """

    def __init__(self, baseSpace : int, X:pd.DataFrame,  dimension : int = 2):
        DimReducMethod.__init__(self, baseSpace, DimReducMethod.TSNE, dimension, X)
    
    def compute(self) -> pd.DataFrame:
        self.setProgress(0)
        tsne = TSNE(n_components=self.getDimension())
        X_tsne = tsne.fit_transform(self.getX())
        X_tsne = pd.DataFrame(X_tsne)
        self.setProgress(100)
        return X_tsne
    
class UMAPDimReduc(DimReducMethod):
    """
    UMAP computation class.
    """
    def __init__(self, baseSpace : int, X:pd.DataFrame, dimension : int = 2):
        DimReducMethod.__init__(self, baseSpace, DimReducMethod.UMAP, dimension, X)

    def compute(self) -> pd.DataFrame:
        self.setProgress(0)
        reducer = umap.UMAP(n_components=self.getDimension())
        embedding = reducer.fit_transform(self.getX())
        embedding = pd.DataFrame(embedding)
        self.setProgress(100)
        return embedding

class PaCMAPDimReduc(DimReducMethod):
    """
    PaCMAP computation class.

    __paramDict : dict, 
        Paramters for the PaCMAP algorithm
        Keys : "neighbours", "MN_ratio", "FP_ratio"


    """

    def __init__(self, baseSpace : int, X:pd.DataFrame, dimension : int = 2, **kwargs):
        self._paramDict = dict()
        if kwargs is not None :
            if "neighbours" in kwargs:
                    _paramDict.update({"neighbours": kwargs["neighbours"]})
            if "MN_ratio" in kwargs :
                    _paramDict.update({"MN_ratio": kwargs["MN_ratio"]})
            if "FP_ratio" in kwargs :
                    _paramDict.update({"FP_ratio": kwargs["FP_ratio"]})
        DimReducMethod.__init__(self, baseSpace, DimReducMethod.PaCMAP, dimension, X)


    
    def compute(self, *args):
        # compute fonction allows to define parmameters for the PaCMAP algorithm
        self.setProgress(0)
        if len(args) == 3 :
                self._paramDict.update({"neighbours": args[0]})
                self._paramDict.update({"MN_ratio": args[1]})
                self._paramDict.update({"FP_ratio": args[2]})
                reducer = pacmap.PaCMAP(
                    n_components=self.getDimension(),
                    n_neighbors=self._paramDict["neighbours"],
                    MN_ratio=self._paramDict["MN_ratio"],
                    FP_ratio=self._paramDict["FP_ratio"],
                    random_state=9,
                )
        else :
            reducer = pacmap.PaCMAP(
                    n_components=self.getDimension(),  
                    random_state=9,
                )
        embedding = reducer.fit_transform(self.getX(), init="pca")
        embedding = pd.DataFrame(embedding)
        self.setProgress(100)
        return embedding


def computeProjection(baseSpace : int, X:pd.DataFrame, dimReducMethod:int, dimension : int, **kwargs) -> pd.DataFrame:
    
    if not DimReducMethod.isValidDimReducType(dimReducMethod) or not DimReducMethod.isValidDimNumber(dimension):
        raise ValueError("Cannot compute proj method #", dimReducMethod, " in ", dimension, " dimensions")

    projValues = None

    if dimReducMethod == DimReducMethod.PCA :    
        projValues =  PCADimReduc(baseSpace, X, dimension).compute()
    elif dimReducMethod == DimReducMethod.TSNE :
        projValues =  TSNEDimReduc(baseSpace, X, dimension).compute()
    elif dimReducMethod == DimReducMethod.UMAP :
        projValues =  UMAPDimReduc(baseSpace, X, dimension).compute()
    elif dimReducMethod == DimReducMethod.PaCMAP :
        logger.debug(f"PaCMAPDimReduc.compute : baseSpace = {baseSpace}")
        if kwargs is None or len(kwargs) == 0 :
            projValues =  PaCMAPDimReduc(baseSpace, X, dimension).compute()
        else : 
            projValues =  PaCMAPDimReduc(baseSpace, X, dimension, **kwargs).compute()
    else :
        raise ValueError(f"This projection type {dimReducMethod} is not valid!")

    return projValues
       
        

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


# --------------- update figues (we're still in display method---------------------

def updateFigures(ds : Dataset, xds : ExplanationsDataset, explanationMethod : int, projectionVS : tuple, leftVSFigure :Widget, leftVSFigure3D : Widget, projectionES : tuple, rightESFigure : Widget, rightESFigure3D : Widget) :


    # We update if needed on the left / VS
    projVSValues = ds.getXProjValues(projectionVS[0], projectionVS[1])

    if projVSValues is None :
        projVSValues = computeProjection(ds.getXValues(Dataset.CURRENT), projectionVS[0], projectionVS[1])


   # We update if needed on the right / ES
    projESValues = xds.getValues(explanationMethod, projectionES[0], projectionES[1])
    if projESValues is None :
        projESValues = computeProjection(ds.getXValues(Dataset.CURRENT), projectionVS[0], projectionES[1])
   

    # We now update the figures :
    if projectionVS[1]==DimReducMethod.DIM_TWO and leftVSFigure is not None :
        with leftVSFigure.batch_update():
            leftVSFigure.data[0].x, leftVSFigure.data[0].y  = projVSValues[0], projVSValues[1]
    elif projectionVS[1]==DimReducMethod.DIM_THREE and leftVSFigure3D is not None :
            with leftVSFigure3D.batch_update():
                leftVSFigure3D.data[0].x, leftVSFigure3D.data[0].y, leftVSFigure3D.data[0].z = projVSValues[0], projVSValues[1], projVSValues[2]

    if projectionES[1]==DimReducMethod.DIM_TWO and rightESFigure is not None :
        with rightESFigure.batch_update():
            rightESFigure.data[0].x, rightESFigure.data[0].y = projESValues[0], projESValues[1]
    elif projectionES[1]==DimReducMethod.DIM_THREE and rightESFigure3D is not None :
            with rightESFigure3D.batch_update():
                rightESFigure3D.data[0].x, rightESFigure3D.data[0].y, rightESFigure3D.data[0].z = projESValues[0], projESValues[1], projESValues[2]
