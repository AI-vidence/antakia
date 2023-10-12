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

import mvlearn
from skrules import SkopeRules
import sklearn
import sklearn.cluster

# Imports for the explanations
import lime
import lime.lime_tabular
import shap

import logging
from logging import getLogger
from copy import deepcopy

from ipywidgets.widgets.widget import Widget

from antakia.data import Dataset, ExplanationDataset, LongTask, ExplanationMethod, DimReducMethod, Model
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
        logger.debug(f"SHAPExplanation.compute : starting ...")
        time_init = time.time()
    
        # explainer = shap.Explainer(self.get_model().predict)
        explainer = shap.Explainer(self.get_model().predict, self.get_X())
        # valuesSHAP = pd.DataFrame().reindex_like(self.get_X())
        self.set_progress(0)
        values_shap = explainer(self.get_X())
        self.set_progress(100)

        # colNames = list(self.get_X().columns)
        # for i in range(len(colNames)):
        #     colNames[i] = colNames[i] + "_shap"

        # X = self.get_X()

        # for i in range(len(X)):
        #     shap_value = explainer(X[i : i + 1], max_evals=1400) 
        #     valuesSHAP.iloc[i] = shap_value.values
        #     p = int(100*(i/len(X)))
        #     logger.debug("SHAPExplanation.compute : progress is {p}%")
        #     self.set_progress(p)

        # valuesSHAP.columns = colNames
        df = pd.DataFrame.from_records(values_shap.values)
        logger.debug(f"SHAPExplanation.compute : returns a {type(df)} with {df.shape}")
        return df
    
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
        logger.debug(f"LIMEExplanation.compute : starting ...")
        time_init = time.time()

        # TODO : It seems we defined class_name in order to work with California housing dataset. We should find a way to generalize this.
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(self.get_X()), feature_names=self.get_X().columns, class_names=['price'], verbose=False, mode='regression')

        N = len(self.get_X().shape[0])
        values_lime = pd.DataFrame(np.zeros((N, self.get_X().shape[-1])))
        
        for j in range(N):
            l = []
            exp = explainer.explain_instance(
                self.get_X().values[j], self._model.predict
            )
            l = []
            size = self.get_X().shape[-1]
            for ii in range(size):
                exp_map = exp.as_map()[0]
                l.extend(exp_map[ii][1] for jj in range(size) if ii == exp_map[jj][0])
            
            values_lime.iloc[j] = pd.Series(l)
            self._progress += 100 / len(self.get_X())
            self.set_progress()
        j = list(self.get_X().columns)
        for i in range(len(j)): 
            j[i] = j[i] + "_lime"
        values_lime.columns = j
        logger.debug(f"LIMExplanation.compute : returns a {type(values_lime)} with {values_lime.shape}")
        return values_lime

    @staticmethod
    def get_explanation_method() -> int:
        return ExplanationMethod.LIME
# --------------------------------------------------------------------------

def compute_explanations(X:pd.DataFrame, model : Model, explanation_method:int) -> pd.DataFrame:
    """ Generaic method to compute explanations, SHAP or LIME
    """
    if explanation_method == ExplanationMethod.SHAP :
        return SHAPExplanation(X, model).compute()
    elif explanation_method == ExplanationMethod.LIME :
        return LIMExplanation(X, model).compute()
    else :
        raise ValueError(f"This explanation method {explanation_method} is not valid!")


# ===========================================================
#         Projections / Dim Reductions implementations
# ===========================================================


class PCADimReduc(DimReducMethod):
    """
    PCA computation class.
    """
    def __init__(self, base_space : int, X:pd.DataFrame, X_all:pd.DataFrame, dimension : int = 2):
        DimReducMethod.__init__(self, base_space, DimReducMethod.PCA, dimension, X)

    def compute(self) -> pd.DataFrame:
        self.set_progress(0)
        pca = PCA(n_components=self.get_dimension())
        pca.fit(self.get_X())
        X_pca = pca.transform(self.get_X())
        X_pca = pd.DataFrame(X_pca)
        # TODO : we need to iterated over the dataset in order to set_progress messages to the GUI
        self.set_progress(100)
        return X_pca

    
class TSNEDimReduc(DimReducMethod):
    """
    T-SNE computation class.
    """

    def __init__(self, baseSpace : int, X:pd.DataFrame,  dimension : int = 2):
        DimReducMethod.__init__(self, baseSpace, DimReducMethod.TSNE, dimension, X)
    
    def compute(self) -> pd.DataFrame:
        self.set_progress(0)
        tsne = TSNE(n_components=self.get_dimension())
        X_tsne = tsne.fit_transform(self.get_X())
        X_tsne = pd.DataFrame(X_tsne)
        self.set_progress(100)
        return X_tsne
    
class UMAPDimReduc(DimReducMethod):
    """
    UMAP computation class.
    """
    def __init__(self, baseSpace : int, X:pd.DataFrame, dimension : int = 2):
        DimReducMethod.__init__(self, baseSpace, DimReducMethod.UMAP, dimension, X)

    def compute(self) -> pd.DataFrame:
        self.set_progress(0)
        reducer = umap.UMAP(n_components=self.get_dimension())
        embedding = reducer.fit_transform(self.get_X())
        embedding = pd.DataFrame(embedding)
        self.set_progress(100)
        return embedding

class PaCMAPDimReduc(DimReducMethod):
    """
    PaCMAP computation class.

    _param_dict : dict, 
        Paramters for the PaCMAP algorithm
        Keys : "neighbours", "MN_ratio", "FP_ratio"
    """

    def __init__(self, base_space: int, X:pd.DataFrame, dimension: int = 2, **kwargs):
        self._paramDict = dict()
        if kwargs is not None :
            if "neighbours" in kwargs:
                    self._param_dict.update({"neighbours": kwargs["neighbours"]})
            if "MN_ratio" in kwargs :
                    self._param_dict.update({"MN_ratio": kwargs["MN_ratio"]})
            if "FP_ratio" in kwargs :
                    self._param_dict.update({"FP_ratio": kwargs["FP_ratio"]})
        DimReducMethod.__init__(self, base_space, DimReducMethod.PaCMAP, dimension, X)

    
    def compute(self, *args) -> pd.DataFrame :
        # compute fonction allows to define parmameters for the PaCMAP algorithm
        self.set_progress(0)
        if len(args) == 3 :
                self._param_dict.update({"neighbours": args[0]})
                self._param_dict.update({"MN_ratio": args[1]})
                self._param_dict.update({"FP_ratio": args[2]})
                reducer = pacmap.PaCMAP(
                    n_components=self.get_dimension(),
                    n_neighbors=self._param_dict["neighbours"],
                    MN_ratio=self._param_dict["MN_ratio"],
                    FP_ratio=self._param_dict["FP_ratio"],
                    random_state=9,
                )
        else :
            reducer = pacmap.PaCMAP(
                    n_components=self.get_dimension(),  
                    random_state=9,
                )

        embedding = reducer.fit_transform(self.get_X(), init="pca")
        embedding = pd.DataFrame(embedding)
        self.set_progress(100)
        return embedding


# --------------------------------------------------------------------------

def compute_projection(baseSpace : int, X:pd.DataFrame, dimReducMethod:int, dimension : int, **kwargs) -> pd.DataFrame:
    
    if not DimReducMethod.is_valid_dimreduc_method(dimReducMethod) or not DimReducMethod.is_valid_dim_number(dimension):
        raise ValueError("Cannot compute proj method #", dimReducMethod, " in ", dimension, " dimensions")

    projValues = None

    if dimReducMethod == DimReducMethod.PCA :    
        projValues =  PCADimReduc(baseSpace, X, dimension).compute()
    elif dimReducMethod == DimReducMethod.TSNE :
        projValues =  TSNEDimReduc(baseSpace, X, dimension).compute()
    elif dimReducMethod == DimReducMethod.UMAP :
        projValues =  UMAPDimReduc(baseSpace, X, dimension).compute()
    elif dimReducMethod == DimReducMethod.PaCMAP :
        if kwargs is None or len(kwargs) == 0 :
            projValues =  PaCMAPDimReduc(baseSpace, X, dimension).compute()
        else : 
            logger.debug(f"computeProjection: PaCMAPDimReduc with kwargs {kwargs}")
            projValues =  PaCMAPDimReduc(baseSpace, X, dimension, **kwargs).compute()
    else :
        raise ValueError(f"This projection type {dimReducMethod} is not valid!")

    # logger.debug(f"computeProjection: returns a {type(projValues)} with {projValues.shape} ({dimension}D) in {DimReducMethod.getBaseSpaceAsStr(baseSpace)} baseSpace")
    
    return projValues


def auto_cluster(X: pd.DataFrame, shap_values: pd.DataFrame, n_clusters: int, default: bool) -> list:
    # TODO : we must return a list of Potato
    m_kmeans = mvlearn.cluster.MultiviewKMeans(n_clusters=n_clusters, random_state=9)
    a_list = m_kmeans.fit_predict([X, shap_values])
    clusters_number = 0
    if default is True:
        X_train = pd.DataFrame(X.copy())
        recall_min = 0.8
        precision_min = 0.8
        max_ = max(a_list) + 1
        l_copy = deepcopy(a_list)
        for i in range(n_clusters):
            y_train = [1 if x == i else 0 for x in l_copy]
            indices = [i for i in range(len(y_train)) if y_train[i] == 1]
            skope_rules_clf = SkopeRules(
                feature_names=X_train.columns,
                random_state=42,
                n_estimators=5,
                recall_min=recall_min,
                precision_min=precision_min,
                max_depth_duplication=0,
                max_samples=1.0,
                max_depth=3,
            )
            skope_rules_clf.fit(X_train, y_train)
            if len(skope_rules_clf.rules_) == 0:
                k = _find_best_k(X, indices, recall_min, precision_min)
                clusters_number += k
                # kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=9)
                agglo = sklearn.cluster.AgglomerativeClustering(n_clusters=k)
                agglo.fit(X.iloc[indices])
                labels = np.array(agglo.labels_) + max_
                max_ += k
                a_list[indices] = labels
            else:
                clusters_number += 1
        a_list = _reset_list(a_list)
    a_list = list(np.array(a_list) - min(a_list))
    return _invert_list(a_list, max(a_list) + 1), a_list


def _invert_list(a_list: list, size: int) -> list:
    newList = [[] for _ in range(size)]
    for i in range(len(a_list)):
        newList[a_list[i]].append(i)
    return newList


def _reset_list(a_list: list) -> list:
    a_list = list(a_list)
    for i in range(max(a_list) + 1):
        if a_list.count(i) == 0:
            a_list = list(np.array(a_list) - 1)
    return a_list


def _find_best_k(X: pd.DataFrame, indices: list, recall_min: float, precision_min: float) -> int:
    recall_min = 0.7
    precision_min = 0.7
    new_X = X.iloc[indices]
    ind_f = 2
    for i in range(2, 9):
        # kmeans = sklearn.cluster.KMeans(n_clusters=i, random_state=9, n_init="auto")
        agglo = sklearn.cluster.AgglomerativeClustering(n_clusters=i)
        agglo.fit(new_X)
        labels = agglo.labels_
        a = True
        for j in range(max(labels) + 1):
            y = []
            for k in range(len(X)):
                if k in indices and labels[indices.index(k)] == j:
                    y.append(1)
                else:
                    y.append(0)
            skope_rules_clf = SkopeRules(
                feature_names=new_X.columns,
                random_state=42,
                n_estimators=5,
                recall_min=recall_min,
                precision_min=precision_min,
                max_depth_duplication=0,
                max_samples=1.0,
                max_depth=3,
            )
            skope_rules_clf.fit(X, y)
            if len(skope_rules_clf.rules_) == 0:
                ind_f = i - 1
                a = False
                break
        if not a:
            break
    return 2 if ind_f == 1 else ind_f