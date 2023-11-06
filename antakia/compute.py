import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# umpa fires NumbaDeprecationWarning warning, so :
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=UserWarning)

import umap
import pacmap
import lime
import lime.lime_tabular
import shap
from copy import deepcopy, copy


import mvlearn
from skrules import SkopeRules
import sklearn
import sklearn.cluster

from ipywidgets.widgets.widget import Widget

from antakia.data import LongTask, ExplanationMethod, DimReducMethod
from antakia.utils import conf_logger
import logging
from logging import getLogger
logger = logging.getLogger(__name__)
conf_logger(logger)



# ===========================================================
#              Explanations implementations
# ===========================================================

class SHAPExplanation(ExplanationMethod):
    """
    SHAP computation class.
    """

    def __init__(self, X:pd.DataFrame, model, progress_updated: callable = None):
        super().__init__(ExplanationMethod.SHAP, X, model, progress_updated)

    def compute(self) -> pd.DataFrame :
        #TOPO split this !!
        self.publish_progress(0)
        explainer = shap.Explainer(self.model.predict, self.X)
        shap_values = explainer(self.X)
        self.publish_progress(100)
        return pd.DataFrame(shap_values)

class LIMExplanation(ExplanationMethod):
    """
    LIME computation class.
    """
    
    def __init__(self, X:pd.DataFrame, model, progress_updated: callable = None):
        super().__init__(ExplanationMethod.LIME, X, model, progress_updated)

    def compute(self) -> pd.DataFrame :
        self.publish_progress(0)

        # TODO : It seems we defined class_name in order to work with California housing dataset. We should find a way to generalize this.
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(self.get_X()), feature_names=self.X.columns, class_names=['price'], verbose=False, mode='regression')

        N = len(self.get_X().shape[0])
        values_lime = pd.DataFrame(np.zeros((N, self.X.shape[-1])))
        
        for j in range(N):
            l = []
            exp = explainer.explain_instance(
                self.X.values[j], self.model.predict
            )
            l = []
            size = self.X.shape[-1]
            for ii in range(size):
                exp_map = exp.as_map()[0]
                l.extend(exp_map[ii][1] for jj in range(size) if ii == exp_map[jj][0])
            
            values_lime.iloc[j] = pd.Series(l)
            self.progress += 100 / len(self.X)
            self.publish_progress()
        j = list(self.X.columns)
        for i in range(len(j)): 
            j[i] = j[i] + "_lime"
        values_lime.columns = j
        self.publish_progress(100)
        return values_lime

# --------------------------------------------------------------------------

def compute_explanations(X:pd.DataFrame, model, explanation_method:int, callback:callable) -> pd.DataFrame:
    """ Generic method to compute explanations, SHAP or LIME
    """
    if explanation_method == ExplanationMethod.SHAP :
        return SHAPExplanation(X, model, callback).compute()
    elif explanation_method == ExplanationMethod.LIME :
        return LIMExplanation(X, model, callback).compute()
    else :
        raise ValueError(f"This explanation method {explanation_method} is not valid!")


# ===========================================================
#         Projections / Dim Reductions implementations
# ===========================================================


class PCADimReduc(DimReducMethod):
    """
    PCA computation class.
    """
    def __init__(self, X:pd.DataFrame, dimension: int = 2, callback : callable = None):
        DimReducMethod.__init__(self, DimReducMethod.PCA, dimension, X, callback)

    def compute(self) -> pd.DataFrame:
        self.publish_progress(0)

        pca = PCA(n_components=self.get_dimension())
        pca.fit(self.X)
        X_pca = pca.transform(self.X)
        X_pca = pd.DataFrame(X_pca)

        self.publish_progress(100)
        return X_pca

    
class TSNEDimReduc(DimReducMethod):
    """
    T-SNE computation class.
    """

    def __init__(self, X:pd.DataFrame,  dimension : int = 2, callback : callable = None):
        DimReducMethod.__init__(self, DimReducMethod.TSNE, dimension, X, callback)
    
    def compute(self) -> pd.DataFrame:
        self.publish_progress(0)

        tsne = TSNE(n_components=self.get_dimension())
        X_tsne = tsne.fit_transform(self.X)
        X_tsne = pd.DataFrame(X_tsne)

        self.publish_progress(100)
        return X_tsne
    
class UMAPDimReduc(DimReducMethod):
    """
    UMAP computation class.
    """
    def __init__(self, X:pd.DataFrame, dimension : int = 2, callback : callable = None):
        DimReducMethod.__init__(self, DimReducMethod.UMAP, dimension, X, callback)

    def compute(self) -> pd.DataFrame:
        self.publish_progress(0)

        reducer = umap.UMAP(n_components=self.get_dimension())
        embedding = reducer.fit_transform(self.X)
        embedding = pd.DataFrame(embedding)

        self.publish_progress(100)
        return embedding

class PaCMAPDimReduc(DimReducMethod):
    """
    PaCMAP computation class.

    _param_dict : dict, 
        Paramters for the PaCMAP algorithm
        Keys : "neighbours", "MN_ratio", "FP_ratio"
    """

    def __init__(self, X:pd.DataFrame, dimension: int = 2, callback:callable = None, **kwargs):
        self._paramDict = dict()
        if kwargs is not None :
            if "neighbours" in kwargs:
                    self._param_dict.update({"neighbours": kwargs["neighbours"]})
            if "MN_ratio" in kwargs :
                    self._param_dict.update({"MN_ratio": kwargs["MN_ratio"]})
            if "FP_ratio" in kwargs :
                    self._param_dict.update({"FP_ratio": kwargs["FP_ratio"]})
        DimReducMethod.__init__(self, DimReducMethod.PaCMAP, dimension, X, callback)

    
    def compute(self, *args) -> pd.DataFrame :
        self.publish_progress(0)
        # compute fonction allows to define parmameters for the PaCMAP algorithm
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

        embedding = reducer.fit_transform(self.X, init="pca")
        embedding = pd.DataFrame(embedding)

        self.publish_progress(100)
        return embedding


# --------------------------------------------------------------------------

def compute_projection(X: pd.DataFrame, dimreduc_method:int, dimension : int, callback: callable= None, **kwargs) -> pd.DataFrame:
    
    if not DimReducMethod.is_valid_dimreduc_method(dimreduc_method) or not DimReducMethod.is_valid_dim_number(dimension):
        raise ValueError("Cannot compute proj method #", dimreduc_method, " in ", dimension, " dimensions")

    proj_values = None

    if dimreduc_method == DimReducMethod.PCA :    
        proj_values =  PCADimReduc(X, dimension, callback).compute()
    elif dimreduc_method == DimReducMethod.TSNE :
        proj_values =  TSNEDimReduc(X, dimension, callback).compute()
    elif dimreduc_method == DimReducMethod.UMAP :
        proj_values =  UMAPDimReduc(X, dimension, callback).compute()
    elif dimreduc_method == DimReducMethod.PaCMAP :
        if kwargs is None or len(kwargs) == 0 :
            proj_values =  PaCMAPDimReduc(X, dimension, callback).compute()
        else: 
            logger.debug(f"compute_projection: PaCMAPDimReduc with kwargs {kwargs}")
            proj_values =  PaCMAPDimReduc(X, dimension, callback, **kwargs).compute()

    return proj_values


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