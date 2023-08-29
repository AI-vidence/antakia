import pandas as pd
import numpy as np
import threading
import time
from abc import ABC, abstractmethod
import ipyvuetify as v

# Imports for the dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import pacmap

# Imports for the explanations
import lime
import lime.lime_tabular
import shap

class LongTask(ABC):
    '''
    Abstract class to compute long tasks, often in a separate thread.

    Attributes
    ----------
    X : pandas dataframe
        The dataframe containing the data to explain.
    X_all : pandas dataframe
        The dataframe containing the entire dataset, in order for the explanations to be computed.
    model : model object
        The "black-box" model to explain.
    '''
    def __init__(self, X, X_all, model):
        self.X = X
        self.X_all = X_all
        self.model = model

        self.progress = 0
        self.progress_widget = v.Textarea(v_model=0)
        self.text_widget = v.Textarea(v_model=None)
        self.done_widget = v.Textarea(v_model=True)
        self.value = None
        self.thread = None
    
    @abstractmethod
    def compute(self):
        """
        Method to compute the long task.
        """
        pass

    def compute_in_thread(self):
        """
        Method to compute the long task in a separate thread.
        """
        self.thread = threading.Thread(target=self.compute)
        self.thread.start()

    def generation_texte(self, i, tot, time_init, progress):
        progress = float(progress)
        # allows to generate the progress text of the progress bar
        time_now = round((time.time() - time_init) / progress * 100, 1)
        minute = int(time_now / 60)
        seconde = time_now - minute * 60
        minute_passee = int((time.time() - time_init) / 60)
        seconde_passee = int((time.time() - time_init) - minute_passee * 60)
        return (
            str(round(progress, 1))
            + "%"
            + " ["
            + str(i + 1)
            + "/"
            + str(tot)
            + "] - "
            + str(minute_passee)
            + "m"
            + str(seconde_passee)
            + "s (estimated time : "
            + str(minute)
            + "min "
            + str(round(seconde))
            + "s)"
        )


# ===========================================================
#                   Explanations
# ===========================================================

class ExplainationMethod(LongTask):
    """
    Abstract class (Long Task) to compute explaination values for the Explanation Space (ES)
    """

    # Class attributes : ExplainationMethod types
    SHAP = 0
    LIME = 1
    OTHER = 2 

    @abstractmethod
    def compute(self) -> pd.DataFrame :
        pass
    
    @abstractmethod
    def getType(self) -> int :
        """
        Returns the type of the explained values
        """
        pass


class SHAPExplaination(ExplainationMethod):
    """
    SHAP computation class.
    """
    def compute(self) -> pd.DataFrame :
        self.progress = 0
        self.done_widget.v_model = "primary"
        self.text_widget.v_model = None
        time_init = time.time()
        explainer = shap.Explainer(self.model.predict, self.X_all)
        valuesSHAP = pd.DataFrame().reindex_like(self.X)
        j = list(self.X.columns)
        for i in range(len(j)):
            j[i] = j[i] + "_shap"
        for i in range(len(self.X)):
            shap_value = explainer(self.X[i : i + 1], max_evals=1400)
            valuesSHAP.iloc[i] = shap_value.values
            self.progress += 100 / len(self.X)
            self.progress_widget.v_model = self.progress
            self.text_widget.v_model = self.generation_texte(i, len(self.X), time_init, self.progress_widget.v_model)
        valuesSHAP.columns = j
        self.value = valuesSHAP
        self.done_widget.v_model = "success"
        return valuesSHAP
    
    def getType(self) -> int:
        return SHAP

class LIMExplaination(ExplainationMethod):
    """
    LIME computation class.
    """
    
    def compute(self) -> pd.DataFrame :
        self.done_widget.v_model = "primary"
        self.progress_widget.v_model = 0
        self.text_widget.v_model = None
        time_init = time.time()
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(self.X_all), feature_names=self.X.columns, class_names=['price'], verbose=False, mode='regression')
        N = len(self.X)
        valuesLIME = pd.DataFrame(np.zeros((N, self.X.shape[-1])))
        l = []
        for j in range(N):
            l = []
            exp = explainer.explain_instance(
                self.X.values[j], self.model.predict
            )
            l = []
            taille = self.X.shape[-1]
            for ii in range(taille):
                exp_map = exp.as_map()[0]
                l.extend(exp_map[ii][1] for jj in range(taille) if ii == exp_map[jj][0])
            valuesLIME.iloc[j] = l
            self.progress_widget.v_model  += 100 / len(self.X)
            self.text_widget.v_model = self.generation_texte(j, len(self.X), time_init, self.progress_widget.v_model)
        j = list(self.X.columns)
        for i in range(len(j)):
            j[i] = j[i] + "_shap"
        valuesLIME.columns = j
        self.value = valuesLIME
        self.done_widget.v_model = "success"
        return valuesLIME

    def getType(self) -> int:
        return LIME

# ===========================================================
#                   Projections
# ===========================================================

class DimensionalityReduction(LongTask):
    """
    Class that allows to reduce the dimensionality of the data.
    """

    PCA = 1
    TSNE = 2
    UMAP = 3
    PacMAP = 4

    DIM_ALL = -1
    DIM_TWO = 2
    DIM_THREE = 3

    def __init__(self):
        """
        Constructor of the class DimensionalityReduction.
        """
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def getType(self) -> dict :
        """
        Returns the type and the dimension in a dict
        'type' and 'dim' are the keys
        """
        pass        
        
class PCADimReduc(DimensionalityReduction):
    """
    PCA computation class.
    """

    def __init__(self):
        """
        Constructor of the class PCADimReduc.
        """
        pass        

    def compute(self, X, n, default=True):

        self.dim = n
        # definition of the method PCA, used for the EE and the EV
        if default:
            pca = PCA(n_components=n)
        pca.fit(X)
        X_pca = pca.transform(X)
        X_pca = pd.DataFrame(X_pca)
        return X_pca

    def getType(self) -> dict:
        if n != 2 and n != 3:
            return {'type': DimensionalityReduction.PCA, 'dim': -1}
        else:
            return {'type': DimensionalityReduction.PCA, 'dim': n}
    
class TSNEDimReduc(DimensionalityReduction):
    """
    T-SNE computation class.
    """
    def compute(self, X, n, default=True):
        self.dim = n # definition of the method TSNE, used for the EE and the EV
        if default:
            tsne = TSNE(n_components=n)
        X_tsne = tsne.fit_transform(X)
        X_tsne = pd.DataFrame(X_tsne)
        return X_tsne

    def getType(self) -> dict:
        if n != 2 and n != 3:
            return {'type': DimensionalityReduction.TSNE, 'dim': -1}
        else:
            return {'type': DimensionalityReduction.TSNE, 'dim': n}
    
class UMAPDimReduc(DimensionalityReduction):
    """
    UMAP computation class.
    """
    def compute(self, X, n, default=True):
        self.n = n
        if default:
            reducer = umap.UMAP(n_components=n)
        embedding = reducer.fit_transform(X)
        embedding = pd.DataFrame(embedding)
        return embedding

    def getType(self) -> dict:
        if n != 2 and n != 3:
            return {'type': DimensionalityReduction.UMAP, 'dim': -1}
        else:
            return {'type': DimensionalityReduction.UMAP,'dim': n}

class PaCMAPDimReduc(DimensionalityReduction):
    """
    PaCMAP computation class.
    """
    def compute(self, X, n, default=True, *args):
        if default:
            reducer = pacmap.PaCMAP(n_components=n, random_state=9)
        else:
            reducer = pacmap.PaCMAP(
                n_components=n,
                n_neighbors=args[0],
                MN_ratio=args[1],
                FP_ratio=args[2],
                random_state=9,
            )
        embedding = reducer.fit_transform(X, init="pca")
        embedding = pd.DataFrame(embedding)
        return embedding
    
    def getType(self) -> dict:
        if n != 2 and n != 3:
            return {'type': DimensionalityReduction.PaCMAP, 'dim': -1}
        else:
            return {'type': DimensionalityReduction.PaCMAP, 'dim': n}


# TOOD : this method doesn't seem to be very useful / DimensionalityReduction.type could be used no ?
def dimensionalityReductionChooser(method:int):
    """
    Function that allows to choose the dimensionality reduction method.

    Parameters
    ----------
    method : int
        The method to use.
    """
    if method == DimensionalityReduction.PCA:
        return computationPCA()
    elif method == DimensionalityReduction.TSNE :
        return computationTSNE()
    elif method == DimensionalityReduction.UMAP :
        return computationUMAP()
    elif method == DimensionalityReduction.PaCMAP:
        return computationPaCMAP()


def initialize_dim_red_VS(X, default_projection:int):
    dim_red = dimensionalityReductionChooser(method=default_projection)
    return dim_red.compute(X, 2, True), dim_red.compute(X, 3, True)

def initialize_dim_red_ES(EXP, default_projection):
    dim_red = dimensionalityReductionChooser(method=default_projection)
    return dim_red.compute(EXP, 2, True), dim_red.compute(EXP, 3, True)

def function_score(y, y_chap):
    y = np.array(y)
    y_chap = np.array(y_chap)
    return round(np.sqrt(sum((y - y_chap) ** 2) / len(y)), 3)

def update_figures(gui, exp, projVS, projES):
    with gui.fig1.batch_update():
        gui.fig1.data[0].x, gui.fig1.data[0].y  = gui.dim_red['VS'][projVS][0][0], gui.dim_red['VS'][projVS][0][1]
    with gui.fig2.batch_update():
        gui.fig2.data[0].x, gui.fig2.data[0].y = gui.dim_red['ES'][exp][projES][0][0], gui.dim_red['ES'][exp][projES][0][1]
    with gui.fig1_3D.batch_update():
        gui.fig1_3D.data[0].x, gui.fig1_3D.data[0].y, gui.fig1_3D.data[0].z = gui.dim_red['VS'][projVS][1][0], gui.dim_red['VS'][projVS][1][1], gui.dim_red['VS'][projVS][1][2]
    with gui.fig2_3D.batch_update():
        gui.fig2_3D.data[0].x, gui.fig2_3D.data[0].y, gui.fig2_3D.data[0].z = gui.dim_red['ES'][exp][projES][1][0], gui.dim_red['ES'][exp][projES][1][1], gui.dim_red['ES'][exp][projES][1][2]

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