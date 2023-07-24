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
            + "s (temps estimé : "
            + str(minute)
            + "min "
            + str(round(seconde))
            + "s)"
        )

class computationSHAP(LongTask):
    """
    SHAP computation class.
    """
    def compute(self):
        self.progress = 0
        self.done_widget.v_model = "primary"
        self.text_widget.v_model = None
        time_init = time.time()
        explainer = shap.Explainer(self.model.predict, self.X_all)
        shap_values = pd.DataFrame().reindex_like(self.X)
        j = list(self.X.columns)
        for i in range(len(j)):
            j[i] = j[i] + "_shap"
        for i in range(len(self.X)):
            shap_value = explainer(self.X[i : i + 1], max_evals=1400)
            shap_values.iloc[i] = shap_value.values
            self.progress += 100 / len(self.X)
            self.progress_widget.v_model = self.progress
            self.text_widget.v_model = self.generation_texte(i, len(self.X), time_init, self.progress_widget.v_model)
        shap_values.columns = j
        self.value = shap_values
        self.done_widget.v_model = "success"
        return shap_values

class computationLIME(LongTask):
    """
    LIME computation class.
    """
    
    def compute(self):
        self.done_widget.v_model = "primary"
        self.progress_widget.v_model = 0
        self.text_widget.v_model = None
        time_init = time.time()
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(self.X_all), feature_names=self.X.columns, class_names=['price'], verbose=False, mode='regression')
        N = len(self.X)
        LIME = pd.DataFrame(np.zeros((N, self.X.shape[-1])))
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
            LIME.iloc[j] = l
            self.progress_widget.v_model  += 100 / len(self.X)
            self.text_widget.v_model = self.generation_texte(j, len(self.X), time_init, self.progress_widget.v_model)
        j = list(self.X.columns)
        for i in range(len(j)):
            j[i] = j[i] + "_shap"
        LIME.columns = j
        self.value = LIME
        self.done_widget.v_model = "success"
        return LIME

class DimensionalityReduction(ABC):
    """
    Class that allows to reduce the dimensionality of the data.
    """
    def __init__(self):
        """
        Constructor of the class DimensionalityReduction.
        """
        pass

    @abstractmethod
    def compute(self):
        pass
        
class computationPCA(DimensionalityReduction):
    """
    PCA computation class.
    """
    def compute(self, X, n, default=True):
        # definition of the method PCA, used for the EE and the EV
        if default:
            pca = PCA(n_components=n)
        pca.fit(X)
        X_pca = pca.transform(X)
        X_pca = pd.DataFrame(X_pca)
        return X_pca
    
class computationTSNE(DimensionalityReduction):
    """
    t-SNE computation class.
    """
    def compute(self, X, n, default=True):
        # definition of the method TSNE, used for the EE and the EV
        if default:
            tsne = TSNE(n_components=n)
        X_tsne = tsne.fit_transform(X)
        X_tsne = pd.DataFrame(X_tsne)
        return X_tsne
    
class computationUMAP(DimensionalityReduction):
    """
    UMAP computation class.
    """
    def compute(self, X, n, default=True):
        if default:
            reducer = umap.UMAP(n_components=n)
        embedding = reducer.fit_transform(X)
        embedding = pd.DataFrame(embedding)
        return embedding
    
class computationPaCMAP(DimensionalityReduction):
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
    
def DimensionalityReductionChooser(method):
    """
    Function that allows to choose the dimensionality reduction method.

    Parameters
    ----------
    method : str
        The name of the method to use.
    """
    if method == 'PCA':
        return computationPCA()
    elif method == 't-SNE':
        return computationTSNE()
    elif method == 'UMAP':
        return computationUMAP()
    elif method == 'PaCMAP':
        return computationPaCMAP()


def initialize_dim_red_EV(X, default_projection):
    dim_red = DimensionalityReductionChooser(method=default_projection)
    return dim_red.compute(X, 2, True), dim_red.compute(X, 3, True)

def initialize_dim_red_EE(EXP, default_projection):
    dim_red = DimensionalityReductionChooser(method=default_projection)
    return dim_red.compute(EXP, 2, True), dim_red.compute(EXP, 3, True)

def fonction_score(y, y_chap):
    y = np.array(y)
    y_chap = np.array(y_chap)
    return round(np.sqrt(sum((y - y_chap) ** 2) / len(y)), 3)

def update_figures(gui, exp, projEV, projEE):
    with gui.fig1.batch_update():
        gui.fig1.data[0].x, gui.fig1.data[0].y  = gui.dim_red['EV'][projEV][0][0], gui.dim_red['EV'][projEV][0][1]
    with gui.fig2.batch_update():
        gui.fig2.data[0].x, gui.fig2.data[0].y = gui.dim_red['EE'][exp][projEE][0][0], gui.dim_red['EE'][exp][projEE][0][1]
    with gui.fig1_3D.batch_update():
        gui.fig1_3D.data[0].x, gui.fig1_3D.data[0].y, gui.fig1_3D.data[0].z = gui.dim_red['EV'][projEV][1][0], gui.dim_red['EV'][projEV][1][1], gui.dim_red['EV'][projEV][1][2]
    with gui.fig2_3D.batch_update():
        gui.fig2_3D.data[0].x, gui.fig2_3D.data[0].y, gui.fig2_3D.data[0].z = gui.dim_red['EE'][exp][projEE][1][0], gui.dim_red['EE'][exp][projEE][1][1], gui.dim_red['EE'][exp][projEE][1][2]

def fonction_beeswarm_shap(gui, exp, nom_colonne):
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