import pandas as pd

# Imports for the dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import pacmap

class DimensionalityReduction:
    """
    Class that allows to reduce the dimensionality of the data.
    """
    def __init__(self, method: str = "PaCMAP", default_parameters: bool = True, *params):
        """
        Constructor of the class DimensionalityReduction.

        Parameters
        ---------
        method : str
            The method used to reduce the dimensionality of the data.
        n_dimensions : int
            The number of dimensions of the data after the reduction.
        default_parameters : bool
            Whether to use the default parameters of the method or not.
        params : list
            The parameters of the method.
        """
        self.method = method
        self.default_parameters = default_parameters
        self.params = params

    def red_PCA(self, X, n, default):
        # definition of the method PCA, used for the EE and the EV
        if default:
            pca = PCA(n_components=n)
        pca.fit(X)
        X_pca = pca.transform(X)
        X_pca = pd.DataFrame(X_pca)
        return X_pca

    def red_TSNE(self, X, n, default):
        # definition of the method TSNE, used for the EE and the EV
        if default:
            tsne = TSNE(n_components=n)
        X_tsne = tsne.fit_transform(X)
        X_tsne = pd.DataFrame(X_tsne)
        return X_tsne

    def red_UMAP(self, X, n, default):
        # definition of the method UMAP, used for the EE and the EV
        if default:
            reducer = umap.UMAP(n_components=n)
        embedding = reducer.fit_transform(X)
        embedding = pd.DataFrame(embedding)
        return embedding

    def red_PACMAP(self, X, n, default, *args):
        # definition of the method PaCMAP, used for the EE and the EV
        # if default : no change of parameters (only for PaCMAP for now)
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

    def compute(self, X: pd.DataFrame, n_dimensions: int):
        """
        Function that computes the dimensionality reduction.

        Parameters
        ---------
        X : pandas dataframe
            The dataframe containing the data to reduce.

        Returns
        -------
        X_reduced : pandas dataframe
            The dataframe containing the data reduced.
        """
        if self.method == "PCA":
            return self.red_PCA(X, n_dimensions, self.default_parameters)
        elif self.method == "t-SNE":
            return self.red_TSNE(X, n_dimensions, self.default_parameters)
        elif self.method == "UMAP":
            return self.red_UMAP(X, n_dimensions, self.default_parameters)
        elif self.method == "PaCMAP":
            return self.red_PACMAP(X, n_dimensions, self.default_parameters, *self.params)
        else:
            raise ValueError("The method is not valid.")

def initialize_dim_red_EV(X, default_projection):
    if default_projection not in ["PCA", "t-SNE", "UMAP", "PaCMAP"]:
        default_projection = "PaCMAP"
    indice = ["PCA", "t-SNE", "UMAP", "PaCMAP"].index(default_projection)
    dim_red = DimensionalityReduction(method=default_projection, default_parameters=True)
    Espace_valeurs = ["None", "None", "None", "None"]
    Espace_valeurs_3D = ["None", "None", "None", "None"]
    Espace_valeurs[indice] = dim_red.compute(X, 2)
    Espace_valeurs_3D[indice] = dim_red.compute(X, 3)
    return Espace_valeurs, Espace_valeurs_3D

def initialize_dim_red_EE(EXP, default_projection):
    if default_projection not in ["PCA", "t-SNE", "UMAP", "PaCMAP"]:
        default_projection = "PaCMAP"
    indice = ["PCA", "t-SNE", "UMAP", "PaCMAP"].index(default_projection)
    dim_red = DimensionalityReduction(method=default_projection, default_parameters=True)
    Espace_explications = ["None", "None", "None", "None"]
    Espace_explications_3D = ["None", "None", "None", "None"]
    Espace_explications[indice] = dim_red.compute(EXP, 2)
    Espace_explications_3D[indice] = dim_red.compute(EXP, 3)
    return Espace_explications, Espace_explications_3D

def fonction_score(y, y_chap):
    # function that calculates the score of a machine-learning model
    y = np.array(y)
    y_chap = np.array(y_chap)
    return round(np.sqrt(sum((y - y_chap) ** 2) / len(y)), 3)