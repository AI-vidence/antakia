import pandas as pd

# Imports for the dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import pacmap


def red_PCA(X, n, default):
    # definition of the method PCA, used for the EE and the EV
    if default:
        pca = PCA(n_components=n)
    pca.fit(X)
    X_pca = pca.transform(X)
    X_pca = pd.DataFrame(X_pca)
    return X_pca

def red_TSNE(X, n, default):
    # definition of the method TSNE, used for the EE and the EV
    if default:
        tsne = TSNE(n_components=n)
    X_tsne = tsne.fit_transform(X)
    X_tsne = pd.DataFrame(X_tsne)
    return X_tsne

def red_UMAP(X, n, default):
    # definition of the method UMAP, used for the EE and the EV
    if default:
        reducer = umap.UMAP(n_components=n)
    embedding = reducer.fit_transform(X)
    embedding = pd.DataFrame(embedding)
    return embedding

def red_PACMAP(X, n, default, *args):
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