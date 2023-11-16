import pacmap
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from antakia.data import DimReducMethod


# ===========================================================
#         Projections / Dim Reductions implementations
# ===========================================================


class PCADimReduc(DimReducMethod):
    """
    PCA computation class.
    """

    def __init__(self, X: pd.DataFrame, dimension: int = 2, callback: callable = None):
        DimReducMethod.__init__(self, DimReducMethod.PCA, dimension, X, callback)
        self.allowed_kwargs = ['n_components', 'copy', 'whiten', 'svd_solver', 'tol', 'iterated_power', 'n_oversamples',
                               'power_iteration_normalizer', 'random_state']

    def compute(self, **kwargs) -> pd.DataFrame:
        self.publish_progress(0)

        kwargs['n_components'] = self.get_dimension()
        pca = PCA(**kwargs)
        pca.fit(self.X)
        X_pca = pca.transform(self.X)
        X_pca = pd.DataFrame(X_pca)

        self.publish_progress(100)
        return X_pca


class TSNEDimReduc(DimReducMethod):
    """
    T-SNE computation class.
    """

    def __init__(self, X: pd.DataFrame, dimension: int = 2, callback: callable = None):
        DimReducMethod.__init__(self, DimReducMethod.TSNE, dimension, X, callback)
        self.allowed_kwargs = ['n_components', 'perplexity', 'early_exaggeration', 'learning_rate', 'n_iter',
                               'n_iter_without_progress', 'min_grad_norm', 'metric', 'metric_params', 'init', 'verbose',
                               'random_state', 'method', 'angle', 'n_jobs'
                               ]

    def compute(self, **kwargs) -> pd.DataFrame:
        self.publish_progress(0)
        kwargs['n_components'] = self.get_dimension()
        tsne = TSNE(kwargs)
        X_tsne = tsne.fit_transform(self.X)
        X_tsne = pd.DataFrame(X_tsne)

        self.publish_progress(100)
        return X_tsne


class UMAPDimReduc(DimReducMethod):
    """
    UMAP computation class.
    """

    def __init__(self, X: pd.DataFrame, dimension: int = 2, callback: callable = None):
        DimReducMethod.__init__(self, DimReducMethod.UMAP, dimension, X, callback)
        self.allowed_kwargs = ['n_neighbors', 'n_components', 'metric', 'metric_kwds', 'output_metric',
                               'output_metric_kwds', 'n_epochs', 'learning_rate', 'init', 'min_dist', 'spread',
                               'low_memory', 'n_jobs', 'set_op_mix_ratio', 'local_connectivity', 'repulsion_strength',
                               'negative_sample_rate', 'transform_queue_size', 'a', 'b', 'random_state',
                               'angular_rp_forest', 'target_n_neighbors', 'target_metric', 'target_metric_kwds',
                               'target_weight', 'transform_seed', 'transform_mode', 'force_approximation_algorithm',
                               'verbose', 'tqdm_kwds', 'unique', 'densmap', 'dens_lambda', 'dens_frac',
                               'dens_var_shift', 'output_dens', 'disconnection_distance', 'precomputed_knn',
                               ]

    def compute(self, **kwargs) -> pd.DataFrame:
        self.publish_progress(0)

        kwargs['n_components'] = self.get_dimension()
        reducer = umap.UMAP(**kwargs)
        embedding = reducer.fit_transform(self.X)
        embedding = pd.DataFrame(embedding)

        self.publish_progress(100)
        return embedding


class PaCMAPDimReduc(DimReducMethod):
    """
    PaCMAP computation class.

    """

    def __init__(self, X: pd.DataFrame, dimension: int = 2, callback: callable = None):
        DimReducMethod.__init__(self, DimReducMethod.PaCMAP, dimension, X, callback)
        self.allowed_kwargs = ['n_components', 'n_neighbors', 'MN_ratio', 'FP_ratio', 'pair_neighbors', 'pair_MN',
                               'pair_FP', 'distance', 'lr', 'num_iters', 'apply_pca', 'intermediate',
                               'intermediate_snapshots', 'random_state']

    def compute(self, **kwargs) -> pd.DataFrame:
        self.publish_progress(0)
        kwargs['n_components'] = self.get_dimension()
        reducer = pacmap.PaCMAP(**kwargs)
        embedding = reducer.fit_transform(self.X, init="pca")
        embedding = pd.DataFrame(embedding)

        self.publish_progress(100)
        return embedding


dim_reduc_factory = {
    DimReducMethod.PCA: PCADimReduc,
    DimReducMethod.TSNE: TSNEDimReduc,
    DimReducMethod.UMAP: UMAPDimReduc,
    DimReducMethod.PaCMAP: PaCMAPDimReduc,
}


def compute_projection(X: pd.DataFrame, dimreduc_method: int, dimension: int, callback: callable = None,
                       **kwargs) -> pd.DataFrame:
    if not DimReducMethod.is_valid_dimreduc_method(dimreduc_method) or not DimReducMethod.is_valid_dim_number(
            dimension):
        raise ValueError("Cannot compute proj method #", dimreduc_method, " in ", dimension, " dimensions")

    dim_reduc = dim_reduc_factory.get(dimreduc_method)
    default_kwargs = {'random_state': 9}
    default_kwargs.update(kwargs)
    dim_reduc_kwargs = {k: v for k, v in default_kwargs.items() if k in dim_reduc.allowed_kwargs}
    proj_values = dim_reduc(X, dimension, callback).compute(**dim_reduc_kwargs)
    return proj_values
