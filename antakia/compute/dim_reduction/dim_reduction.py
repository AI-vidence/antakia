import pacmap
import pandas as pd
import umap
from sklearn.decomposition import PCA
from openTSNE import TSNE

from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod


# ===========================================================
#         Projections / Dim Reductions implementations
# ===========================================================


class PCADimReduc(DimReducMethod):
    """
    PCA computation class.
    """
    dimreduc_method = DimReducMethod.dimreduc_method_as_int('PCA')

    def __init__(self, X: pd.DataFrame, dimension: int = 2, callback: callable = None):
        super().__init__(
            self.dimreduc_method,
            PCA,
            dimension,
            X,
            progress_updated=callback,
            default_parameters={
                'n_components': dimension,
            }
        )
        self.allowed_kwargs = ['copy', 'whiten', 'svd_solver', 'tol', 'iterated_power', 'n_oversamples',
                               'power_iteration_normalizer', 'random_state']


class TSNEwrapper(TSNE):
    def fit_transform(self, X):
        return pd.DataFrame(self.fit(X.values), index=X.index)


class TSNEDimReduc(DimReducMethod):
    """
    T-SNE computation class.
    """
    dimreduc_method = DimReducMethod.dimreduc_method_as_int('TSNE')

    def __init__(self, X: pd.DataFrame, dimension: int = 2, callback: callable = None):
        super().__init__(
            self.dimreduc_method,
            TSNEwrapper,
            dimension,
            X,
            progress_updated=callback,
            default_parameters={
                'n_components': dimension,
                'n_jobs': -1
            }
        )
        self.allowed_kwargs = ['perplexity', 'early_exaggeration', 'learning_rate', 'n_iter',
                               'n_iter_without_progress', 'min_grad_norm', 'metric', 'metric_params', 'init', 'verbose',
                               'random_state', 'method', 'angle', 'n_jobs'
                               ]

    @classmethod
    def parameters(cls):
        return {
            'perplexity': {
                'type': float,
                'min': 5,
                'max': 50,
                'default': 12
            },
            'learning_rate': {
                'type': [float, str],
                'min': 10,
                'max': 1000,
                'default': 'auto'
            }
        }


class UMAPDimReduc(DimReducMethod):
    """
    UMAP computation class.
    """
    dimreduc_method = DimReducMethod.dimreduc_method_as_int('UMAP')

    def __init__(self, X: pd.DataFrame, dimension: int = 2, callback: callable = None):
        super().__init__(
            self.dimreduc_method,
            umap.UMAP,
            dimension,
            X,
            progress_updated=callback,
            default_parameters={
                'n_components': dimension,
                'n_jobs': -1
            }
        )
        self.allowed_kwargs = ['n_neighbors', 'metric', 'metric_kwds', 'output_metric',
                               'output_metric_kwds', 'n_epochs', 'learning_rate', 'init', 'min_dist', 'spread',
                               'low_memory', 'n_jobs', 'set_op_mix_ratio', 'local_connectivity', 'repulsion_strength',
                               'negative_sample_rate', 'transform_queue_size', 'a', 'b', 'random_state',
                               'angular_rp_forest', 'target_n_neighbors', 'target_metric', 'target_metric_kwds',
                               'target_weight', 'transform_seed', 'transform_mode', 'force_approximation_algorithm',
                               'verbose', 'tqdm_kwds', 'unique', 'densmap', 'dens_lambda', 'dens_frac',
                               'dens_var_shift', 'output_dens', 'disconnection_distance', 'precomputed_knn',
                               ]

    @classmethod
    def parameters(cls):
        return {
            'n_neighbors': {
                'type': int,
                'min': 1,
                'max': 200,
                'default': 15
            },
            'min_dist': {
                'type': float,
                'min': 0.1,
                'max': 0.99,
                'default': 0.1
            }
        }


class PaCMAPDimReduc(DimReducMethod):
    """
    PaCMAP computation class.

    """
    dimreduc_method = DimReducMethod.dimreduc_method_as_int('PaCMAP')

    def __init__(self, X: pd.DataFrame, dimension: int = 2, callback: callable = None):
        super().__init__(
            self.dimreduc_method,
            pacmap.PaCMAP,
            dimension,
            X,
            progress_updated=callback,
            default_parameters={
                'n_components': dimension,
            }
        )
        self.allowed_kwargs = ['n_neighbors', 'MN_ratio', 'FP_ratio', 'pair_neighbors', 'pair_MN',
                               'pair_FP', 'distance', 'lr', 'num_iters', 'apply_pca', 'intermediate',
                               'intermediate_snapshots', 'random_state']

    @classmethod
    def parameters(cls):
        return {
            'n_neighbors': {
                'type': int,
                'min': 1,
                'max': 200,
                'default': 15
            },
            'MN_ratio': {
                'type': float,
                'min': 0.1,
                'max': 10,
                'default': 0.5,
                'scale': 'log'
            },
            'FP_ratio': {
                'type': float,
                'min': 0.1,
                'max': 10,
                'default': 2,
                'scale': 'log'
            }
        }


dim_reduc_factory = {
    dm.dimreduc_method: dm for dm in [PCADimReduc, TSNEDimReduc, UMAPDimReduc, PaCMAPDimReduc]
}


def compute_projection(X: pd.DataFrame, y: pd.Series, dimreduc_method: int, dimension: int, callback: callable = None,
                       **kwargs) -> pd.DataFrame:
    if not DimReducMethod.is_valid_dimreduc_method(dimreduc_method) or not DimReducMethod.is_valid_dim_number(
            dimension):
        raise ValueError("Cannot compute proj method #", dimreduc_method, " in ", dimension, " dimensions")

    X_scaled = DimReducMethod.scale_value_space(X, y)

    dim_reduc = dim_reduc_factory.get(dimreduc_method)
    default_kwargs = {'random_state': 9}
    default_kwargs.update(kwargs)
    dim_reduc_kwargs = {k: v for k, v in default_kwargs.items() if k in dim_reduc.allowed_kwargs}
    proj_values = dim_reduc(X_scaled, dimension, callback).compute(**dim_reduc_kwargs)
    proj_values.index = X.index
    return proj_values
