import numpy as np
import pacmap
import pandas as pd
import pytest
import umap

from sklearn.decomposition import PCA
from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.compute.dim_reduction.dim_reduction import compute_projection, dim_reduc_factory, PCADimReduc, TSNEwrapper, \
    TSNEDimReduc, UMAPDimReduc, PaCMAPDimReduc
from tests.utils_fct import generate_df_series_callable


def test_init_PCA():
    X, function = generate_df_series_callable()[0:3:2]

    dr_pca = PCADimReduc(X, 2, function)
    assert dr_pca.dimreduc_method == 1
    np.testing.assert_array_equal(dr_pca.default_parameters, {'n_components': 2})
    assert dr_pca.dimension == 2
    assert dr_pca.dimreduc_model == PCA
    assert dr_pca.X.equals(X)
    assert dr_pca.progress_updated == function
    np.testing.assert_array_equal(dr_pca.allowed_kwargs,
                                  ['copy', 'whiten', 'svd_solver', 'tol', 'iterated_power', 'n_oversamples',
                                   'power_iteration_normalizer', 'random_state'])


def test_fit_TSNEwrapper():
    tsn = TSNEwrapper()
    X = generate_df_series_callable()[0]
    tsn.fit_transform(X)


def test_init_TSNEDimReduc():
    X, function = generate_df_series_callable()[0:3:2]
    tsne = TSNEDimReduc(X, 2, function)

    assert tsne.dimreduc_method == 2
    np.testing.assert_array_equal(tsne.default_parameters, {'n_components': 2, 'n_jobs': -1})
    assert tsne.dimension == 2
    assert tsne.dimreduc_model == TSNEwrapper
    assert tsne.X.equals(X)
    assert tsne.progress_updated == function
    np.testing.assert_array_equal(tsne.allowed_kwargs,
                                  ['perplexity', 'early_exaggeration', 'learning_rate', 'n_iter',
                                   'n_iter_without_progress', 'min_grad_norm', 'metric', 'metric_params', 'init',
                                   'verbose',
                                   'random_state', 'method', 'angle', 'n_jobs'
                                   ]
                                  )


def test_parameters_TSNEDimReduc():
    X, function = generate_df_series_callable()[0:3:2]
    tsne = TSNEDimReduc(X, 2, function)
    np.testing.assert_array_equal(tsne.parameters(), {'perplexity': {'type': float, 'min': 5, 'max': 50, 'default': 12},
                                                      'learning_rate': {'type': [float, str], 'min': 10, 'max': 1000,
                                                                        'default': 'auto'}})


def test_init_UMAPDimReduc():
    X, function = generate_df_series_callable()[0:3:2]
    umap_dr = UMAPDimReduc(X, 2, function)

    assert umap_dr.dimreduc_method == 3
    np.testing.assert_array_equal(umap_dr.default_parameters, {'n_components': 2, 'n_jobs': -1})
    assert umap_dr.dimension == 2
    assert umap_dr.dimreduc_model == umap.UMAP
    assert umap_dr.X.equals(X)
    assert umap_dr.progress_updated == function
    np.testing.assert_array_equal(umap_dr.allowed_kwargs,
                                  ['n_neighbors', 'metric', 'metric_kwds', 'output_metric',
                                   'output_metric_kwds', 'n_epochs', 'learning_rate', 'init', 'min_dist', 'spread',
                                   'low_memory', 'n_jobs', 'set_op_mix_ratio', 'local_connectivity',
                                   'repulsion_strength',
                                   'negative_sample_rate', 'transform_queue_size', 'a', 'b', 'random_state',
                                   'angular_rp_forest', 'target_n_neighbors', 'target_metric', 'target_metric_kwds',
                                   'target_weight', 'transform_seed', 'transform_mode', 'force_approximation_algorithm',
                                   'verbose', 'tqdm_kwds', 'unique', 'densmap', 'dens_lambda', 'dens_frac',
                                   'dens_var_shift', 'output_dens', 'disconnection_distance', 'precomputed_knn',
                                   ]
                                  )


def test_parameters_UMAPDimReduc():
    X, function = generate_df_series_callable()[0:3:2]
    umap_dr = UMAPDimReduc(X, 2, function)
    np.testing.assert_array_equal(umap_dr.parameters(),
                                  {'n_neighbors': {'type': int, 'min': 1, 'max': 200, 'default': 15},
                                   'min_dist': {'type': float, 'min': 0.1, 'max': 0.99, 'default': 0.1}})


def test_init_PacMAPDimReduc():
    X, function = generate_df_series_callable()[0:3:2]
    pacmap_dr = PaCMAPDimReduc(X, 2, function)

    assert pacmap_dr.dimreduc_method == 4
    np.testing.assert_array_equal(pacmap_dr.default_parameters, {'n_components': 2})
    assert pacmap_dr.dimension == 2
    assert pacmap_dr.dimreduc_model == pacmap.PaCMAP
    assert pacmap_dr.X.equals(X)
    assert pacmap_dr.progress_updated == function
    np.testing.assert_array_equal(pacmap_dr.allowed_kwargs,
                                  ['n_neighbors', 'MN_ratio', 'FP_ratio', 'pair_neighbors', 'pair_MN',
                                   'pair_FP', 'distance', 'lr', 'num_iters', 'apply_pca', 'intermediate',
                                   'intermediate_snapshots', 'random_state']
                                  )


def test_parameters_PacMAPDimReduc():
    X, function = generate_df_series_callable()[0:3:2]
    pacmap_dr = PaCMAPDimReduc(X, 2, function)
    np.testing.assert_array_equal(pacmap_dr.parameters(),
                                  {'n_neighbors': {'type': int, 'min': 1, 'max': 200, 'default': 15},
                                   'MN_ratio': {'type': float, 'min': 0.1, 'max': 10, 'default': 0.5, 'scale': 'log'},
                                   'FP_ratio': {'type': float, 'min': 0.1, 'max': 10, 'default': 2, 'scale': 'log'}})


def test_compute_projection():  # not ok
    function = generate_df_series_callable()[2]
    X = pd.DataFrame(np.random.random((30, 5)), index=np.random.choice(np.random.randint(100, size=40), size=30),
                     columns=[f'c{i}' for i in range(5)])
    y = X.sum(axis=1)

    with pytest.raises(ValueError):
        compute_projection(X, y, 8, 2, function)

    np.testing.assert_array_equal(compute_projection(X, y, 1, 2, function).index, X.index)


def test_dim_reduction():  # ok sauf PaCMAP : windows fatal error (access violation File) pour PaCMAP
    X = pd.DataFrame(np.random.random((10, 5)), index=np.random.choice(np.random.randint(100, size=20), size=10),
                     columns=[f'c{i}' for i in range(5)])
    y = X.sum(axis=1)

    # for dim_method in DimReducMethod.dimreduc_methods_as_list():
    for dim_method in [1, 2, 3]:
        params = dim_reduc_factory.get(dim_method).parameters()
        params = {k: v['default'] for k, v in params.items()}
        cpt_proj_2D = compute_projection(X, y, dim_method, 2, lambda x, y, z: None, **params)
        assert cpt_proj_2D.shape == (len(X), 2)
        assert X.index.equals(cpt_proj_2D.index)
        cpt_proj_3D = compute_projection(X, y, dim_method, 3, lambda x, y, z: None, **params)
        assert cpt_proj_3D.shape == (len(X), 3)
        assert X.index.equals(cpt_proj_3D.index)
