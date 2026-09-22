import torch

from mlcolvar.core.estimators.pca import PCA


def test_pca():
    in_features = 10

    torch.manual_seed(42)
    X = torch.rand(100, in_features) * 100

    # find all components (default)
    pca = PCA(in_features)
    _ = pca.compute(X)
    s = pca(X)

    assert s.shape[1] == in_features
    assert len(pca.explained_variance) == in_features
    assert len(pca.cumulative_explained_variance) == in_features

    # select first n_components after calculation
    n_components = 5
    pca.out_features = n_components

    assert len(pca.explained_variance) == n_components
    assert len(pca.cumulative_explained_variance) == n_components

    # select n_components in init
    pca = PCA(in_features, n_components)
    _ = pca.compute(X)
    s = pca(X)

    assert s.shape[1] == n_components
    assert len(pca.explained_variance) == n_components
    assert len(pca.cumulative_explained_variance) == n_components