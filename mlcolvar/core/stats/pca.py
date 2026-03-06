"""Principal component analysis"""

import torch

from mlcolvar.core.stats import Stats

__all__ = ["PCA"]


class PCA(Stats):
    """
    Principal Component Analysis class.

    Attributes
    ----------
    in_features : int
        Number of features
    out_features : int
        Number of output features
    evals : torch.Tensor
        PCA eigenvalues
    evecs : torch.Tensor
        PCA eigenvectors
    """

    def __init__(self, in_features: int, out_features: int = None):
        """
        Initialize a PCA object. If out_features (<in_features) is given a low rank
        approximation will be performed.

        Parameters
        ----------
        in_features : int
            number of input features
        out_features : int, optional
            number of principal components, by default None (equal to in_features)
        """
        super().__init__()

        # Save attributes
        self.in_features = in_features
        self.out_features = in_features if out_features is None else out_features

        # create eigenvector and eigenvalue buffer
        self.register_buffer("evecs", torch.eye(in_features, self.out_features))
        self.register_buffer("evals", torch.zeros(self.out_features))

    def extra_repr(self) -> str:
        repr = f"in_features={self.in_features}, out_features={self.out_features}"
        return repr

    def compute(self, X, save_params=True, **kwargs):
        """
        Compute PCA eigenvalues and eigenvectors via torch.pca_lowrank method.

        Parameters
        ----------
        X : array-like of shape (n_samples, in_features)
            Training data.
        save_params: bool, optional
            Whether to store parameters in model
        kwargs: optional
            Keyword arguments passed to torch.pca_lowrank (e.g. center)

        Returns
        -------
        eigvals : torch.Tensor
            PCA eigenvalues
        eigvecs : torch.Tensor
            PCA eigenvectors

        Notes
        -----
        The eigenvecs object which is returned is a matrix whose column eigvecs[:,i] is the eigenvector associated to eigvals[i]
        """

        n, d = X.shape
        U, S, V = torch.pca_lowrank(X, q=self.out_features, **kwargs)

        evals = torch.square(S) / (n - 1)
        evecs = V

        if save_params:
            self.evals = evals
            self.evecs = evecs

        return evals, evecs

    @property
    def explained_variance(self):
        """Explained variance ratio of the principal components selected.

        Returns
        -------
        exp_var, torch.Tensor
            explained variance ratio
        """
        return self.evals[: self.out_features] / self.evals.sum()

    @property
    def cumulative_explained_variance(self):
        """Cumulative explained variance.

        Returns
        -------
        cum_exp_var, torch.Tensor
            cumulative variance ratio
        """
        return torch.cumsum(self.explained_variance[: self.out_features], 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute linear combination with saved eigenvectors.
        If self.out_features is < then the number of eigenvectors only
        the first out_features components will be used.

        Parameters
        ----------
        x: torch.Tensor
            input

        Returns
        -------
        out : torch.Tensor
            output
        """
        return torch.matmul(x, self.evecs[:, : self.out_features])


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


if __name__ == "__main__":
    test_pca()
