"""Estimators for the infinitesimal generator"""

__all__ = ["Generator"]

import torch
import torch_geometric
from mlcolvar.core.estimators import Estimator

from typing import Union, Tuple
from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictLoader
from mlcolvar.core.estimators.utils_generator import compute_eigenfunctions



class Generator(Estimator):
    """Estimator for eigenfunctions of the infinitesimal generator."""

    def __init__(self, in_features, out_features=None, feature_method=None):
        """Initialize the generator estimator.

        Parameters
        ----------
        in_features : int
            Number of input features in the learned representation.
        out_features : int, optional
            Number of representation components used for eigenfunction computation. Defaults to ``in_features``.
        feature_method : callable, optional
            Function mapping model inputs to the learned representation. If None, the identity mapping is used.
        """
        super().__init__()

        # save attributes
        self.in_features = in_features
        self.out_features = out_features if out_features is not None else in_features

        # buffers
        # generator eigenvectors
        self.register_buffer("evecs", torch.eye(in_features, self.out_features))
        # mean to obtain mean free inputs
        # init other attributes
        self.evals = None

        # Regularization
        self.evecs = None
        if feature_method is None:
            self.feature_method = self._forward_dummy # if there is no feature, we just apply identity
        else:
            self.feature_method = feature_method
    
    def _forward_dummy(self,x):
        return x
    
    def extra_repr(self) -> str:
        repr = f"in_features={self.in_features}, out_features={self.out_features}"
        return repr

    def compute(self,
                dataloader : Union[DictLoader, torch_geometric.loader.DataLoader],        
                eta : float = None, 
                friction : float = None,         
                tikhonov_reg : float = 1e-4,    
                is_graph=False,
                descriptors_derivatives : Union[SmartDerivatives, torch.Tensor] = None,
                n_dim=3,
                softmax_postproc=True,
                cell=None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute generator eigenfunctions from the learned representation.

        The method estimates weighted covariance and gradient covariance matrices and solves the corresponding generalized eigenvalue problem. The computed eigenvalues and eigenvectors are stored in ``self.evals`` and ``self.evecs``, respectively.

        Parameters
        ----------
        dataloader : DictLoader or torch_geometric.loader.DataLoader
            Dataloader providing input configurations and statistical weights. For non-graph models, batches must contain ``"data"`` and ``"weights"``.
        eta : float, optional
            Resolvent shift parameter. If None, the method attempts to use ``self.eta``.
        friction : torch.Tensor, optional
            Per-atom diffusion prefactors used to weight coordinate gradients. If None, the method attempts to use ``self.friction``.
        tikhonov_reg : float, default=1e-4
            Tikhonov regularization strength added to the shifted covariance matrix before solving the eigenvalue problem.
        is_graph : bool, default=False
            Whether the input data consists of molecular graphs.
        descriptors_derivatives : SmartDerivatives or torch.Tensor, optional
            Derivatives of descriptors with respect to atomic positions.
        n_dim : int, default=3
            Number of spatial dimensions.
        softmax_postproc : bool, default=True
            Whether to apply softmax to the learned representation and append a constant basis function.
        cell : torch.Tensor, optional
            Simulation cell passed to the feature method when provided.

        Returns
        -------
        eigenfunctions : torch.Tensor
            Eigenfunctions evaluated on the dataset, with shape ``(n_samples, n_modes)``.
        evals : torch.Tensor
            Generator eigenvalues, with shape ``(n_modes,)``.
        evecs : torch.Tensor
            Eigenvectors mapping the learned representation to eigenfunctions, with shape ``(n_basis, n_modes)``.
        output : torch.Tensor
            Representation used for eigenfunction computation, with shape ``(n_samples, n_basis)``.

        Notes
        -----
        When ``softmax_postproc=True``, a constant basis function is appended, giving ``n_basis = out_features + 1``. Otherwise, ``n_basis = out_features``.

        The number of returned modes may be smaller than the number of basis functions due to numerical filtering.

        The current implementation requires ``eta`` and ``friction`` to be provided unless the corresponding estimator attributes exist.
        """
        if friction is None:
            friction = self.friction
        if eta is None:
            eta = self.eta
        
        

        eigenfunctions, evals, evecs, output = compute_eigenfunctions(
            dataloader=dataloader,
            feature_method=self.feature_method,
            r=self.out_features,
            eta=eta,
            friction=friction,
            tikhonov_reg=tikhonov_reg,
            descriptors_derivatives=descriptors_derivatives,
            n_dim=n_dim,
            softmax_postproc=softmax_postproc,
            is_graph=is_graph,
            cell=cell,
            )
        self.evals = evals
        self.evecs = evecs
        return eigenfunctions, evals, evecs, output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features onto the learned eigenfunctions.

        Parameters
        ----------
        x : torch.Tensor
            Input representation with shape ``(n_samples, n_basis)``.

        Returns
        -------
        torch.Tensor
            Projected eigenfunctions with shape ``(n_samples, n_modes)``.

        Notes
        -----
        The eigenvectors must first be computed using ``compute()``.
        """

        return torch.matmul(x, self.evecs)
