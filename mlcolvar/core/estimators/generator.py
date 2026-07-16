"""Time-lagged independent component analysis"""

__all__ = ["Generator"]

import torch
import torch_geometric
from mlcolvar.core.estimators import Estimator

from typing import Union, Tuple
from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset, DictLoader
from mlcolvar.core.estimators.utils_generator import compute_eigenfunctions



class Generator(Estimator):
    """
    Estimator for eigenfunctions of the infinitesimal generator.
    """

    def __init__(self, in_features, out_features=None, feature_method=None):
        """
        Initialize a generator object.
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

        If eigenvectors have already been computed and ``recompute=False``, this
        method reuses the cached eigenvectors and eigenvalues and only evaluates the
        current model on ``dataset``.

        Parameters
        ----------
        dataset : DictDataset
            Dataset containing at least ``"data"`` and ``"weights"``. If the model
            uses runtime-cell preprocessing, the dataset must also contain ``"cell"``.
        eta : float, optional
            Resolvent shift used for this computation. Defaults to the value used at
            initialization.
        friction : torch.Tensor, optional
            Friction prefactor used for this computation. Defaults to the value used
            at initialization.
        tikhonov_reg : float, default=1e-4
            Tikhonov regularization parameter used when solving the linear problem.
        descriptors_derivatives : SmartDerivatives or torch.Tensor, optional
            Descriptor derivatives used to compute gradients efficiently.
        batch_size : int, default=100
            Batch size used during eigenfunction computation.

        Returns
        -------
        eigenfunctions : torch.Tensor
            Eigenfunctions evaluated on the dataset, with shape ``(n_samples, r)``.
        evals : torch.Tensor
            Generator eigenvalues, with shape ``(r,)``.
        evecs : torch.Tensor
            Eigenvectors mapping the learned representation to eigenfunctions, with
            shape ``(r, r)``.
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
        """
        Compute linear combination with saved eigenvectors

        Parameters
        ----------
        x: torch.Tensor
            input

        Returns
        -------
        out : torch.Tensor
            output
        """

        return torch.matmul(x, self.evecs)
def test_generator():
    from mlcolvar.data import DictDataset, DictModule

    in_features = 2
    X = torch.rand(100, in_features) * 100

    w = torch.rand(len(X))

    # Compute generator
    generator = Generator(in_features, out_features=2)
    dataset = DictDataset({"data": X, "weights": w})
    datamodule = DictModule(dataset, lengths=[0.8,0.2])
    datamodule.setup()

    generator.compute(datamodule.train_dataloader(), eta=0.1, friction=torch.Tensor([1.0, 1.0]), tikhonov_reg=1e-4,n_dim=1, softmax_postproc=False)
    s = generator(X)
    print(X.shape, "-->", s.shape)
    print("eigvals", generator.evals)

