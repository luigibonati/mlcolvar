import torch
import lightning
from typing import Union, Tuple,List
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, BaseGNN
from mlcolvar.core.loss.generator_loss import GeneratorLoss
from mlcolvar.cvs.generator.utils import SoftmaxPostProcessing
from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset, DictModule
from mlcolvar.core.estimators import Generator
__all__ = ["DeepGenerator"]

class DeepGenerator(BaseCV):
    """Learn eigenfunctions of the infinitesimal generator.

    The model learns a neural-network representation of the eigenspace of the infinitesimal generator from weighted configurations and their spatial derivatives, without requiring time-lagged data.
    **Data**: Training requires a DictDataset containing ``"data"`` and ``"weights"``.
    **Loss**: Minimize the variational representation loss together with an orthonormality regularization term.

    References
    ----------
    .. [*] T. Devergne, V. Kostic, M. Pontil, M. Parrinello, "Slow dynamical modes from static averages", J. Chem. Phys., 2025, DOI: 10.1063/5.0246248

    See Also
    --------
    mlcolvar.core.loss.generator_loss.GeneratorLoss
        Loss function for learning generator eigenfunction representations.
    mlcolvar.core.estimators.utils_generator.compute_eigenfunctions
        Compute generator eigenfunctions and eigenvalues from a representation.
    mlcolvar.core.estimators.utils_generator.forecast_observable_evolution
        Forecast the time evolution of observables from generator eigenfunctions.
    """

    DEFAULT_BLOCKS = ["nn"]
    MODEL_BLOCKS = ["nn"]

    def __init__(self,
                 r: int,
                 model: Union[List[int], FeedForward, BaseGNN],
                 eta: float,
                 alpha: float,
                 friction: torch.Tensor,
                 descriptors_derivatives: Union[SmartDerivatives, torch.Tensor] = None,
                 n_dim: int = 3,
                 split:bool = True,
                 softmax_postproc: bool = True,
                 options: dict = None,
                 **kwargs
                 ):
        """Initialize a neural-network representation of generator eigenfunctions.

        Parameters
        ----------
        r : int
            Number of output components in the learned representation.
        model : list[int] or FeedForward or BaseGNN
            Neural-network architecture. If a list is provided, it defines the layer sizes of a :class:`FeedForward` model. Otherwise, the supplied model instance is used directly.
        eta : float
            Resolvent shift parameter used in ``(I - L/eta)^-1``.
        alpha : float
            Weight of the orthonormality regularization term in the loss.
        friction : torch.Tensor
            Per-atom diffusion prefactors used to weight coordinate gradients.
        descriptors_derivatives : SmartDerivatives or torch.Tensor, optional
            Derivatives of descriptors with respect to atomic positions. Providing these derivatives avoids recomputing descriptor Jacobians.
        n_dim : int, default=3
            Number of spatial dimensions.
        split : bool, default=True
            Whether to use the split-batch estimator for the loss. Currently, only ``split=True`` is supported.
        softmax_postproc : bool, default=True
            Whether to apply softmax-based post-processing to the learned representation.
        options : dict, optional
            Options for the neural network and optimizer configuration.
        **kwargs
            Additional keyword arguments passed to :class:`BaseCV`.
        """
        super().__init__(model, **kwargs)

        self.r = r
        self.eta = eta
        self.friction = friction
        self.n_dim=n_dim
        self.softmax_postproc = softmax_postproc

        # =======  LOSS  =======
        self.loss_fn = GeneratorLoss(r=r,
                                     eta=eta, 
                                     alpha=alpha, 
                                     friction=friction, 
                                     descriptors_derivatives=descriptors_derivatives,
                                     n_dim=n_dim,
                                     split=split,
                                     softmax_postproc=self.softmax_postproc
                                     )        


        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS ======= 
        if not self._override_model:
            # initialize NN
            o = "nn"
            # set default activation to tanh
            if "activation" not in options[o]:
                options[o]["activation"] = "tanh"
        
            self.nn = FeedForward(self.layers, **options[o])
        else:
            self.nn = model
        
        if self.softmax_postproc:
            self.postprocessing=SoftmaxPostProcessing(r)
        # For inference, we provide only the softmaxs, because the Generator.compute learns a linear combimation of the representation,
        # Therefore, there is no need for two linear layers, one in the representation and one in the Generator.compute.
        self.generator = Generator(in_features=r, out_features=r, feature_method=self.forward_nn)

    def compute_eigenfunctions(self,
                               datamodule : DictModule,        
                               eta : float = None, 
                               friction : float = None,         
                               tikhonov_reg : float = 1e-4, 
                               descriptors_derivatives : Union[SmartDerivatives, torch.Tensor] = None,
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute generator eigenfunctions from the learned representation.

        The method computes the eigenfunctions and eigenvalues of the infinitesimal generator using the learned representation and weighted configurations from the training dataloader.

        Parameters
        ----------
        datamodule : DictModule
            Datamodule containing at least ``"data"`` and ``"weights"``. If runtime-cell preprocessing is used, the dataset must also provide the required cell information.
        eta : float, optional
            Resolvent shift parameter. Defaults to the value used at initialization.
        friction : torch.Tensor, optional
            Per-atom diffusion prefactors. Defaults to the values provided at initialization.
        tikhonov_reg : float, default=1e-4
            Tikhonov regularization strength added to the shifted covariance matrix before solving the eigenvalue problem.
        descriptors_derivatives : SmartDerivatives or torch.Tensor, optional
            Descriptor derivatives used to compute coordinate gradients.

        Returns
        -------
        eigenfunctions : torch.Tensor
            Eigenfunctions evaluated on the processed samples, with shape ``(n_samples, n_modes)``.
        evals : torch.Tensor
            Generator eigenvalues, with shape ``(n_modes,)``.
        evecs : torch.Tensor
            Eigenvectors mapping the learned representation to eigenfunctions, with shape ``(n_basis, n_modes)``.

        Notes
        -----
        When ``softmax_postproc=True``, a constant basis function is appended, giving ``n_basis = r + 1``. Otherwise, ``n_basis = r``.
        The number of returned modes may be smaller than the number of basis functions due to numerical filtering.
        """
        # inherit friction and eta from the model if not provided
        if friction is None:
            friction = self.friction
        if eta is None:
            eta = self.eta
        
        # check if using GNN
        is_graph = isinstance(self.nn, BaseGNN)
        
        if is_graph:
            cell=None
        else:
            cell= self._get_batch_cell(datamodule.dataset)
        eigenfunctions, evals, evecs, output = self.generator.compute(dataloader=datamodule.train_dataloader(),
                                                                      eta=eta,
                                                                      friction=friction,
                                                                      tikhonov_reg=tikhonov_reg,
                                                                      descriptors_derivatives=descriptors_derivatives,
                                                                      n_dim=self.n_dim,
                                                                      softmax_postproc=self.softmax_postproc,
                                                                      is_graph=is_graph,
                                                                      cell=cell
                                                )
            
            # register evals and evecs to the model


        return eigenfunctions, evals, evecs

    def forward_nn(self, x, cell=None):
        """Compute the neural-network representation before eigenfunction projection.

        Parameters
        ----------
        x : torch.Tensor or torch_geometric.data.Batch
            Input configurations, descriptors, or graph data.
        cell : torch.Tensor, optional
            Simulation cell used by preprocessing when required.

        Returns
        -------
        torch.Tensor
            Learned representation with shape ``(n_samples, r)``.
        """
        if self.preprocessing is not None:
            x = self._apply_module(self.preprocessing, x, cell=cell)
        z = self.nn(x)
        return z

    def forward(self, x, cell=None):
        """Evaluate the learned representation or generator eigenfunctions.

        Before calling ``compute_eigenfunctions()``, returns the raw neural-network representation.
        After eigenfunction computation, projects the representation onto the learned eigenfunctions.

        Parameters
        ----------
        x : torch.Tensor or torch_geometric.data.Batch
            Input configurations, descriptors, or graph data.
        cell : torch.Tensor, optional
            Simulation cell used by preprocessing when required.

        Returns
        -------
        torch.Tensor
            Raw representation with shape ``(n_samples, r)`` before eigenfunction computation,
            or projected eigenfunctions with shape ``(n_samples, n_modes)`` afterward.
        """
        if self.generator.evecs is not None:

            output = self.forward_nn(x, cell=cell)
            if self.softmax_postproc:
                output = torch.nn.functional.softmax(output,dim=-1)
                one_column = torch.ones((output.shape[0],1))
                output = torch.cat((output,one_column),dim=1)
            eigenfunctions = output @ self.generator.evecs.to(output.device)
            return eigenfunctions
        else: #This should only be called upon initialization
            return self.forward_nn(x, cell=cell)

    def training_step(self, 
                      train_batch, 
                      batch_idx):
        """Compute and return the training loss and record metrics."""
        torch.set_grad_enabled(True)
        if isinstance(self.nn, FeedForward):
        # =================get data===================
            x = train_batch["data"]
        # check data are have shape (n_data, -1)
            x = x.reshape((x.shape[0], -1))

            x.requires_grad = True

            weights = train_batch["weights"]
        elif isinstance(self.nn, BaseGNN):
            x = self._setup_graph_data(train_batch)
            labels = x['graph_labels']
            weights = x['weight'].clone()
        try:
            ref_idx = train_batch["ref_idx"]
        except KeyError:
            ref_idx = None 

        cell = self._get_batch_cell(train_batch)

        # =================forward====================
        # we use forward and not forward_cv to also apply the preprocessing (if present)
        z = self.forward_nn(x, cell=cell)
        if self.postprocessing is not None:
            q=self.postprocessing(z)
        else:
            q=z
        # ===================loss=====================
        if self.training:
            loss, loss_ef, loss_ortho = self.loss_fn(x, q, weights, ref_idx)
        else:
            loss, loss_ef, loss_ortho = self.loss_fn(x, q, weights, ref_idx)
        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_var", loss_ef, on_epoch=True)
        self.log(f"{name}_loss_ortho", loss_ortho, on_epoch=True)
        return loss
