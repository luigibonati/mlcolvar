import torch
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
    """
    Baseclass for learning a representation for the eigenfunctions of the infinitesimal generator.
    The representation is expressed as a concatenation of the output of r neural networks.
    
    **Data**: for training it requires a DictDataset with the keys 'data', and 'weights'
    
    **Loss**: Minimize the representation loss and the orthonormalization loss

    References
    ----------
    .. [*] T. Devergne, V. Kostic, M. Pontil, M. Parrinello, "Slow dynamical modes from static averages", J. Chem. Phys., 2025, DOI: 10.1063/5.0246248

    See also
    --------
    mlcolvar.core.loss.generator_loss
        Loss function to learn a representation for the infinitesimal generator
    mlcolvar.cvs.generator.utils.compute_eigenfunctions
        Computes eigenfunctions and eigenvalues from a learned representation
    mlcolvar.cvs.generator.utils.forecast_state_occupation
        Computes the time evolution of state occupation probabilities in a dynamical system from the learned eigenfunctions

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
            Number of eigenfunctions to learn.
        model : list[int] or FeedForward or BaseGNN
            Neural-network architecture. If a list is provided, it is used as the
            layer sizes for a :class:`FeedForward` model. If a model instance is
            provided, it is used directly.
        eta : float
            Resolvent shift parameter used in ``( I - L/eta)^-1``.
        alpha : float
            Weight of the orthonormality penalty in the total loss,
            ``loss = loss_ef + alpha * loss_ortho``.
        friction : torch.Tensor
            Langevin friction-related prefactor, usually one value per atom.
        descriptors_derivatives : SmartDerivatives or torch.Tensor, optional
            Derivatives of descriptors with respect to atomic positions. Supplying
            this can avoid recomputing descriptor derivatives during loss evaluation.
        n_dim : int, default=3
            Number of spatial dimensions.
        split : bool, default=True
            Whether to split the data internally when computing the loss.
        softmax_postproc : bool, default=True
            Whether to apply softmax-based post-processing to the network output.
        options : dict, optional
            Options passed to model blocks and optimizer configuration.
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

        If eigenvectors have already been computed and ``recompute=False``, this
        method reuses the cached eigenvectors and eigenvalues and only evaluates the
        current model on ``dataset``.

        Parameters
        ----------
        datamodule : DictModule
            Datamodule containing at least ``"data"`` and ``"weights"``. If the model
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
        if self.preprocessing is not None:
            x = self._apply_module(self.preprocessing, x, cell=cell)
        z = self.nn(x)
        return z
    
    def forward(self, x, cell=None):
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

    def evaluate_loss(
        self,
        batch,
        batch_idx: int,
        update_state: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the generator loss.

        Returns the variational and orthonormalization loss terms as metrics.
        """
        # The loss requires derivatives with respect to the inputs.
        with torch.enable_grad():
            if isinstance(self.nn, FeedForward):
                x = batch["data"].reshape((batch["data"].shape[0], -1))
                x.requires_grad_(True)
                weights = batch["weights"]
            elif isinstance(self.nn, BaseGNN):
                x = self._setup_graph_data(batch)
                weights = x["weight"].clone()

            ref_idx = batch.get("ref_idx", None)
            cell = self._get_batch_cell(batch)
            z = self.forward_nn(x, cell=cell)
            q = self.postprocessing(z) if self.postprocessing is not None else z
            loss, loss_ef, loss_ortho = self.loss_fn(x, q, weights, ref_idx)

        return {
            "loss": loss,
            "loss_var": loss_ef,
            "loss_ortho": loss_ortho,
        }
