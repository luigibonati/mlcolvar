import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
from mlcolvar.core.loss import GeneratorLoss, compute_eigenfunctions, evaluate_eigenfunctions

__all__ = ["Generator", "Generator_singleNN"]
class Generator(BaseCV, lightning.LightningModule):
    """
    Baseclass for learning a representation for the eigenfunctions of the generator. 
    The representation is expressed as a concatenation of the output of r neural networks.
    **Data**: for training it requires a DictDataset with the keys 'data', and 'weights' 
              and optionally 'derivatives' which should contain the descriptors derivatives
    **Loss**: Minimize the representation loss and the orthonormalization loss
    
    """
    BLOCKS = ["nn"]

    def __init__(
        self, 
        layers: list,
        eta: float,
        r: int,
        alpha: float,
        cell: float = None,
        friction = None,
        options: dict = None,

        **kwargs,
    ):
        """Define a NN-based generator model

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        eta : float
            Hyperparameter for the shift to define the resolvent. $(\eta I-_mathcal{L})^{-1}$
        r : int
            Hyperparamer for the number of eigenfunctions wanted
        alpha : float
            Hyperparamer that scales the orthonormality loss
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        friction: torch.tensor, optional 
            Langevin friction which should contain \sqrt{k_B*T/(gamma*m_i)}
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['nn'] .
        """
        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs) 
        
        # =======  LOSS  =======
        self.loss_fn = GeneratorLoss(self.forward,
                                     eta=eta,
                                     alpha=alpha,
                                     cell=cell,
                                     friction=friction,
                                     n_cvs=r
        )
        self.r = r
        self.eta = eta
        self.friction = friction
        self.cell = cell
        self.evecs = None
        self.evals = None 
        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======
        # initialize NN turning
        o = "nn"
        # set default activation to tanh
        if "activation" not in options[o]: 
            options[o]["activation"] = "tanh"
        self.nn = torch.nn.ModuleList([FeedForward(layers, **options[o]) for idx in range(r)])
    
    def compute_eigenfunctions(self, dataset, friction=None, eta=None, r=None,cell=None, tikhonov_reg=1e-4):
        if friction is None:
            friction = self.friction
        if eta is None:
            eta = self.eta
        if r is None:
            r = self.r
        if cell is None:
            cell = self.cell
        if self.evecs is None:
            eigenfunctions, evals, evecs = compute_eigenfunctions(self.forward, dataset, friction, eta, r, cell)
            self.evals = evals
            self.evecs = evecs
            return eigenfunctions, evals, evecs
        else:
            eigenfunctions = evaluate_eigenfunctions(self.forward, dataset,self.evecs)
            return eigenfunctions, self.evals, self.evecs

    def forward_cv(self, x: torch.Tensor) -> (torch.Tensor):

        return torch.cat([nn(x) for nn in self.nn], dim=1)

    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        torch.set_grad_enabled(True)
        # =================get data===================
        x = train_batch["data"]
        # check data are have shape (n_data, -1)
        x = x.reshape((x.shape[0], -1))

        
        x.requires_grad = True

        weights = train_batch["weights"]
        if "derivatives" in train_batch.keys():
            derivatives = train_batch["derivatives"]
        else:
            derivatives = None

        # =================forward====================
        # we use forward and not forward_cv to also apply the preprocessing (if present)
        q = self.forward(x)
        # ===================loss=====================
        if self.training:
            loss, loss_ef, loss_ortho = self.loss_fn(
                x, q, weights, derivatives 
            )
        else:
            loss, loss_ef, loss_ortho = self.loss_fn(
                x, q, weights, derivatives 
            )
        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_var", loss_ef, on_epoch=True)
        self.log(f"{name}_loss_ortho", loss_ortho, on_epoch=True)
        return loss


class Generator_singleNN(BaseCV, lightning.LightningModule):
    """
    This is just a test to see how the model perform when using a single NN instead of concatenated ones
    Baseclass for learning a representation for the eigenfunctions of the generator. 
    The representation is expressed as a concatenation of the output of r neural networks.
    **Data**: for training it requires a DictDataset with the keys 'data', and 'weights' 
              and optionally 'derivatives' which should contain the descriptors derivatives
    **Loss**: Minimize the representation loss and the orthonormalization loss
    
    """
    BLOCKS = ["nn"]

    def __init__(
        self, 
        layers: list,
        eta: float,
        r: int,
        alpha: float,
        cell: float = None,
        friction = None,
        options: dict = None,

        **kwargs,
    ):
        """Define a NN-based generator model

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        eta : float
            Hyperparameter for the shift to define the resolvent. $(\eta I-_mathcal{L})^{-1}$
        r : int
            Hyperparamer for the number of eigenfunctions wanted
        alpha : float
            Hyperparamer that scales the orthonormality loss
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        friction: torch.tensor, optional 
            Langevin friction which should contain \sqrt{k_B*T/(gamma*m_i)}
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['nn'] .
        """
        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs) 
        
        # =======  LOSS  =======
        self.loss_fn = GeneratorLoss(self.forward,
                                     eta=eta,
                                     alpha=alpha,
                                     cell=cell,
                                     friction=friction,
                                     n_cvs=r
        )

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======
        # initialize NN turning
        o = "nn"
        # set default activation to tanh
        if "activation" not in options[o]: 
            options[o]["activation"] = "tanh"
        self.nn = FeedForward(layers, **options[o])


    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        torch.set_grad_enabled(True)
        # =================get data===================
        x = train_batch["data"]
        # check data are have shape (n_data, -1)
        x = x.reshape((x.shape[0], -1))

        
        x.requires_grad = True

        weights = train_batch["weights"]
        if "derivatives" in train_batch.keys():
            derivatives = train_batch["derivatives"]
        else:
            derivatives = None

        # =================forward====================
        # we use forward and not forward_cv to also apply the preprocessing (if present)
        q = self.forward(x)
        # ===================loss=====================
        if self.training:
            loss, loss_ef, loss_ortho = self.loss_fn(
                x, q, weights, derivatives 
            )
        else:
            loss, loss_ef, loss_ortho = self.loss_fn(
                x, q, weights, derivatives 
            )
        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_var", loss_ef, on_epoch=True)
        self.log(f"{name}_loss_ortho", loss_ortho, on_epoch=True)
        return loss