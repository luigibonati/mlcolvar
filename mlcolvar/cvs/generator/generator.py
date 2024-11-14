import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
from mlcolvar.core.loss import GeneratorLoss

__all__ = ["Generator"]
class Generator(BaseCV, lightning.LightningModule):

    BLOCKS = ["nn"]

    def __init__(
        self, 
        layers: list,
        eta: float,
        r: int,
        gamma: float = 10000,
        cell: float = None,
        friction = None,
        options: dict = None,
        **kwargs,
    ):

        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs) 
        
        # =======  LOSS  =======
        self.loss_fn = GeneratorLoss(self.forward,
                                     eta=eta,
                                     gamma=gamma,
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
        self.nn = torch.nn.ModuleList([FeedForward(layers, **options[o]) for idx in range(r)])


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