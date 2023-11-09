import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
from mlcolvar.core.loss import CommittorLoss

__all__ = ["Committor"]


class Committor(BaseCV, lightning.LightningModule):
    """
    Committor class
    """

    BLOCKS = ["nn"]

    def __init__(
        self, 
        layers: list,
        mass: torch.Tensor,
        alpha : float,
        cell_size: float = None,
        gamma : float = 10000,
        delta_f: float = 0,
        options: dict = None,
        **kwargs,
    ):
        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs) 
        
        # =======  LOSS  =======
        self.loss_fn = CommittorLoss(mass=mass,
                                     alpha=alpha,
                                     cell_size=cell_size,
                                     gamma=gamma,
                                     delta_f=delta_f
        )

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)
        
        # add the relevant nn options, set tanh for hidden layers and sharp sigmoid for output layer
        activ_list = ["tanh" for i in range( len(layers) - 1 )]
        activ_list.append("sharp_sigmoid")
        # update options dict for activations if not already set
        if not "activation" in options["nn"]:
            options["nn"]["activation"] = activ_list

        # ======= CHECKS =======
        # should be empty in this case


        # ======= BLOCKS =======
        # initialize NN turning on last layer activation
        o = "nn"
        self.nn = FeedForward(layers, last_layer_activation=True, **options[o])

    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        # =================get data===================
        x = train_batch["data"]
        labels = train_batch["labels"]
        weights = train_batch["weights"]

        # =================forward====================
        q = self.forward_cv(x)
        # ===================loss=====================
        loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
            x, q, labels, weights 
        )
        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_var", loss_var, on_epoch=True)
        self.log(f"{name}_loss_bound_A", loss_bound_A, on_epoch=True)
        self.log(f"{name}_loss_bound_B", loss_bound_B, on_epoch=True)
        return loss
    

# TODO add test function