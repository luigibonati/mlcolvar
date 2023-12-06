import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
from mlcolvar.core.loss import CommittorLoss

__all__ = ["Committor"]


class Committor(BaseCV, lightning.LightningModule):
    """Base class for data-driven learning of committor function.
    The committor function q is expressed as the output of a neural network optimized with a self-consistent
    approach based on the Kolmogorov's variational principle for the committor and on the imposition of its boundary conditions. 
    TODO: Add reference upon publication

    **Data**: for training it requires a DictDataset with the keys 'data', 'labels' and 'weights'

    **Loss**: Minimize Kolmogorov's variational functional of q and impose boundary condition on the metastable states (CommittorLoss)
    
    References
    ----------
    .. [*] P. Kang, E. Trizio, and M. Parrinello, "Amazing committor paper", xxxx yy, 2zzz

    See also
    --------
    mlcolvar.core.loss.CommittorLoss
        Kolmogorov's variational optimization of committor and imposition of boundary conditions
    mlcolvar.cvs.committor.utils.compute_committor_weights
        Utils to compute the appropriate weights for the training set
    mlcolvar.cvs.committor.utils.initialize_committor_masses
        Utils to initialize the masses tensor for the training
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
        """Define a NN-based committor model

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        mass : torch.Tensor
            List of masses of all the atoms we are using, for each atom we need to repeat three times for x,y,z.
            The mlcolvar.cvs.committor.utils.initialize_committor_masses can be used to simplify this.
        alpha : float
            Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
        cell_size : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        gamma : float, optional
            Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound), by default 10000
        delta_f : float, optional
            Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
            State B is supposed to be higher in energy.
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['nn'] .
        """
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
        activ_list = ["tanh" for i in range( len(layers) - 2 )]
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
        x.requires_grad = True

        labels = train_batch["labels"]
        weights = train_batch["weights"]

        # =================forward====================
        q = self.forward_cv(x)
        # ===================loss=====================
        if self.training:
            loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
                x, q, labels, weights 
            )
        else:
            loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
                x, q, labels, weights, create_graph=False 
            )
        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_var", loss_var, on_epoch=True)
        self.log(f"{name}_loss_bound_A", loss_bound_A, on_epoch=True)
        self.log(f"{name}_loss_bound_B", loss_bound_B, on_epoch=True)
        return loss
    
# we override the default configure_optimizer function of BaseCV to allow setting a lr_scheduler
# This may be improve and/or fixed in future commits TODO !
    def configure_optimizers(self):
        """
        Initialize the optimizer based on self._optimizer_name and self.optimizer_kwargs.

        Returns
        -------
        torch.optim
            Torch optimizer
        """
        lr_scheduler_dict = self.optimizer_kwargs.pop('lr_scheduler', None)
    
        optimizer = getattr(torch.optim, self._optimizer_name)(
            self.parameters(), **self.optimizer_kwargs
        )
        if lr_scheduler_dict is not None:
            lr_scheduler_name = lr_scheduler_dict.pop('scheduler')
            lr_scheduler = {
                'scheduler': lr_scheduler_name(optimizer, **lr_scheduler_dict),
            }
            lr_scheduler_dict['scheduler'] = lr_scheduler_name
            self.optimizer_kwargs['lr_scheduler'] = lr_scheduler_dict
            return [optimizer] , [lr_scheduler]
        else: 
            return optimizer
    
    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

# TODO add test function