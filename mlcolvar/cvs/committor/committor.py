import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
from mlcolvar.core.loss import CommittorLoss
from mlcolvar.core.nn.utils import Custom_Sigmoid

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
    .. [*] P. Kang, E. Trizio, and M. Parrinello, "Computing the Committor using the Committor: an Anatomy of the Transition state Ensemble", xxxx yy, 20zz

    See also
    --------
    mlcolvar.core.loss.CommittorLoss
        Kolmogorov's variational optimization of committor and imposition of boundary conditions
    mlcolvar.cvs.committor.utils.compute_committor_weights
        Utils to compute the appropriate weights for the training set
    mlcolvar.cvs.committor.utils.initialize_committor_masses
        Utils to initialize the masses tensor for the training
    """

    BLOCKS = ["nn", "sigmoid"]

    def __init__(
        self, 
        layers: list,
        mass: torch.Tensor,
        alpha: float,
        gamma: float = 10000,
        delta_f: float = 0,
        cell: float = None,
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
        gamma : float, optional
            Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound), by default 10000
        delta_f : float, optional
            Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
            State B is supposed to be higher in energy.
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['nn'] .
        """
        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs) 
        
        # =======  LOSS  =======
        self.loss_fn = CommittorLoss(mass=mass,
                                     alpha=alpha,
                                     gamma=gamma,
                                     delta_f=delta_f,
                                     cell=cell
        )

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======
        # initialize NN turning
        o = "nn"
        self.nn = FeedForward(layers, **options[o])

        # separately add sigmoid activation on last layer, this way it can be deactived
        o = "sigmoid"
        if (options[o] is not False) and (options[o] is not None):
            self.sigmoid = Custom_Sigmoid(**options[o])


    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        # =================get data===================
        x = train_batch["data"]
        x.requires_grad = True

        labels = train_batch["labels"]
        weights = train_batch["weights"]

        # =================forward====================
        # we use forward and not forward_cv to also apply the preprocessing (if present)
        q = self.forward(x)
        # ===================loss=====================
        if self.training:
            loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
                x, q, labels, weights 
            )
        else:
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


def test_committor():
    from mlcolvar.data import DictDataset, DictModule
    from mlcolvar.cvs.committor.utils import initialize_committor_masses

    atomic_masses = initialize_committor_masses(atoms_map=[[1,1]], n_dims=2)
    model = Committor(layers=[2, 4, 2, 1], mass=atomic_masses, alpha=1e-1, delta_f=0)

    # create dataset
    samples = 50
    X = torch.randn((2*samples, 2))
    
    # create labels
    y = torch.zeros(X.shape[0])
    y[samples:] += 1
    
    # create weights
    w = torch.ones(X.shape[0])

    dataset = DictDataset({"data": X, "labels": y, "weights": w})
    datamodule = DictModule(dataset, lengths=[1])
    
    # train model
    trainer = lightning.Trainer(max_epochs=5, logger=None, enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0)
    trainer.fit(model, datamodule)

    model(X).sum().backward()

if __name__ == "__main__":
    test_committor()