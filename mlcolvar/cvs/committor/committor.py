import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, BaseGNN
from mlcolvar.core.loss import CommittorLoss
from mlcolvar.core.nn.utils import Custom_Sigmoid
from typing import Union, List

__all__ = ["Committor"]


class Committor(BaseCV, lightning.LightningModule):
    """Base class for data-driven learning of committor function.
    The committor function q is expressed as the output of a neural network optimized with a self-consistent
    approach based on the Kolmogorov's variational principle for the committor and on the imposition of its boundary conditions. 

    **Data**: for training it requires a DictDataset containing:
        - If using descriptors as input, the keys 'data', 'labels' and 'weights'.
        - If using graphs as input, `torch_geometric.data` with 'graph_labels' and 'weight' in their 'data_list'.
        
    **Loss**: Minimize Kolmogorov's variational functional of q and impose boundary condition on the metastable states (CommittorLoss)
    
    References
    ----------
    .. [*] P. Kang, E. Trizio, and M. Parrinello, "Computing the committor using the committor to study the transition state ensemble", Nat. Comput. Sci., 2024, DOI: 10.1038/s43588-024-00645-0

    See also
    --------
    mlcolvar.core.loss.CommittorLoss
        Kolmogorov's variational optimization of committor and imposition of boundary conditions
    mlcolvar.cvs.committor.utils.compute_committor_weights
        Utils to compute the appropriate weights for the training set
    mlcolvar.cvs.committor.utils.initialize_committor_masses
        Utils to initialize the masses tensor for the training
    """

    DEFAULT_BLOCKS = ["nn", "sigmoid"]
    MODEL_BLOCKS = ["nn", "sigmoid"]

    def __init__(
        self, 
        model: Union[List[int], FeedForward, BaseGNN],
        mass: torch.Tensor,
        alpha: float,
        gamma: float = 10000,
        delta_f: float = 0,
        separate_boundary_dataset: bool = True,
        descriptors_derivatives: torch.nn.Module = None,
        log_var: bool = True,
        z_regularization: float = 0.0,
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
        separate_boundary_dataset : bool, optional
            Switch to exculde boundary condition labeled data from the variational loss, by default True
        descriptors_derivatives : torch.nn.Module, optional
            `SmartDerivatives` object to save memory and time when using descriptors. Cannot be used with GNN models.
            See also mlcolvar.core.loss.committor_loss.SmartDerivatives
        log_var : bool, optional
            Switch to minimize the log of the variational functional, by default True.
        z_regularization : float, optional
            Introduces a regularization on the learned z space avoiding too large absolute values.
            The magnitude of the regularization is scaled by the given number, by default 0.0
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['nn'].
        """
        super().__init__(model, **kwargs) 

        self.register_buffer('is_committor', torch.tensor(1, dtype=int))
        
        # =======  LOSS  =======
        self.loss_fn = CommittorLoss(atomic_masses=mass,
                                     alpha=alpha,
                                     gamma=gamma,
                                     delta_f=delta_f,
                                     separate_boundary_dataset=separate_boundary_dataset,
                                     descriptors_derivatives=descriptors_derivatives,
                                     log_var=log_var,
                                     z_regularization=z_regularization
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
        elif self._override_model:
            self.nn = model

        if self.nn.out_features != 1:
            raise ValueError('Output of the model must be of dimension 1')

        # separately add sigmoid activation on last layer, this way it can be deactived
        o = "sigmoid"
        if (options[o] is not False) and (options[o] is not None):
            self.sigmoid = Custom_Sigmoid(**options[o])

    def forward_nn(self, x):
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        z = self.nn(x)

        return z

    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        # =================get data===================
        if isinstance(self.nn, FeedForward):
            x = train_batch["data"]
            # check data are have shape (n_data, -1)
            x = x.reshape((x.shape[0], -1))
            x.requires_grad = True

            labels = train_batch["labels"]
            weights = train_batch["weights"]
        elif isinstance(self.nn, BaseGNN):
            x = self._setup_graph_data(train_batch)
            labels = x['graph_labels']
            weights = x['weight'].clone()

        # =================forward====================
        z = self.forward_nn(x)
        if self.sigmoid is not None:
            q = self.sigmoid(z)
        else:
            q = z

        # ===================loss=====================
        if self.training:
            loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
                x, z, q, labels, weights 
            )
        else:
            loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
                x, z, q, labels, weights 
            )
        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_var", loss_var, on_epoch=True)
        self.log(f"{name}_loss_bound_A", loss_bound_A, on_epoch=True)
        self.log(f"{name}_loss_bound_B", loss_bound_B, on_epoch=True)
        return loss


def test_committor():
    import os
    from mlcolvar.data import DictDataset, DictModule
    from mlcolvar.cvs.committor.utils import initialize_committor_masses, KolmogorovBias

    # create two fake atoms and use their fake positions
    atomic_masses = initialize_committor_masses(atom_types=[0,1], masses=[15.999, 1.008])
    # create dataset
    samples = 50
    X = torch.randn((4*samples, 6))
    
    # create labels
    y = torch.zeros(X.shape[0])
    y[samples:] += 1
    y[int(2*samples):] += 1
    y[int(3*samples):] += 1
    
    # create weights
    w = torch.ones(X.shape[0])

    dataset = DictDataset({"data": X, "labels": y, "weights": w})
    datamodule = DictModule(dataset, lengths=[1])
    
    # train model
    trainer = lightning.Trainer(max_epochs=5, logger=False, enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0)
    
    print()
    print('NORMAL')
    print()
    # dataset separation
    model = Committor(model=[6, 4, 2, 1], mass=atomic_masses, alpha=1e-1, delta_f=0)
    trainer.fit(model, datamodule)
    model(X).sum().backward()
    bias_model = KolmogorovBias(input_model=model, beta=1, epsilon=1e-6, lambd=1)
    bias_model(X)

    # naive whole dataset
    trainer = lightning.Trainer(max_epochs=5, logger=False, enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0)
    model = Committor(model=[6, 4, 2, 1], mass=atomic_masses, alpha=1e-1, delta_f=0, separate_boundary_dataset=False)
    trainer.fit(model, datamodule)
    model(X).sum().backward()


    print()
    print('EXTERNAL FEEDFORWARD')
    print()
    # dataset separation
    ff_model = FeedForward([6, 4, 2, 1])
    model = Committor(model=ff_model, mass=atomic_masses, alpha=1e-1, delta_f=0)
    trainer.fit(model, datamodule)
    model(X).sum().backward()
    bias_model = KolmogorovBias(input_model=model, beta=1, epsilon=1e-6, lambd=1)
    bias_model(X)

    # naive whole dataset
    trainer = lightning.Trainer(max_epochs=5, logger=False, enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0)
    model = Committor(model=ff_model, mass=atomic_masses, alpha=1e-1, delta_f=0, separate_boundary_dataset=False)
    trainer.fit(model, datamodule)
    model(X).sum().backward()


    print()
    print('EXTERNAL GNN')
    print()
    from mlcolvar.core.nn.graph import SchNetModel
    from mlcolvar.data.graph.utils import create_test_graph_input
    gnn_model = SchNetModel(1, 0.1, [1, 8])

    model = Committor(model=gnn_model, 
                      mass=atomic_masses, 
                      alpha=1e-1, 
                      delta_f=0)

    datamodule = create_test_graph_input(output_type='datamodule', n_samples=100, n_states=3, n_atoms=3)
    trainer = lightning.Trainer(max_epochs=5, logger=False, enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0, enable_model_summary=False)
    trainer.fit(model, datamodule)

    example_input_graph_test = create_test_graph_input(output_type='example', n_atoms=4, n_samples=3, n_states=2)

    model(example_input_graph_test).sum().backward()


if __name__ == "__main__":
    test_committor()