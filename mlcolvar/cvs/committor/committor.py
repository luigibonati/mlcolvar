import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization, BaseGNN
from mlcolvar.core.loss import CommittorLoss
from mlcolvar.core.nn.utils import Custom_Sigmoid
from typing import Union, List

__all__ = ["Committor"]


class Committor(BaseCV):
    """Base class for data-driven learning of committor function.
    The committor function q is expressed as the output of a neural network optimized with a self-consistent
    approach based on the Kolmogorov's variational principle for the committor and on the imposition of its boundary conditions (see Refs. [1,2]).
    It is also possible to use an approximated variational approach without explicit dependence on the atomic coordinates (see Ref. [3]).
 

    **Data**: for training it requires a DictDataset containing:
        - If using descriptors as input, the keys 'data', 'labels' and 'weights'.
        - If using graphs as input, `torch_geometric.data` with 'graph_labels' and 'weight' in their 'data_list'.
        
    **Loss**: Minimize Kolmogorov's variational functional of q and impose boundary condition on the metastable states (CommittorLoss) from Refs. [1,2].
              It is also possible to use an approximated variational approach without explicit dependence on the atomic coordinates
    
    References
    ----------
    .. [1] P. Kang, E. Trizio, and M. Parrinello, "Computing the committor using the committor to study the transition state ensemble", Nat. Comput. Sci., 2024, DOI: 10.1038/s43588-024-00645-0
    .. [2] E. Trizio, P. Kang, and M. Parrinello, "Everything everywhere all at once: a probability-based enhanced sampling approach to rare events", Nat. Comput. Sci., 2025, DOI: 10.1038/s43588-025-00799-5
    .. [3] E. Trizio, G. Rossi, and M. Parrinello, "Ceci n'est pas un committor: Efficient sampling via approximated committor functions", J Chem. Phys., 2026, DOI: 10.1063/5.0331622

    See also
    --------
    mlcolvar.cvs.committor.utils.compute_committor_weights
        Utils to compute the appropriate weights for the training set
    mlcolvar.cvs.committor.utils.initialize_committor_masses
        Utils to initialize the masses tensor for the training
    mlcolvar.core.loss.CommittorLoss
        Kolmogorov's variational optimization of committor and imposition of boundary conditions
    mlcolvar.core.loss.utils.SmartDerivatives
        Class to optimize the gradients calculation imporving speed and memory efficiency.
    """

    DEFAULT_BLOCKS = ["norm_in", "nn", "sigmoid"]
    MODEL_BLOCKS = ["nn", "sigmoid"]

    def __init__(
        self, 
        model: Union[List[int], FeedForward, BaseGNN],
        alpha: float,
        atomic_masses: torch.Tensor = None,
        gamma: float = 10000,
        delta_f: float = 0,
        separate_boundary_dataset: bool = True,
        descriptors_derivatives: torch.nn.Module = None,
        log_var: bool = False,
        use_gradients_wrt_positions: bool = True,
        z_regularization: float = 0.0,
        z_threshold: float = None,
        n_dim: int = None,
        norm_in: bool = False,
        options: dict = None,
        **kwargs,
    ):
        """Define a NN-based committor model

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        alpha : float
            Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
        atomic_masses : torch.Tensor
            List of masses of all the atoms we are using, for each atom we need to repeat three times for x,y,z, by default None.
            The mlcolvar.cvs.committor.utils.initialize_committor_masses can be used to simplify this.
            If the position-less loss is used, this must be set to None.
        gamma : float, optional
            Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound), by default 10000
        delta_f : float, optional
            Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
            State B is supposed to be higher in energy.
        separate_boundary_dataset : bool, optional
            Switch to exculde boundary condition labeled data from the variational loss, by default True
        descriptors_derivatives : torch.nn.Module, optional
            `SmartDerivatives` object to save memory and time when using descriptors. Cannot be used with GNN models.
            See also mlcolvar.core.loss.committor_loss.SmartDerivatives
        log_var : bool, optional
            Switch to minimize the log of the variational functional, by default False.
        use_gradients_wrt_positions : bool, optional
            Whether to use gradients with respect to positions as prescribed in the original Kolmogorov variational functional, by default True.
            Set to false to use the approximated variational principle defined in Ref. [3] without explicit dependence on the atomic coordinates derivatives.
        z_regularization : float, optional
            Scales a regularization on the learned z space preventing it from exceeding the threshold given with 'z_threshold'.
            The magnitude of the regularization is scaled by the given number, by default 0.0
        z_threshold : float, optional
            Sets a maximum threshold for the z value during the training, by default None. 
            The magnitude of the regularization term is scaled via the `z_regularization` key.
        n_dim : int
            Number of dimensions, by default None. 
            If None, it defaults to 3 for the position-based loss and to 1 for the position-less loss.
        norm_in : bool
            Whether to normalize the input of the NN model, by default False.
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['nn'].
        """
        super().__init__(model, **kwargs) 
                
        if use_gradients_wrt_positions and atomic_masses is None:
            raise ValueError("atomic_masses must be provided when using Kolmogorov variational functional (use_gradients_wrt_positions is True)")
        elif not use_gradients_wrt_positions:
            if atomic_masses is not None:
                raise ValueError("atomic_masses must be None when using approximated variational principle (use_gradients_wrt_positions is False)")
            if descriptors_derivatives is not None:
                raise ValueError("descriptors_derivatives must be None when using approximated variational principle (use_gradients_wrt_positions is False)")
    
        # =======  LOSS  =======
        self.loss_fn = CommittorLoss(alpha=alpha,
                                     atomic_masses=atomic_masses,
                                     gamma=gamma,
                                     delta_f=delta_f,
                                     separate_boundary_dataset=separate_boundary_dataset,
                                     descriptors_derivatives=descriptors_derivatives,
                                     log_var=log_var,
                                     use_gradients_wrt_positions=use_gradients_wrt_positions,
                                     z_regularization=z_regularization,
                                     z_threshold=z_threshold,
                                     n_dim=n_dim
        )

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======
        if not self._override_model:
            # Initialize norm_in
            o = "norm_in"
            if norm_in and (options[o] is not False) and (options[o] is not None):
                self.norm_in = Normalization(self.in_features, **options[o])

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

    def forward_nn(self, x, cell=None):
        if self.preprocessing is not None:
            x = self._apply_module(self.preprocessing, x, cell=cell)
        if not self._override_model and self.norm_in is not None:
            x = self.norm_in(x)
        z = self.nn(x)
        return z

    def training_step(self, train_batch, batch_idx):
        torch.set_grad_enabled(True)

        """Compute and return the training loss and record metrics."""
        # =================get data===================
        if isinstance(self.nn, FeedForward):
            x = train_batch["data"]
            # check data have shape (n_data, -1)
            x = x.reshape((x.shape[0], -1))
            x.requires_grad = True

            labels = train_batch["labels"]
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
        z = self.forward_nn(x, cell=cell)
        
        if self.sigmoid is not None:
            q = self.sigmoid(z)
        else:
            q = z        
        
        # ===================loss=====================
        if self.training:
            loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
                x, z, q, labels, weights, ref_idx
            )
        else:
            loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
                x, z, q, labels, weights, ref_idx
            )

        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_var", loss_var, on_epoch=True)
        self.log(f"{name}_loss_bound_A", loss_bound_A, on_epoch=True)
        self.log(f"{name}_loss_bound_B", loss_bound_B, on_epoch=True)
        return loss
