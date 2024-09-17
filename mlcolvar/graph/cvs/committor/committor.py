
import torch
import torch_geometric as tg
from typing import Dict, Any, List

from mlcolvar.core.nn.utils import Custom_Sigmoid

from mlcolvar.graph.cvs import GraphBaseCV
from mlcolvar.graph.cvs.cv import test_get_data
from mlcolvar.graph.cvs.committor.utils import GraphCommittorLoss
from mlcolvar.graph.utils import torch_tools

"""
Data-driven learning of committor function, based on Graph Neural Networks
(GNNs).
"""

__all__ = ['GraphCommittor']


class GraphCommittor(GraphBaseCV):
    """
    Data-driven learning of committor function, based on GNNs.

    The committor function q is expressed as the output of a neural network
    optimized with a self-consistent approach based on the Kolmogorov's
    variational principle for the committor and on the imposition of its
    boundary conditions.

    Parameters
    ----------
    cutoff: float
        Cutoff radius of the basis functions. Should be the same as the cutoff
        radius used to build the graphs.
    atomic_numbers: List[int]
        The atomic numbers mapping, e.g. the `atomic_numbers` attribute of a
        `mlcolvar.graph.data.GraphDataSet` instance.
    atomic_masses : List[float]
        List of masses of all the atoms we are using.
    model_name: str
        Name of the GNN model.
    model_options: Dict[Any, Any]
        Model options.
    extra_loss_options: Dict[Any, Any]
        Extra loss function options.
    optimizer_options: Dict[Any, Any]
        Optimizer options.

    References
    ----------
    .. [*] P. Kang, E. Trizio, and M. Parrinello, "Computing the committor
        using the committor to study the transition state ensemble",
        Nat. Comput. Sci., 2024, DOI: 10.1038/s43588-024-00645-0

    See also
    --------
    mlcolvar.cvs.committor.Committor
        The feedforward NN based ML committor module.
    mlcolvar.graph.cvs.committor.utils.GraphCommittorLoss
        Kolmogorov's variational optimization of committor and imposition of
        boundary conditions.
    mlcolvar.graph.cvs.committor.utils.compute_committor_weights
        Utils to compute the appropriate weights for the training set
    """
    def __init__(
        self,
        cutoff: float,
        atomic_numbers: List[int],
        atomic_masses: List[float],
        model_name: str = 'GVPModel',
        model_options: Dict[Any, Any] = {},
        extra_loss_options: Dict[Any, Any] = {
            'alpha': 1.0, 'gamma': 100.0, 'delta_f': 0.0, 'sigmoid_p': 3.0
        },
        optimizer_options: Dict[Any, Any] = {},
        **kwargs,
    ) -> None:
        if model_options.pop('n_out', None) is not None:
            raise RuntimeError(
                'The `n_out` key of parameter `model_options` will be ignored!'
            )
        if optimizer_options != {}:
            kwargs['optimizer_options'] = optimizer_options

        super().__init__(
            1, cutoff, atomic_numbers, model_name, model_options, **kwargs
        )

        atomic_masses = torch.tensor(
            atomic_masses, dtype=torch.get_default_dtype()
        )
        self.register_buffer('atomic_masses', atomic_masses)
        self.register_buffer('is_committor', torch.tensor(1, dtype=int))

        self.sigmoid = Custom_Sigmoid(extra_loss_options.get('sigmoid_p', 3.0))

        self.loss_fn = GraphCommittorLoss(
            atomic_masses,
            alpha=float(extra_loss_options.get('alpha', 1.0)),
            gamma=float(extra_loss_options.get('gamma', 10000.0)),
            delta_f=float(extra_loss_options.get('delta_f', 0.0)),
        )

    def forward_nn(
        self,
        data: Dict[str, torch.Tensor],
        token: bool = False
    ) -> torch.Tensor:
        """
        The forward pass for the NN.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        token: bool
            To be used.
        """
        data['positions'].requires_grad_(True)
        data['node_attrs'].requires_grad_(True)

        return self._model(data)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        token: bool = False
    ) -> torch.Tensor:
        """
        The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        token: bool
            To be used.
        """
        z = self.forward_nn(data)
        q = self.sigmoid(z)

        return torch.hstack([z, q])

    def training_step(
        self, train_batch: tg.data.Batch, *args, **kwargs
    ) -> torch.Tensor:
        """
        Compute and return the training loss and record metrics.

        Parameters
        ----------
        train_batch: torch_geometric.data.Batch
            The data batch.
        """
        torch.set_grad_enabled(True)

        batch_dict = train_batch.to_dict()
        z = self.forward_nn(batch_dict)
        q = self.sigmoid(z)

        loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
            batch_dict, q
        )

        name = 'train' if self.training else 'valid'
        self.log(f'{name}_loss', loss, on_epoch=True)
        self.log(f'{name}_loss_variational', loss_var, on_epoch=True)
        self.log(f'{name}_loss_boundary_A', loss_bound_A, on_epoch=True)
        self.log(f'{name}_loss_boundary_B', loss_bound_B, on_epoch=True)
        return loss
