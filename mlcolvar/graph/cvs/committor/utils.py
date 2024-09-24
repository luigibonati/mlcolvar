import torch
import numpy as np

from typing import Tuple, Dict, Optional, List
from mlcolvar.graph.cvs.cv import GraphBaseCV
from mlcolvar.graph import data as gdata
from mlcolvar.graph import utils as gutils

"""
GNN committor utils.
"""

__all__ = [
    'GraphCommittorLoss',
    'get_dataset_kolmogorov_bias',
    'compute_committor_weights'
]


class GraphCommittorLoss(torch.nn.Module):
    """
    Compute Kolmogorov's variational principle loss and impose boundary
    conditions on the metastable states. Modified for Graph Neural Networks
    (GNNs).

    Parameters
    ----------
    atomic_masses : torch.Tensor
        Atomic masses of the atoms in the system.
    alpha : float
        Hyperparamer that scales the boundary conditions contribution to loss,
        i.e. alpha*(loss_bound_A + loss_bound_B)
    gamma : float
        Hyperparamer that scales the whole loss to avoid too small numbers,
        i.e. gamma*(loss_var + loss_bound), by default 10000
    delta_f : float
        Delta free energy between A (label 0) and B (label 1), units is kBT,
        by default 0. State B is supposed to be higher in energy.

    See Also
    --------
    mlcolvar.core.loss.committor.CommittorLoss
        The original `CommittorLoss` module.
    """

    def __init__(
        self,
        atomic_masses: torch.Tensor,
        alpha: float,
        gamma: float = 10000.0,
        delta_f: float = 0.0
    ) -> None:
        super().__init__()
        atomic_masses = torch.tensor(
            atomic_masses, dtype=torch.get_default_dtype()
        )
        self.register_buffer('atomic_masses', atomic_masses)
        self.alpha = alpha
        self.gamma = gamma
        self.delta_f = delta_f

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        q: torch.Tensor,
        create_graph: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data batch.
        q : torch.Tensor
            Committor quess q(x), it is the output of NN
        """
        return graph_committor_loss(
            data=data,
            q=q,
            atomic_masses=self.atomic_masses,
            alpha=self.alpha,
            gamma=self.gamma,
            delta_f=self.delta_f,
            create_graph=create_graph
        )


def graph_committor_loss(
    data: Dict[str, torch.Tensor],
    q: torch.Tensor,
    atomic_masses: torch.Tensor,
    alpha: float,
    gamma: float = 10000.0,
    delta_f: float = 0.0,
    create_graph: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute variational loss for committor optimization with boundary
    conditions.Modified for Graph Neural Networks (GNNs).

    Parameters
    ----------
    data: Dict[str, torch.Tensor]
        The data batch.
    q : torch.Tensor
        Committor quess q(x), it is the output of NN
    atomic_masses : torch.Tensor
        List of masses of all the atoms we are using.
    alpha : float
        Hyperparamer that scales the boundary conditions contribution to loss,
        i.e. alpha*(loss_bound_A + loss_bound_B)
    gamma : float
        Hyperparamer that scales the whole loss to avoid too small numbers,
        i.e. gamma*(loss_var + loss_bound)
    delta_f : float
        Delta free energy between A (label 0) and B (label 1), units is kBT.
    create_graph : bool
        Make loss backwardable, deactivate for validation to save memory.

    Returns
    -------
    loss : torch.Tensor
        Loss value.
    gamma*loss_var : torch.Tensor
        The variational loss term
    gamma*alpha*loss_a : torch.Tensor
        The boundary loss term on basin A
    gamma*alpha*loss_b : torch.Tensor
        The boundary loss term on basin B

    See Also
    --------
    mlcolvar.core.loss.committor.committor_loss
        The original `committor_loss` function.
    """
    # inherit right device and dtpye
    dtype = data['positions'].dtype
    device = data['positions'].device

    atomic_masses = atomic_masses.to(dtype).to(device)

    # Create masks to access different states data
    labels = data['graph_labels'].long().squeeze()
    mask_a = labels == 0
    mask_b = labels == 1
    mask_t = labels > 1

    # Update weights of basin B using the information on the delta_f
    factor = torch.exp(torch.tensor([delta_f], dtype=dtype, device=device))
    weights = data['weight'].clone()
    if delta_f < 0:  # B higher in energy --> A-B < 0
        weights[mask_b] = weights[mask_b] * factor
    if delta_f > 0:  # A higher in energy --> A-B > 0
        weights[mask_a] = weights[mask_a] * factor

    # Each loss contribution is scaled by the number of samples

    # We need the gradient of q(x)
    grad_outputs: Optional[List[Optional[torch.Tensor]]] = [
        torch.ones_like(q, device=device)
    ]
    gradients = torch.autograd.grad(
        [q],
        [data['positions']],
        grad_outputs=grad_outputs,
        retain_graph=True,
        create_graph=create_graph
    )[0]  # [n_nodes, 3]
    assert gradients is not None

    # we sanitize the shapes of mass and weights tensors
    node_types = torch.where(data['node_attrs'])[1]  # [n_graphs, 1]
    atomic_masses = atomic_masses[node_types].unsqueeze(-1)  # [n_nodes, 1]
    weights = weights.unsqueeze(-1)  # [n_graphs, 1]

    # square, do the mass-weight, and sum over Cartesian dims
    gradients_atomic = torch.pow(gradients, 2) / atomic_masses  # [n_nodes, 3]
    gradients_atomic = torch.sum(
        gradients_atomic, dim=1, keepdim=True
    )  # [n_nodes, 1]
    # sum over batchs
    gradients_batch = gutils.torch_tools.scatter_sum(
        gradients_atomic, data['batch'], dim=0
    )  # [n_graphs, 1]
    # ensemble avg.
    loss_v = torch.mean((gradients_batch * weights)[mask_t])  # [,]

    # boundary conditions
    loss_a = torch.mean(torch.pow(q[mask_a], 2))
    loss_b = torch.mean(torch.pow((q[mask_b] - 1.0), 2))

    loss = loss_v.log() + gamma * (alpha * (loss_a + loss_b))

    return (
        loss, loss_v.log(), alpha * gamma * loss_a, alpha * gamma * loss_b
    )


def get_dataset_kolmogorov_bias(
    model: GraphBaseCV,
    dataset: gdata.GraphDataSet,
    beta: float,
    epsilon: float = 1E-6,
    lambd: float = 0.0,
    batch_size: int = None,
    device: str = 'cpu',
    show_progress: bool = True,
    progress_prefix: str = 'Calculating KM Bias'
) -> np.ndarray:
    """
    Wrappper class to compute the Kolmogorov bias V_K from a GNN-based
    committor model.

    Parameters
    ----------
    input_model : torch.nn.Module
        Model to compute the bias from.
    dataset: mlcovar.graph.data.GraphDataSet
        Dataset on which to compute the bias.
    beta: float
        Inverse temperature in the right energy units, i.e. 1/(k_B*T)
    epsilon : float
        Regularization term in the logarithm.
    lambd : float
        Multiplicative term for the whole bias.
    batch_size:
        Batch size used for evaluating the CV.
    show_progress: bool
        If show the progress bar.
    """
    epsilon = torch.tensor(epsilon, dtype=torch.float64)

    datamodule = gdata.GraphDataModule(
        dataset,
        lengths=(1.0,),
        batch_size=batch_size,
        random_split=False,
        shuffle=False
    )
    datamodule.setup()

    gradients_list = []

    if show_progress:
        items = gutils.progress.pbar(
            datamodule.train_dataloader(),
            frequency=0.001,
            prefix=progress_prefix
        )
    else:
        items = datamodule.train_dataloader()

    for batchs in items:
        batch_dict = batchs.to(device).to_dict()
        q = model(batch_dict)[:, 1].unsqueeze(-1)
        grad_outputs: Optional[List[Optional[torch.Tensor]]] = [
            torch.ones_like(q, device=device)
        ]
        gradients = torch.autograd.grad(
            outputs=[q],
            inputs=[batch_dict['positions']],
            grad_outputs=grad_outputs,
            retain_graph=False,
            create_graph=False,
        )[0]

        # square and sum over Cartesian dims
        gradients_atomic = torch.pow(gradients, 2)  # [n_nodes, 3]
        gradients_atomic = torch.sum(
            gradients_atomic, dim=1, keepdim=True
        )  # [n_nodes, 1]
        # sum over batchs
        gradients_batch = gutils.torch_tools.scatter_sum(
            gradients_atomic, batch_dict['batch'], dim=0
        )  # [n_graphs, 1]

        gradients_list.append(gradients_batch)

    gradients = torch.vstack(gradients_list)
    bias = -lambd * (1 / beta) * (
        torch.log(gradients + epsilon) - torch.log(epsilon)
    )

    return bias.cpu().numpy()


def compute_committor_weights(
    dataset: gdata.GraphDataSet,
    bias: torch.Tensor,
    beta: float
) -> gdata.GraphDataSet:
    """
    Utils to update a `GraphDataSet` object with the appropriate weights for
    the training set for the learning of committor function.

    Parameters
    ----------
    dataset: mlcovar.graph.data.GraphDataSet
        The graph dataset.
    bias : torch.Tensor
        Bias values for the data in the dataset, usually it should be the
        committor-based bias.
    beta : float
        Inverse temperature in the right energy units

    Returns
    -------
    dataset: mlcovar.graph.data.GraphDataSet
        Updated dataset with weights and updated labels.
    """
    assert len(dataset) == len(bias)

    bias = torch.tensor(bias, dtype=torch.get_default_dtype())
    if bias.isnan().any():
        raise ValueError(
            'Found Nan(s) in bias tensor. Check before proceeding! '
            + 'If no bias was applied replace Nan with zero!'
        )

    # TODO sign if not from committor bias
    weights = torch.exp(beta * bias)
    labels = torch.tensor(
        [d['graph_labels'][0, 0] for d in dataset]
    ).long()

    for i in np.unique(labels.cpu().numpy()):
        # compute average of exp(beta*V) on this simualtions
        coeff = 1 / torch.mean(
            weights[torch.nonzero(labels == i, as_tuple=True)]
        )
        # update the weights
        weights[
            torch.nonzero(labels == i, as_tuple=True)
        ] = coeff * weights[
            torch.nonzero(labels == i, as_tuple=True)
        ]

    # update dataset
    for i in range(len(dataset)):
        dataset[i]['weight'] = weights[i]

    return dataset
