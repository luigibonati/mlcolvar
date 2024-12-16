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
