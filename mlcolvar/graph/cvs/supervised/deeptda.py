import torch
import torch_geometric as tg
import numpy as np
from typing import Dict, Any, List, Union

from mlcolvar.core.loss import TDALoss
from mlcolvar.graph.cvs import GraphBaseCV
from mlcolvar.graph.cvs.cv import test_get_data
from mlcolvar.graph import data as gdata
from mlcolvar.graph.utils import torch_tools

"""
The Deep Targeted Discriminant Analysis (Deep-TDA) CV based on Graph Neural
Networks (GNN).
"""

__all__ = ['GraphDeepTDA']


class GraphDeepTDA(GraphBaseCV):
    """
    The Deep Targeted Discriminant Analysis (Deep-TDA) CV [1] based on Graph
    Neural Networks (GNN).

    Parameters
    ----------
    n_cvs: int
        Number of components of the CV.
    cutoff: float
        Cutoff radius of the basis functions. Should be the same as the cutoff
        radius used to build the graphs.
    atomic_numbers: List[int]
        The atomic numbers mapping, e.g. the `atomic_numbers` attribute of a
        `mlcolvar.graph.data.GraphDataSet` instance.
    n_cvs : int
        Number of collective variables to be trained
    target_centers : list
        Centers of the Gaussian targets
    target_sigmas : list
        Standard deviations of the Gaussian targets
    model_name: str
        Name of the GNN model.
    model_options: Dict[Any, Any]
        Model options.
    optimizer_options: Dict[Any, Any]
        Optimizer options.

    References
    ----------
    .. [1] E. Trizio and M. Parrinello,
        'From enhanced sampling to reaction profiles',
        The Journal of Physical Chemistry Letters 12, 8621– 8626 (2021).

    See also
    --------
    mlcolvar.core.loss.TDALoss
        Distance from a simple Gaussian target distribution.
    """

    def __init__(
        self,
        n_cvs: int,
        cutoff: float,
        atomic_numbers: List[int],
        target_centers: Union[List[float], List[List[float]]],
        target_sigmas: Union[List[float], List[List[float]]],
        model_name: str = 'GVPModel',
        model_options: Dict[Any, Any] = {},
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
            n_cvs, cutoff, atomic_numbers, model_name, model_options, **kwargs
        )

        # check size and type of targets
        if not isinstance(target_centers, torch.Tensor):
            target_centers = torch.tensor(
                target_centers, dtype=torch.get_default_dtype()
            )
        if not isinstance(target_sigmas, torch.Tensor):
            target_sigmas = torch.tensor(
                target_sigmas, dtype=torch.get_default_dtype()
            )

        self._n_states = target_centers.shape[0]
        if target_centers.shape != target_sigmas.shape:
            raise ValueError(
                'Size of target_centers and target_sigmas should be the same!'
            )
        if len(target_centers.shape) == 1:
            if n_cvs != 1:
                raise ValueError(
                    'Size of target_centers at dimension 1 should match the '
                    + f'number of cvs! Expected 1 found {n_cvs}'
                )
        elif len(target_centers.shape) == 2:
            if n_cvs != target_centers.shape[1]:
                raise ValueError(
                    'Size of target_centers at dimension 1 should match the '
                    + f'number of cvs! Expected {n_cvs} found '
                    + f'{target_centers.shape[1]}'
                )
        elif len(target_centers.shape) > 2:
            raise ValueError('Too much target_centers dimensions!')

        self.loss_fn = TDALoss(
            n_states=target_centers.shape[0],
            target_centers=target_centers,
            target_sigmas=target_sigmas,
        )

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
        output = self.forward(train_batch.to_dict())

        loss, loss_centers, loss_sigmas = self.loss_fn(
            output,
            train_batch.graph_labels.squeeze(),
            return_loss_terms=True
        )

        name = 'train' if self.training else 'valid'
        self.log(f'{name}_loss', loss, on_epoch=True)
        self.log(f'{name}_loss_centers', loss_centers, on_epoch=True)
        self.log(f'{name}_loss_sigmas', loss_sigmas, on_epoch=True)
        return loss

    @property
    def example_input_array(self) -> Dict[str, torch.Tensor]:
        """
        Example data.
        """
        numbers = self._model.atomic_numbers.cpu().numpy().tolist()
        positions = np.random.randn(2, len(numbers), 3)
        cell = np.identity(3, dtype=float) * 0.2
        graph_labels = np.array([[[0]], [[1]]])
        node_labels = np.array([[0]] * len(numbers))
        z_table = gdata.atomic.AtomicNumberTable.from_zs(numbers)

        config = [
            gdata.atomic.Configuration(
                atomic_numbers=numbers,
                positions=positions[i],
                cell=cell,
                pbc=[True] * 3,
                node_labels=node_labels,
                graph_labels=graph_labels[i],
            ) for i in range(2)
        ]
        dataset = gdata.create_dataset_from_configurations(
            config, z_table, 0.1, show_progress=False
        )

        loader = gdata.GraphDataModule(
            dataset,
            lengths=(1.0,),
            batch_size=10,
            shuffle=False,
        )
        loader.setup()

        return next(iter(loader.train_dataloader()))


def test_deep_tda():
    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')

    cv = GraphDeepTDA(
        2,
        0.1,
        [1, 8],
        [[-1, -1], [1, 1]],
        [[1, 1], [1, 1]],
        model_options={
            'n_bases': 6,
            'n_polynomials': 6,
            'n_layers': 2,
            'n_messages': 2,
            'n_feedforwards': 1,
            'n_scalars_node': 16,
            'n_vectors_node': 8,
            'n_scalars_edge': 16,
            'drop_rate': 0,
            'activation': 'SiLU',
        }
    )

    data = test_get_data()

    assert (
        torch.abs(
            cv(data)
            - torch.tensor([[0.6100070244145421, -0.2559670171962067]] * 6)
        ) < 1E-12
    ).all()

    assert torch.abs(
        cv.training_step(data) - torch.tensor(404.8752553674548)
    ) < 1E-12

    try:
        cv = GraphDeepTDA(2, 0.1, [1, 8], [-1, 1], [1, 1])
    except ValueError:
        pass
    else:
        raise RuntimeError

    try:
        cv = GraphDeepTDA(2, 0.1, [1, 8], [[-1, -1], [1, 1]], [1, 1])
    except ValueError:
        pass
    else:
        raise RuntimeError


if __name__ == '__main__':
    test_deep_tda()
