import torch
import torch_geometric as tg
import numpy as np
from typing import Dict, Any, List

from mlcolvar.core.stats import TICA
from mlcolvar.core.loss import ReduceEigenvaluesLoss
from mlcolvar.graph.cvs import GraphBaseCV
from mlcolvar.graph.cvs.cv import test_get_data
from mlcolvar.graph import data as gdata
from mlcolvar.graph.utils import torch_tools

"""
The Deep time-lagged independent component analysis (Deep-TICA) CV based on
Graph Neural Networks (GNN).
"""

__all__ = ['GraphDeepTICA']


class GraphDeepTICA(GraphBaseCV):
    """
    Graph neural network-based time-lagged independent component analysis
    (Deep-TICA).

    It is a non-linear generalization of TICA in which a feature map is learned
    by a neural network optimized as to maximize the eigenvalues of the
    transfer operator, approximated by TICA. The method is described in [1]_.
    Note that from the point of view of the architecture DeepTICA is similar to
    the SRV [2]_ method.

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
    model_name: str
        Name of the GNN model.
    model_options: Dict[Any, Any]
        Model options. Note that the `n_out` key of this dict is REQUIRED,
        which stands for the dimension of the output of the network.
    optimizer_options: Dict[Any, Any]
        Optimizer options.

    References
    ----------
    .. [1] L. Bonati, G. Piccini, and M. Parrinello, "Deep learning the slow
        modes for rare events sampling." PNAS USA 118, e2113533118 (2021)
    .. [2] W. Chen, H. Sidky, and A. L. Ferguson, "Nonlinear discovery of slow
        molecular modes using state-free reversible vampnets."
        JCP 150, 214114 (2019).

    See also
    --------
    mlcolvar.core.stats.TICA
        Time Lagged Indipendent Component Analysis
    mlcolvar.core.loss.ReduceEigenvalueLoss
        Eigenvalue reduction to a scalar quantity
    mlcolvar.utils.timelagged.create_timelagged_dataset
        Create dataset of time-lagged data.
    """
    def __init__(
        self,
        n_cvs: int,
        cutoff: float,
        atomic_numbers: List[int],
        model_name: str = 'GVPModel',
        model_options: Dict[Any, Any] = {'n_out': 6},
        optimizer_options: Dict[Any, Any] = {},
        **kwargs,
    ) -> None:
        if 'n_out' not in model_options.keys():
            raise RuntimeError(
                'The `n_out` key of parameter `model_options` is required!'
            )
        model_options['drop_rate'] = 0
        n_out = model_options['n_out']

        if optimizer_options != {}:
            kwargs['optimizer_options'] = optimizer_options

        super().__init__(
            n_cvs, cutoff, atomic_numbers, model_name, model_options, **kwargs
        )

        self.loss_fn = ReduceEigenvaluesLoss(mode='sum2')

        self.tica = TICA(n_out, n_cvs)

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        """
        nn_outputs = super(GraphDeepTICA, self).forward(data)
        outputs = self.tica(nn_outputs)

        return outputs

    def training_step(
        self,
        train_batch: Dict[str, tg.data.Batch],
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute and return the training loss and record metrics.

        Parameters
        ----------
        train_batch: Tuple[Dict[str, torch_geometric.data.Batch], int, int]
            The data batch.
        """
        data_t = train_batch['dataset_1'].to_dict()
        data_lag = train_batch['dataset_2'].to_dict()

        nn_outputs_t = super(GraphDeepTICA, self).forward(data_t)
        nn_outputs_lag = super(GraphDeepTICA, self).forward(data_lag)

        eigvals, _ = self.tica.compute(
            data=[nn_outputs_t, nn_outputs_lag],
            weights=[data_t['weight'], data_lag['weight']],
            save_params=True
        )

        loss = self.loss_fn(eigvals)
        name = 'train' if self.training else 'valid'
        loss_dict = {f'{name}_loss': loss}
        eig_dict = {
            f'{name}_eigval_{i+1}': eigvals[i] for i in range(len(eigvals))
        }
        self.log_dict(dict(loss_dict, **eig_dict), on_step=True, on_epoch=True)
        return loss

    def set_regularization(self, c0_reg=1e-6) -> None:
        """
        Add identity matrix multiplied by `c0_reg` to correlation matrix C(0)
        to avoid instabilities in performin Cholesky.

        Parameters
        ----------
        c0_reg : float
            Regularization value for C_0.
        """
        self.tica.reg_C_0 = c0_reg

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
