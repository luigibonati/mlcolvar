import torch
import lightning
import numpy as np
import torch_geometric as tg
from typing import Dict, Any, List, Union, Tuple

from mlcolvar.graph import data as gdata
from mlcolvar.graph.core.nn import models

"""
Base collective variable class for Graph Neural Networks.
"""

__all__ = ['GraphBaseCV']


class GraphBaseCV(lightning.LightningModule):
    """
    Base collective variable class for Graph Neural Networks.

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
    model_name: str
        Name of the GNN model.
    model_options: Dict[Any, Any]
        Model options.
    optimizer_options: Dict[Any, Any]
        Optimizer options.
    """

    def __init__(
        self,
        n_cvs: int,
        cutoff: float,
        atomic_numbers: List[int],
        model_name: str = 'GVPModel',
        model_options: Dict[Any, Any] = {
            'n_bases': 8,
            'n_polynomials': 6,
            'n_layers': 2,
            'n_messages': 1,
            'n_feedforwards': 1,
            'n_scalars_node': 16,
            'n_vectors_node': 16,
            'n_scalars_edge': 16,
            'drop_rate': 0.2,
            'activation': 'SiLU',
        },
        optimizer_options: Dict[Any, Any] = {
            'optimizer': {'lr': 1E-3, 'weight_decay': 1E-4},
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ExponentialLR,
                'gamma': 0.9999
            }
        },
        *args,
        **kwargs,
    ) -> None:
        """
        Base CV class options.
        """
        super().__init__(*args, **kwargs)

        for key in ['n_out', 'cutoff', 'atomic_numbers']:
            model_options.pop(key, None)

        if not hasattr(models, model_name):
            raise RuntimeError(f'Unknown model: {model_name}')
        self._model = eval(f'models.{model_name}')(
            n_out=n_cvs,
            cutoff=cutoff,
            atomic_numbers=atomic_numbers,
            **model_options
        )

        self._optimizer_name = 'Adam'
        self.optimizer_kwargs = {}
        self.lr_scheduler_kwargs = {}
        self._parse_optimizer(optimizer_options)

        self.save_hyperparameters(ignore=['n_cvs', 'cutoff', 'atomic_numbers'])

    def __setattr__(self, key, value) -> None:
        # PyTorch overrides __setattr__ to raise a TypeError when you try to
        # assignan attribute that is a Module to avoid substituting the model's
        # component by mistake. This means we can't simply assign to loss_fn a
        # lambda function after it's been assigned a Module, but we need to
        # delete the Module first.
        #    https://github.com/pytorch/pytorch/issues/51896
        #    https://stackoverflow.com/questions/61116433
        try:
            super().__setattr__(key, value)
        except TypeError as e:
            # We make an exception only for loss_fn.
            if (key == 'loss_fn') and ('cannot assign' in str(e)):
                del self.loss_fn
                super().__setattr__(key, value)

    def _parse_optimizer(self, options: dict) -> None:
        """
        Parse optimizer options.

        Parameters
        ----------
        options: Dict[Any, Any]
            The options
        """
        optimizer_kwargs = options.get('optimizer')
        if optimizer_kwargs is not None:
            self.optimizer_kwargs.update(optimizer_kwargs)

        lr_scheduler_kwargs = options.get('lr_scheduler')
        if lr_scheduler_kwargs is not None:
            self.lr_scheduler_kwargs.update(lr_scheduler_kwargs)

    def forward(self, data: tg.data.Batch) -> torch.Tensor:
        """
        The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        """
        return self._model(data.to_dict())

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        """
        Equal to training step if not overridden. Different behaviors for
        train/valid step can be enforced in `training_step` based on the
        `self.training` variable.
        """
        return self.training_step(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        """
        Equal to training step if not overridden. Different behaviors for
        train/valid step can be enforced in `training_step` based on the
        `self.training` variable.
        """
        return self.training_step(*args, **kwargs)

    @property
    def optimizer_name(self) -> str:
        """
        Optimizer name. Options can be set using optimizer_kwargs. Actual
        optimizer will be return during training from configure_optimizer
        function.
        """
        return self._optimizer_name

    @optimizer_name.setter
    def optimizer_name(self, optimizer_name: str) -> None:
        if not hasattr(torch.optim, optimizer_name):
            raise AttributeError(
                f'torch.optim does not have a {optimizer_name} optimizer.'
            )
        self._optimizer_name = optimizer_name

    def configure_optimizers(self) -> Union[
        torch.optim.Optimizer,
        Tuple[
            List[torch.optim.Optimizer],
            List[torch.optim.lr_scheduler.LRScheduler]
        ]
    ]:
        """
        Initialize the optimizer based on `self._optimizer_name` and
        `self.optimizer_kwargs`.
        """

        optimizer = getattr(torch.optim, self._optimizer_name)(
            self.parameters(), **self.optimizer_kwargs
        )

        if self.lr_scheduler_kwargs:
            scheduler_cls = self.lr_scheduler_kwargs['scheduler']
            scheduler_kwargs = {
                k: v for k, v in self.lr_scheduler_kwargs.items()
                if k != 'scheduler'
            }
            lr_scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    @property
    def n_cvs(self) -> int:
        """
        Number of components of the CV.
        """
        return self._model.n_out.item()

    @property
    def cutoff(self) -> float:
        """
        Number of components of the CV.
        """
        return self._model.cutoff.item()


def test_get_data(receivers: List[int] = [0, 1, 2]) -> tg.data.Batch:
    # TODO: This is not a real test, but a helper function for other tests.
    # Maybe should change its name.

    numbers = [8, 1, 1]
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
            [[0.0, 0.0, 0.0], [-0.07, 0.07, 0.0], [0.07, 0.07, 0.0]],
            [[0.0, 0.0, 0.0], [0.07, -0.07, 0.0], [0.07, 0.07, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, -0.07, 0.07], [0.0, 0.07, 0.07]],
            [[0.0, 0.0, 0.0], [0.07, 0.0, 0.07], [-0.07, 0.0, 0.07]],
            [[0.1, 0.0, 1.1], [0.17, 0.07, 1.1], [0.17, -0.07, 1.1]],
        ],
        dtype=np.float64
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[[0]], [[1]]] * 3)
    node_labels = np.array([[0], [1], [1]])
    z_table = gdata.atomic.AtomicNumberTable.from_zs(numbers)

    config = [
        gdata.atomic.Configuration(
            atomic_numbers=numbers,
            positions=positions[i],
            cell=cell,
            pbc=[True] * 3,
            node_labels=node_labels,
            graph_labels=graph_labels[i],
            edge_receivers=receivers,
        ) for i in range(0, 6)
    ]
    dataset = gdata.create_dataset_from_configurations(config, z_table, 0.1)

    loader = gdata.GraphDataModule(
        dataset,
        lengths=(1.0,),
        batch_size=10,
        shuffle=False,
    )
    loader.setup()

    return next(iter(loader.train_dataloader()))


def test_base_cv() -> None:
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    cv = GraphBaseCV(2, 0.1, [1, 2, 3])

    assert cv.n_cvs == 2
    assert (cv.cutoff - 0.1) < 1E-12
    assert (cv._model.atomic_numbers == torch.tensor([1, 2, 3])).all()

    assert cv.optimizer_name == 'Adam'
    objects = cv.configure_optimizers()
    assert isinstance(objects[0][0], torch.optim.Adam)
    assert isinstance(objects[1][0], torch.optim.lr_scheduler.ExponentialLR)
    assert objects[0][0].param_groups[0]['weight_decay'] == 1E-4
    assert objects[0][0].param_groups[0]['lr'] == 1E-3
    assert objects[1][0].gamma == 0.9999

    cv.optimizer_name = 'SGD'
    cv.optimizer_kwargs = {'lr': 2E-3, 'weight_decay': 1E-4}
    objects = cv.configure_optimizers()
    assert isinstance(objects[0][0], torch.optim.SGD)
    assert objects[0][0].param_groups[0]['weight_decay'] == 1E-4
    assert objects[0][0].param_groups[0]['lr'] == 2E-3

    cv.lr_scheduler_kwargs = {
        'scheduler': torch.optim.lr_scheduler.StepLR,
        'gamma': 0.999,
        'step_size': 1
    }
    objects = cv.configure_optimizers()
    assert isinstance(objects[0][0], torch.optim.SGD)
    assert isinstance(objects[1][0], torch.optim.lr_scheduler.StepLR)
    assert objects[0][0].param_groups[0]['weight_decay'] == 1E-4
    assert objects[0][0].param_groups[0]['lr'] == 2E-3
    assert objects[1][0].gamma == 0.999

    cv = GraphBaseCV(
        2,
        0.1,
        [1, 2, 3],
        optimizer_options={
            'optimizer': {'lr': 2E-3, 'weight_decay': 1E-4},
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ExponentialLR,
                'gamma': 0.9999
            }
        }
    )

    objects = cv.configure_optimizers()
    assert isinstance(objects[0][0], torch.optim.Adam)
    assert isinstance(objects[1][0], torch.optim.lr_scheduler.ExponentialLR)
    assert objects[0][0].param_groups[0]['weight_decay'] == 1E-4
    assert objects[0][0].param_groups[0]['lr'] == 2E-3
    assert objects[1][0].gamma == 0.9999

    torch.set_default_dtype(dtype)


if __name__ == '__main__':
    test_base_cv()
