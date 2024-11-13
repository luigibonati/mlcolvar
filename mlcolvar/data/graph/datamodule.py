import torch
import lightning
import torch_geometric as tg
import numpy as np

from typing import Sequence, Union, Optional, Tuple
from lightning.pytorch.utilities import combined_loader

from mlcolvar.data.graph import atomic 
from mlcolvar.data.graph.dataset import create_dataset_from_configurations

"""
The data module for lightning.
"""

__all__ = ['GraphDataModule', 'GraphCombinedDataModule']


class GraphDataModule(lightning.LightningDataModule):
    """
    Lightning DataModule constructed for `torch_geometric.data.Data`. This data
    module automatically splits the input graphs into training, validation,
    and (optionally) test sets.

    Parameters
    ----------
    dataset: List[torch_geometric.data.Data]
        List of graph data.
    lengths: List[int]
        Lengths of the training, validation, and (optionally) test datasets.
        This must be a list of (float) fractions summing to 1.
    batch_size : Union[int, List[int]]
        The batch size.
    random_split: bool
        Whether to randomly split train/valid/test or sequentially.
    shuffle: Union[bool, List[bool]]
        Whether to shuffle the batches in the ``DataLoader``.
    seed: int
        The random seed used to split the dataset.
    """

    def __init__(
        self,
        dataset: Sequence[tg.data.Data],
        lengths: Sequence = (0.8, 0.2),
        batch_size: Union[int, Sequence] = None,
        random_split: bool = True,
        shuffle: Union[bool, Sequence] = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self._dataset = dataset
        self._lengths = lengths
        self._n_total = len(dataset)

        assert self._n_total > 0
        assert len(lengths) in [1, 2, 3]
        assert np.abs(1 - sum(lengths)) < 1e-12

        if self._n_total == 1:
            self._n_train = 1
        else:
            self._n_train = int(lengths[0] * self._n_total)
        if len(lengths) == 3:
            self._n_validation = int(lengths[1] * self._n_total)
            self._n_test = self._n_total - self._n_train - self._n_validation
        elif len(lengths) == 2:
            self._n_validation = self._n_total - self._n_train
            self._n_test = 0

        indices = list(range(self._n_total))
        if random_split:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        indices_train = indices[0:self._n_train]
        if len(lengths) == 3:
            indices_validation = indices[
                self._n_train:(self._n_train + self._n_validation)
            ]
            indices_test = indices[
                (self._n_train + self._n_validation):self._n_total
            ]
        elif len(lengths) == 2:
            indices_validation = indices[
                self._n_train:(self._n_train + self._n_validation)
            ]
            indices_test = []
        else:
            indices_validation = []
            indices_test = []
        self._dataset_indices = [
            indices_train,
            indices_validation,
            indices_test,
        ]

        # Make sure batch_size and shuffle are lists.
        if batch_size is None:
            batch_size = len(dataset)
        if isinstance(batch_size, int):
            self.batch_size = [batch_size for _ in lengths]
        else:
            self.batch_size = batch_size
        if isinstance(shuffle, bool):
            self.shuffle = [shuffle for _ in lengths]
        else:
            self.shuffle = shuffle

        # This is initialized in setup().
        self._dataset_split = None

        # dataloaders
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the datasets.
        """
        if self._dataset_split is None:
            dataset_split = [
                [self._dataset[j] for j in i] for i in self._dataset_indices
            ]
            self._dataset_split = dataset_split

    def train_dataloader(self) -> tg.loader.DataLoader:
        """
        Return training dataloader.
        """
        self._check_setup()
        if self.train_loader is None:
            self.train_loader = tg.loader.DataLoader(
                self._dataset_split[0],
                batch_size=self.batch_size[0],
                shuffle=self.shuffle[0],
            )
        return self.train_loader

    def val_dataloader(self) -> tg.loader.DataLoader:
        """
        Return validation dataloader.
        """
        self._check_setup()
        if len(self._dataset_indices[1]) == 0:
            raise NotImplementedError(
                'Validation dataset not available, you need to pass two '
                + 'lengths to datamodule.'
            )
        if self.valid_loader is None:
            self.valid_loader = tg.loader.DataLoader(
                self._dataset_split[1],
                batch_size=self.batch_size[1],
                shuffle=self.shuffle[1],
            )
        return self.valid_loader

    def test_dataloader(self) -> tg.loader.DataLoader:
        """
        Return test dataloader.
        """
        self._check_setup()
        if len(self._dataset_indices[2]) == 0:
            raise NotImplementedError(
                'Test dataset not available, you need to pass three '
                + 'lengths to datamodule.'
            )
        if self.test_loader is None:
            self.test_loader = tg.loader.DataLoader(
                self._dataset_split[2],
                batch_size=self.batch_size[2],
                shuffle=self.shuffle[2],
            )
        return self.test_loader

    def predict_dataloader(self) -> tg.loader.DataLoader:
        """
        Return predict dataloader.
        """
        raise NotImplementedError()

    def teardown(self, stage: str) -> None:
        pass

    # def __repr__(self) -> str:
    #     result = ''
    #     n_digits = len(str(self._n_total))
    #     data_string_1 = '[ \033[32m{{:{:d}d}}\033[0m\033[36m 󰡷 \033[0m'
    #     data_string_2 = '| \033[32m{{:{:d}d}}\033[0m\033[36m  \033[0m'
    #     shuffle_string_1 = '|\033[36m  \033[0m ]'
    #     shuffle_string_2 = '|\033[36m  \033[0m ]'

    #     prefix = '\033[1m\033[34m  BASEDATA  \033[0m: '
    #     result += (
    #         prefix + self._dataset.__repr__().split('GRAPHDATASET ')[1] + '\n'
    #     )
    #     prefix = '\033[1m\033[34m  TRAINING  \033[0m: '
    #     string = prefix + data_string_1.format(n_digits)
    #     result += string.format(
    #         self._n_train, self._n_train / self._n_total * 100
    #     )
    #     string = data_string_2.format(n_digits)
    #     result += string.format(self.batch_size[0])
    #     if self.shuffle[0]:
    #         result += shuffle_string_1
    #     else:
    #         result += shuffle_string_2

    #     if self._n_validation > 0:
    #         result += '\n'
    #         prefix = '\033[1m\033[34m VALIDATION \033[0m: '
    #         string = prefix + data_string_1.format(n_digits)
    #         result += string.format(
    #             self._n_validation, self._n_validation / self._n_total * 100
    #         )
    #         string = data_string_2.format(n_digits)
    #         result += string.format(self.batch_size[1])
    #         if self.shuffle[1]:
    #             result += shuffle_string_1
    #         else:
    #             result += shuffle_string_2

    #     if self._n_test > 0:
    #         result += '\n'
    #         prefix = '\033[1m\033[34m    TEST    \033[0m: '
    #         string = prefix + data_string_1.format(n_digits)
    #         result += string.format(
    #             self._n_test, self._n_test / self._n_total * 100
    #         )
    #         string = data_string_2.format(n_digits)
    #         result += string.format(self.batch_size[2])
    #         if self.shuffle[2]:
    #             result += shuffle_string_1
    #         else:
    #             result += shuffle_string_2
    #     return result

    def __repr__(self) -> str:
        string = f"DictModule(dataset -> {self._dataset.__repr__()}"
        string += f",\n\t\t     train_loader -> DictLoader(length={self._lengths[0]}, batch_size={self.batch_size[0]}, shuffle={self.shuffle[0]})"
        if len(self._lengths) >= 2:
            string += f",\n\t\t     valid_loader -> DictLoader(length={self._lengths[1]}, batch_size={self.batch_size[1]}, shuffle={self.shuffle[1]})"
        if len(self._lengths) >= 3:
            string += f",\n\t\t\ttest_loader =DictLoader(length={self._lengths[2]}, batch_size={self.batch_size[2]}, shuffle={self.shuffle[2]})"
        string += f")"
        return string

    def _check_setup(self) -> None:
        """
        Raise an error if setup() has not been called.
        """
        if self._dataset_split is None:
            raise AttributeError(
                'The datamodule has not been set up yet. To get the '
                + 'dataloaders outside a Lightning trainer please call '
                + '.setup() first.'
            )


class GraphCombinedDataModule(lightning.LightningDataModule):
    """
    Lightning DataModule constructed for `torch_geometric.data.Data`. This data
    module automatically splits the input graphs into training, validation,
    and (optionally) test sets.
    Being differnet from `GraphDataModule`, this class takes two different
    datasets as input, and evaluates them at the same time during the training.

    Parameters
    ----------
    datasets: Tuple[
        List[torch_geometric.data.Data], List[torch_geometric.data.Data]
    ]
        Lists of graph data.
    lengths: List[int]
        Lengths of the training, validation, and (optionally) test datasets.
        This must be a list of (float) fractions summing to 1.
    batch_size : Union[int, List[int]]
        The batch size.
    random_split: bool
        Whether to randomly split train/valid/test or sequentially.
    seed: int
        The random seed used to split the dataset.
    """

    def __init__(
        self,
        datasets: Tuple[Sequence[tg.data.Data], Sequence[tg.data.Data]],
        lengths: Sequence = (0.8, 0.2),
        batch_size: Union[int, Sequence] = None,
        random_split: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self._datasets = datasets
        self._lengths = lengths
        self._n_total = len(datasets[0])

        assert self._n_total > 0
        assert self._n_total == len(datasets[1])
        assert len(lengths) in [1, 2, 3]
        assert np.abs(1 - sum(lengths)) < 1e-12

        if self._n_total == 1:
            self._n_train = 1
        else:
            self._n_train = int(lengths[0] * self._n_total)
        if len(lengths) == 3:
            self._n_validation = int(lengths[1] * self._n_total)
            self._n_test = self._n_total - self._n_train - self._n_validation
        elif len(lengths) == 2:
            self._n_validation = self._n_total - self._n_train
            self._n_test = 0

        indices = list(range(self._n_total))
        if random_split:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        indices_train = indices[0:self._n_train]
        if len(lengths) == 3:
            indices_validation = indices[
                self._n_train:(self._n_train + self._n_validation)
            ]
            indices_test = indices[
                (self._n_train + self._n_validation):self._n_total
            ]
        elif len(lengths) == 2:
            indices_validation = indices[
                self._n_train:(self._n_train + self._n_validation)
            ]
            indices_test = []
        else:
            indices_validation = []
            indices_test = []
        self._dataset_indices = [
            indices_train,
            indices_validation,
            indices_test,
        ]

        # Make sure batch_size and shuffle are lists.
        if batch_size is None:
            batch_size = self._n_total
        if isinstance(batch_size, int):
            self.batch_size = [batch_size for _ in lengths]
        else:
            self.batch_size = batch_size

        # This is initialized in setup().
        self._dataset_splits = None

        # dataloaders
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the datasets.
        """
        if self._dataset_splits is None:
            dataset_splits = [None] * 2
            dataset_splits[0] = [
                [self._datasets[0][j] for j in i]
                for i in self._dataset_indices
            ]
            dataset_splits[1] = [
                [self._datasets[1][j] for j in i]
                for i in self._dataset_indices
            ]
            self._dataset_splits = dataset_splits

    def train_dataloader(self) -> tg.loader.DataLoader:
        """
        Return training dataloader.
        """
        self._check_setup()
        if self.train_loader is None:
            train_loader_1 = tg.loader.DataLoader(
                self._dataset_splits[0][0],
                batch_size=self.batch_size[0],
                shuffle=False,
            )
            train_loader_2 = tg.loader.DataLoader(
                self._dataset_splits[1][0],
                batch_size=self.batch_size[0],
                shuffle=False,
            )
            self.train_loader = combined_loader.CombinedLoader(
                {'dataset_1': train_loader_1, 'dataset_2': train_loader_2},
                mode='min_size'
            )
        return self.train_loader

    def val_dataloader(self) -> tg.loader.DataLoader:
        """
        Return validation dataloader.
        """
        self._check_setup()
        if len(self._dataset_indices[1]) == 0:
            raise NotImplementedError(
                'Validation dataset not available, you need to pass two '
                + 'lengths to datamodule.'
            )
        if self.valid_loader is None:
            valid_loader_1 = tg.loader.DataLoader(
                self._dataset_splits[0][1],
                batch_size=self.batch_size[1],
                shuffle=False,
            )
            valid_loader_2 = tg.loader.DataLoader(
                self._dataset_splits[1][1],
                batch_size=self.batch_size[1],
                shuffle=False,
            )
            self.valid_loader = combined_loader.CombinedLoader(
                {'dataset_1': valid_loader_1, 'dataset_2': valid_loader_2},
                mode='min_size'
            )
        return self.valid_loader

    def test_dataloader(self) -> tg.loader.DataLoader:
        """
        Return test dataloader.
        """
        self._check_setup()
        if len(self._dataset_indices[2]) == 0:
            raise NotImplementedError(
                'Test dataset not available, you need to pass three '
                + 'lengths to datamodule.'
            )
        if self.test_loader is None:
            test_loader_1 = tg.loader.DataLoader(
                self._dataset_splits[0][2],
                batch_size=self.batch_size[2],
                shuffle=False,
            )
            test_loader_2 = tg.loader.DataLoader(
                self._dataset_splits[1][2],
                batch_size=self.batch_size[2],
                shuffle=False,
            )
            self.test_loader = combined_loader.CombinedLoader(
                {'dataset_1': test_loader_1, 'dataset_2': test_loader_2},
                mode='min_size'
            )
        return self.test_loader

    def predict_dataloader(self) -> tg.loader.DataLoader:
        """
        Return predict dataloader.
        """
        raise NotImplementedError()

    def teardown(self, stage: str) -> None:
        pass

    def __repr__(self) -> str:
        result = ''
        n_digits = len(str(self._n_total))
        data_string_1 = '[ \033[32m{{:{:d}d}}\033[0m\033[36m 󰡷 \033[0m'
        data_string_2 = '| \033[32m{{:{:d}d}}\033[0m\033[36m  \033[0m'
        shuffle_string = '|\033[36m  \033[0m ]'

        prefix = '\033[1m\033[34m  BASEDATA  \033[0m: '
        result += (
            prefix + self._datasets[0].__repr__().split('GRAPHDATASET ')[1]
            + '\n'
            + prefix + self._datasets[1].__repr__().split('GRAPHDATASET ')[1]
            + '\n'
        )
        prefix = '\033[1m\033[34m  TRAINING  \033[0m: '
        string = prefix + data_string_1.format(n_digits)
        result += string.format(
            self._n_train * 2, self._n_train / self._n_total * 100
        )
        string = data_string_2.format(n_digits)
        result += string.format(self.batch_size[0])
        result += shuffle_string

        if self._n_validation > 0:
            result += '\n'
            prefix = '\033[1m\033[34m VALIDATION \033[0m: '
            string = prefix + data_string_1.format(n_digits)
            result += string.format(
                self._n_validation * 2,
                self._n_validation / self._n_total * 100
            )
            string = data_string_2.format(n_digits)
            result += string.format(self.batch_size[1])
            result += shuffle_string

        if self._n_test > 0:
            result += '\n'
            prefix = '\033[1m\033[34m    TEST    \033[0m: '
            string = prefix + data_string_1.format(n_digits)
            result += string.format(
                self._n_test * 2, self._n_test / self._n_total * 100
            )
            string = data_string_2.format(n_digits)
            result += string.format(self.batch_size[2])
            result += shuffle_string
        return result

    def _check_setup(self) -> None:
        """
        Raise an error if setup() has not been called.
        """
        if self._dataset_splits is None:
            raise AttributeError(
                'The datamodule has not been set up yet. To get the '
                + 'dataloaders outside a Lightning trainer please call '
                + '.setup() first.'
            )


def test_datamodule() -> None:
    numbers = [8, 1, 1]
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]], dtype=float
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[1]])
    node_labels = np.array([[0], [1], [1]])
    z_table = atomic.AtomicNumberTable.from_zs(numbers)

    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
    )
    dataset = create_dataset_from_configurations(
        [config] * 10, z_table, 0.1, show_progress=False
    )
    for i, d in enumerate(dataset):
        d['graph_labels'][0][0] = i

    loader = GraphDataModule(
        dataset,
        lengths=(1.0,),
        batch_size=10,
        shuffle=True,
        seed=1
    )
    loader.setup()
    assert len(loader._dataset_indices[0]) == 10
    assert len(loader._dataset_indices[1]) == 0
    assert len(loader._dataset_indices[2]) == 0

    loader = GraphDataModule(
        dataset,
        batch_size=10,
        shuffle=True,
        seed=1
    )
    loader.setup()
    assert len(loader._dataset_indices[0]) == 8
    assert len(loader._dataset_indices[1]) == 2
    assert len(loader._dataset_indices[2]) == 0

    loader = GraphDataModule(
        dataset,
        lengths=(0.6, 0.3, 0.1),
        batch_size=10,
        shuffle=False,
        seed=1
    )
    loader.setup()
    assert loader._dataset_indices == [[8, 4, 7, 0, 1, 2], [5, 9, 6], [3]]

    data_dict = next(iter(loader.train_dataloader())).to_dict()
    assert data_dict['edge_index'].shape == (2, 36)
    assert (
        data_dict['graph_labels'] == torch.tensor(
            [[8], [4], [7], [0], [1], [2]]
        )
    ).all()

    data_dict = next(iter(loader.val_dataloader())).to_dict()
    assert (
        data_dict['edge_index'] == torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
            [2, 1, 0, 2, 1, 0, 5, 4, 3, 5, 4, 3, 8, 7, 6, 8, 7, 6]
        ])
    ).all()
    assert (
        data_dict['batch'] == torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ).all()
    assert (data_dict['graph_labels'] == torch.tensor([[5], [9], [6]])).all()
    assert (data_dict['ptr'] == torch.tensor([0, 3, 6, 9])).all()

    data_dict = next(iter(loader.test_dataloader())).to_dict()
    assert (data_dict['graph_labels'] == torch.tensor([[3]])).all()
    assert (
        data_dict['edge_index'] == torch.tensor(
            [[0, 0, 1, 1, 2, 2], [2, 1, 0, 2, 1, 0]]
        )
    ).all()
    assert (
        data_dict['shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.0],
        ])
    ).all()
    assert (
        data_dict['unit_shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
    ).all()
    assert (
        data_dict['positions'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.07, 0.07, 0.0],
            [0.07, -0.07, 0.0],
        ])
    ).all()
    assert (
        data_dict['cell'] == torch.tensor([
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ])
    ).all()
    assert (
        data_dict['node_attrs'] == torch.tensor([
            [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]
        ])
    ).all()

    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
        system=[0],
        environment=[1, 2]
    )
    dataset = create_dataset_from_configurations(
        [config] * 10, z_table, 0.1, show_progress=False
    )

    loader = GraphDataModule(
        dataset,
        lengths=(0.6, 0.3, 0.1),
        batch_size=10,
        shuffle=False,
        seed=1
    )

    loader.setup()
    assert loader._dataset_indices == [[8, 4, 7, 0, 1, 2], [5, 9, 6], [3]]

    data_dict = next(iter(loader.val_dataloader())).to_dict()
    assert (
        data_dict['edge_index'] == torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
            [2, 1, 0, 2, 1, 0, 5, 4, 3, 5, 4, 3, 8, 7, 6, 8, 7, 6]
        ])
    ).all()

    data_dict = next(iter(loader.test_dataloader())).to_dict()
    assert (
        data_dict['edge_index'] == torch.tensor(
            [[0, 0, 1, 1, 2, 2], [2, 1, 0, 2, 1, 0]]
        )
    ).all()


def test_combined_datamodule() -> None:
    numbers = [8, 1, 1]
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]], dtype=float
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[1]])
    node_labels = np.array([[0], [1], [1]])
    z_table = atomic.AtomicNumberTable.from_zs(numbers)

    config = atomic.Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels,
        graph_labels=graph_labels,
    )
    dataset = create_dataset_from_configurations(
        [config] * 10, z_table, 0.1, show_progress=False
    )
    for i, d in enumerate(dataset):
        d['graph_labels'][0][0] = i

    loader = GraphCombinedDataModule(
        (dataset, dataset),
        lengths=(1.0,),
        batch_size=10,
        seed=1
    )
    loader.setup()
    assert len(loader._dataset_indices[0]) == 10
    assert len(loader._dataset_indices[1]) == 0
    assert len(loader._dataset_indices[2]) == 0

    loader = GraphCombinedDataModule(
        (dataset, dataset),
        lengths=(0.8, 0.2),
        batch_size=10,
        seed=1
    )
    loader.setup()
    assert len(loader._dataset_indices[0]) == 8
    assert len(loader._dataset_indices[1]) == 2
    assert len(loader._dataset_indices[2]) == 0

    loader = GraphCombinedDataModule(
        (dataset, dataset),
        lengths=(0.6, 0.3, 0.1),
        batch_size=10,
        seed=1
    )
    loader.setup()
    assert loader._dataset_indices == [[8, 4, 7, 0, 1, 2], [5, 9, 6], [3]]

    batch = next(iter(loader.train_dataloader()))[0]
    data_dict_1 = batch['dataset_1'].to_dict()
    data_dict_2 = batch['dataset_2'].to_dict()

    assert data_dict_1['edge_index'].shape == (2, 36)
    assert data_dict_2['edge_index'].shape == (2, 36)
    assert (
        data_dict_1['graph_labels'] == torch.tensor(
            [[8], [4], [7], [0], [1], [2]]
        )
    ).all()
    assert (
        data_dict_2['graph_labels'] == torch.tensor(
            [[8], [4], [7], [0], [1], [2]]
        )
    ).all()

    batch = next(iter(loader.val_dataloader()))[0]
    data_dict_1 = batch['dataset_1'].to_dict()
    data_dict_2 = batch['dataset_2'].to_dict()
    assert (
        data_dict_1['edge_index'] == torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
            [2, 1, 0, 2, 1, 0, 5, 4, 3, 5, 4, 3, 8, 7, 6, 8, 7, 6]
        ])
    ).all()
    assert (
        data_dict_2['edge_index'] == torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
            [2, 1, 0, 2, 1, 0, 5, 4, 3, 5, 4, 3, 8, 7, 6, 8, 7, 6]
        ])
    ).all()
    assert (
        data_dict_1['batch'] == torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ).all()
    assert (
        data_dict_2['batch'] == torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ).all()
    assert (data_dict_1['graph_labels'] == torch.tensor([[5], [9], [6]])).all()
    assert (data_dict_2['graph_labels'] == torch.tensor([[5], [9], [6]])).all()
    assert (data_dict_1['ptr'] == torch.tensor([0, 3, 6, 9])).all()
    assert (data_dict_2['ptr'] == torch.tensor([0, 3, 6, 9])).all()

    batch = next(iter(loader.test_dataloader()))[0]
    data_dict_1 = batch['dataset_1'].to_dict()
    data_dict_2 = batch['dataset_2'].to_dict()
    assert (data_dict_1['graph_labels'] == torch.tensor([[3]])).all()
    assert (data_dict_2['graph_labels'] == torch.tensor([[3]])).all()
    assert (
        data_dict_1['edge_index'] == torch.tensor(
            [[0, 0, 1, 1, 2, 2], [2, 1, 0, 2, 1, 0]]
        )
    ).all()
    assert (
        data_dict_2['edge_index'] == torch.tensor(
            [[0, 0, 1, 1, 2, 2], [2, 1, 0, 2, 1, 0]]
        )
    ).all()
    assert (
        data_dict_1['shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.0],
        ])
    ).all()
    assert (
        data_dict_2['shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.0],
        ])
    ).all()
    assert (
        data_dict_1['unit_shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
    ).all()
    assert (
        data_dict_2['unit_shifts'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
    ).all()
    assert (
        data_dict_1['positions'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.07, 0.07, 0.0],
            [0.07, -0.07, 0.0],
        ])
    ).all()
    assert (
        data_dict_2['positions'] == torch.tensor([
            [0.0, 0.0, 0.0],
            [0.07, 0.07, 0.0],
            [0.07, -0.07, 0.0],
        ])
    ).all()
    assert (
        data_dict_1['cell'] == torch.tensor([
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ])
    ).all()
    assert (
        data_dict_2['cell'] == torch.tensor([
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ])
    ).all()
    assert (
        data_dict_1['node_attrs'] == torch.tensor([
            [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]
        ])
    ).all()
    assert (
        data_dict_2['node_attrs'] == torch.tensor([
            [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]
        ])
    ).all()


if __name__ == '__main__':
    test_datamodule()
    test_combined_datamodule()
