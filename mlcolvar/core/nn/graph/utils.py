import torch
import torch_geometric
import numpy as np

from mlcolvar.data.graph import atomic, create_dataset_from_configurations
from mlcolvar.data import DictModule


def _test_get_data() -> torch_geometric.data.Batch:
    # TODO: This is not a real test, but a helper function for other tests.
    # Maybe should change its name.
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

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
    graph_labels = np.array([[1]])
    node_labels = np.array([[0], [1], [1]])
    z_table = atomic.AtomicNumberTable.from_zs(numbers)

    config = [
        atomic.Configuration(
            atomic_numbers=numbers,
            positions=p,
            cell=cell,
            pbc=[True] * 3,
            node_labels=node_labels,
            graph_labels=graph_labels,
        ) for p in positions
    ]
    dataset = create_dataset_from_configurations(
        config, z_table, 0.1, show_progress=False
    )

    datamodule = DictModule(
        dataset,
        lengths=(1.0,),
        batch_size=10,
        shuffle=False,
    )
    datamodule.setup()

    return next(iter(datamodule.train_dataloader()))['data_list']