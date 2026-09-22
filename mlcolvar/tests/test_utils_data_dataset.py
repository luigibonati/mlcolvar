import numpy as np
import pytest
import torch

from mlcolvar.data import DictDataset
from mlcolvar.data.graph.atomic import AtomicNumberTable
from mlcolvar.data.graph.utils import create_test_graph_input
from mlcolvar.tests import data_dir


def test_dataset_print_and_repr(capsys):
    dataset = DictDataset(
        {"data": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "labels": torch.tensor([0.0, 1.0])},
        feature_names=["x", "y"],
        create_ref_idx=True,
    )

    # __repr__ branch.
    text_representation = repr(dataset)
    assert "DictDataset(" in text_representation
    assert '"data"' in text_representation

    # get_stats print path for all keys.
    stats = dataset.get_stats()
    output = capsys.readouterr().out
    assert "KEY:  data" in output
    assert "KEY:  labels" in output
    assert "KEY:  ref_idx" in output
    assert "data" in stats and "labels" in stats


def test_dataset_errors():
    with pytest.raises(TypeError):
        DictDataset(dictionary=[1, 2, 3])

    with pytest.raises(ValueError):
        DictDataset(dictionary={})

    with pytest.raises(ValueError):
        DictDataset({"data": np.ones((2, 2)), "labels": np.ones(3)})

    dataset = DictDataset({"data": torch.tensor([[1.0], [2.0]])})
    with pytest.raises(ValueError):
        dataset["new_key"] = torch.tensor([1.0])

    with pytest.raises(NotImplementedError):
        dataset[0] = {"data": torch.tensor([1.0])}
    

def test_dataset_indexing_preserves_metadata():
    dataset = DictDataset(
        {
            "data": torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [7.0, 8.0],
                ]
            ),
            "labels": torch.tensor(
                [0.0, 1.0, 2.0, 3.0]
            ),
        },
        feature_names=["x", "y"],
        metadata={"temperature": 300},
        create_ref_idx=True,
    )

    subset = dataset[1:3]

    assert isinstance(subset, DictDataset)
    assert len(subset) == 2

    torch.testing.assert_close(
        subset["data"],
        dataset["data"][1:3],
    )
    torch.testing.assert_close(
        subset["labels"],
        dataset["labels"][1:3],
    )
    torch.testing.assert_close(
        subset["ref_idx"],
        dataset["ref_idx"][1:3],
    )

    np.testing.assert_array_equal(
        subset.feature_names,
        dataset.feature_names,
    )

    assert subset.metadata == dataset.metadata
    assert subset.metadata is not dataset.metadata

    # Scalar indexing must remain compatible with DataLoader.
    sample = dataset[0]

    assert isinstance(sample, dict)

    torch.testing.assert_close(
        sample["data"],
        dataset["data"][0],
    )


@pytest.mark.parametrize(
    ("index", "expected_ids"),
    [
        (slice(1, None), [1, 2]),
        ([1], [1]),
        (np.array([1]), [1]),
        (torch.tensor([1]), [1]),
    ],
)
def test_graph_dataset_indexing(
    index,
    expected_ids,
):
    graph_list = [
        {"graph_id": 0},
        {"graph_id": 1},
        {"graph_id": 2},
    ]

    dataset = DictDataset(
        {
            "data_list": graph_list,
            "labels": torch.tensor(
                [0.0, 1.0, 2.0]
            ),
        },
        metadata={
            "cutoff": 5.0,
            "atomic_types": [1, 6, 8],
        },
        data_type="graphs",
    )

    subset = dataset[index]

    assert isinstance(subset, DictDataset)
    assert len(subset) == len(expected_ids)

    assert subset["data_list"] == [
        graph_list[i]
        for i in expected_ids
    ]

    torch.testing.assert_close(
        subset["labels"],
        torch.tensor(
            expected_ids,
            dtype=torch.float32,
        ),
    )

    assert subset.metadata == dataset.metadata
    assert subset.metadata is not dataset.metadata


def test_from_colvars():
    class CustomDataset(DictDataset):
        pass

    with data_dir() as folder:
        dataset = CustomDataset.from_colvars(
            "state_A.dat",
            folder=str(folder),
            filter_args={
                "items": ["n1h", "n2h"],
            },
            stop=5,
            verbose=False,
        )

    assert isinstance(dataset, CustomDataset)
    assert isinstance(dataset[:2], CustomDataset)
    assert len(dataset) == 5
    assert dataset["data"].shape == (5, 2)
    
    
def test_from_colvars_with_labels_and_dataframe():
    with data_dir() as folder:
        dataset, dataframe = DictDataset.from_colvars(
            file_names=[
                "state_A.dat",
                "state_B.dat",
            ],
            folder=str(folder),
            create_labels=True,
            filter_args={"regex": "n|o"},
            start=0,
            stop=5,
            return_dataframe=True,
            verbose=False,
        )

    assert isinstance(dataset, DictDataset)
    assert len(dataset) == 10
    assert len(dataframe) == 10
    assert "labels" in dataset.keys


def test_graph_from_configurations():
    configurations = create_test_graph_input(
        "configurations",
        n_samples=1,
        n_states=1,
        add_noise=False,
    )

    dataset = DictDataset.graph_from_configurations(
        config=configurations,
        atomic_numbers=AtomicNumberTable.from_zs(
            [8, 1, 1]
        ),
        cutoff=0.1,
        show_progress=False,
    )

    assert isinstance(dataset, DictDataset)
    assert len(dataset) == 1
    assert dataset.metadata["data_type"] == "graphs"


def test_graph_from_trajectories():
    with data_dir() as folder:
        dataset = DictDataset.graph_from_trajectories(
            trajectories="r.dcd",
            topologies="r.pdb",
            folder=str(folder),
            cutoff=8.0,
            load_args=[
                {
                    "start": 0,
                    "stop": 2,
                    "stride": 1,
                }
            ],
            system_selection="all and not type H",
            show_progress=False,
        )

    assert isinstance(dataset, DictDataset)
    assert len(dataset) == 2
    assert dataset.metadata["data_type"] == "graphs"