import numpy as np
import pytest
import torch

from mlcolvar.data.dataset import DictDataset


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
    

def test_dataset_slicing_returns_dictdataset():
    dataset = DictDataset(
        dictionary={
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

    # Integer indexing must remain compatible with DataLoader.
    sample = dataset[0]

    assert isinstance(sample, dict)
    assert not isinstance(sample, DictDataset)
    torch.testing.assert_close(
        sample["data"],
        dataset["data"][0],
    )
    
    
def test_graph_dataset_slicing_preserves_metadata():
    graph_list = [
        {"graph_id": 0},
        {"graph_id": 1},
        {"graph_id": 2},
    ]

    dataset = DictDataset(
        dictionary={
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

    subset = dataset[1:]

    assert isinstance(subset, DictDataset)
    assert len(subset) == 2

    assert subset.metadata["data_type"] == "graphs"
    assert subset.metadata["cutoff"] == 5.0
    assert subset.metadata["atomic_types"] == [1, 6, 8]

    assert subset["data_list"] == graph_list[1:]
    torch.testing.assert_close(
        subset["labels"],
        torch.tensor([1.0, 2.0]),
    )
    
    
def test_graph_dataset_single_element_advanced_indexing():
    graph_list = [
        {"graph_id": 0},
        {"graph_id": 1},
        {"graph_id": 2},
    ]

    dataset = DictDataset(
        dictionary={
            "data_list": graph_list,
            "labels": torch.tensor([0.0, 1.0, 2.0]),
        },
        metadata={
            "cutoff": 5.0,
            "atomic_types": [1, 6, 8],
        },
        data_type="graphs",
    )

    indices = [
        [1],
        np.array([1]),
        torch.tensor([1], dtype=torch.long),
    ]

    for index in indices:
        subset = dataset[index]

        assert isinstance(subset, DictDataset)
        assert len(subset) == 1

        assert isinstance(subset["data_list"], list)
        assert subset["data_list"] == [graph_list[1]]

        torch.testing.assert_close(
            subset["labels"],
            torch.tensor([1.0]),
        )

        assert subset.metadata["data_type"] == "graphs"
        assert subset.metadata["cutoff"] == 5.0
        