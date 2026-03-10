import numpy as np
import pytest
import torch

from mlcolvar.data.dataset import DictDataset, test_DictDataset


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


def test_module_dataset_helper():
    test_DictDataset()
