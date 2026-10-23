import torch
from torch import nn

from mlcolvar.data import DictDataset
from mlcolvar.representation import evaluate_dataset


def test_evaluate_dataset():
    model = nn.Linear(3, 2, bias=False)
    dataset = DictDataset({"data": torch.randn(4, 3)})

    output = evaluate_dataset(model, dataset, batch_size=2)

    with torch.no_grad():
        expected = model(dataset["data"])

    torch.testing.assert_close(output, expected)