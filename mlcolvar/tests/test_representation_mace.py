import pytest
import torch
from torch import nn

from mlcolvar.core import FeedForward
from mlcolvar.cvs import RegressionCV
from mlcolvar.representation import MACERepresentation


DTYPE = torch.float64


class DummyMACE(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor([1, 8]))
        self.register_buffer("r_max", torch.tensor(5.0, dtype=DTYPE))
        self.register_buffer("num_interactions", torch.tensor(2))
        self.weight = nn.Parameter(torch.tensor(1.0, dtype=DTYPE))

    def forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = False,
    ) -> dict[str, torch.Tensor]:
        return {"node_feats": data["node_feats"] * self.weight}


def make_data():
    features = torch.zeros(4, 10, dtype=DTYPE)
    features[:, :2] = torch.tensor(
        [[1., 2.], [3., 4.], [2., 4.], [4., 6.]], dtype=DTYPE
    )
    features[:, 8:10] = torch.tensor(
        [[5., 6.], [7., 8.], [6., 8.], [8., 10.]], dtype=DTYPE
    )
    return {
        "node_feats": features,
        "batch": torch.tensor([0, 0, 1, 1]),
        "ptr": torch.tensor([0, 2, 4]),
    }


def make_representation(pooling=None):
    return MACERepresentation(
        DummyMACE(),
        pooling_operation=pooling,
        num_layers=2,
        num_features=2,
        l_max=1,
    )


def make_cv():
    representation = make_representation("mean")
    return RegressionCV(
        model=FeedForward([representation.out_features, 1]),
        preprocessing=representation,
    ).to(DTYPE)


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [
        (
            None,
            torch.tensor(
                [
                    [1., 2., 5., 6.],
                    [3., 4., 7., 8.],
                    [2., 4., 6., 8.],
                    [4., 6., 8., 10.],
                ],
                dtype=DTYPE,
            ),
        ),
        (
            "mean",
            torch.tensor(
                [[2., 3., 6., 7.], [3., 5., 7., 9.]],
                dtype=DTYPE,
            ),
        ),
    ],
)
def test_mace_representation(pooling, expected):
    representation = make_representation(pooling)
    output = representation(make_data())
    torch.testing.assert_close(output, expected)
    assert representation.pooling_operation == pooling
    assert representation.out_features == 4


def test_mace_gradients_and_cv():
    representation = make_representation("mean")
    data = make_data()
    data["node_feats"].requires_grad_(True)
    representation(data).sum().backward()

    assert data["node_feats"].grad is not None
    assert all(
        not parameter.requires_grad
        for parameter in representation.parameters()
    )

    model = RegressionCV(
        model=FeedForward([representation.out_features, 1]),
        preprocessing=representation,
    ).to(DTYPE)
    assert model(make_data()).shape == (2, 1)


def test_mace_torchscript(tmp_path):
    pytest.importorskip("e3nn", reason="MACE TorchScript export requires e3nn.")
    model = make_cv().eval()
    model.preprocessing.prepare_for_torchscript()
    path = tmp_path / "model.ptc"
    model.to_torchscript(
        file_path=path,
        method="trace",
        example_inputs=make_data(),
    )
    loaded = torch.jit.load(str(path)).eval()
    assert path.exists()
    assert loaded(make_data()).shape == (2, 1)


def test_invalid_node_features():
    representation = make_representation()
    data = make_data()
    data["node_feats"] = torch.zeros(4, 9, dtype=DTYPE)
    with pytest.raises(
        RuntimeError,
        match="incompatible with the configured descriptor layout",
    ):
        representation(data)