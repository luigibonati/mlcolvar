from typing import Optional

import torch
from torch import nn

from mlcolvar.core import FeedForward
from mlcolvar.cvs import RegressionCV
from mlcolvar.representation import (
    MLColvarRepresentation,
    VectorRepresentation,
    export_representation_torchscript,
)


class AddCellPreprocessing(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + 1.0
        if cell is not None:
            x = x + cell.reshape(())
        return x


class ScaleNormalization(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return 2.0 * x


class DummyDescriptorCV(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_features = 3
        self.out_features = 1
        self.preprocessing = AddCellPreprocessing()
        self.norm_in = ScaleNormalization()

        self.nn = nn.Linear(
            3,
            2,
            bias=False,
        )
        self.head = nn.Linear(
            2,
            1,
            bias=False,
        )

        with torch.no_grad():
            self.nn.weight.copy_(
                torch.tensor([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0],
                ])
            )
            self.head.weight.copy_(
                torch.tensor([[1.0, -1.0]])
            )

    def forward_cv(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(
            self.nn(
                self.norm_in(x)
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.preprocessing(
            x,
            cell=cell,
        )
        return self.forward_cv(x)


def make_input() -> torch.Tensor:
    return torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 0.0, 1.0],
    ])


def make_representation(
    freeze: bool = True,
) -> VectorRepresentation:
    return MLColvarRepresentation(
        DummyDescriptorCV(),
        freeze=freeze,
    )


def test_vector_representation() -> None:
    representation = make_representation()

    output = representation(
        make_input(),
        cell=torch.tensor(0.5),
    )

    torch.testing.assert_close(
        output,
        torch.tensor([
            [5.0, 16.0],
            [7.0, 8.0],
        ]),
    )

    assert representation.input_kind == "vector"
    assert representation.in_features == 3
    assert representation.out_features == 2
    assert output.shape == (2, 2)


def test_vector_freeze_and_gradients() -> None:
    pretrained = DummyDescriptorCV()

    representation = MLColvarRepresentation(
        pretrained,
        freeze=True,
    )

    x = make_input().requires_grad_(True)
    representation(x).sum().backward()

    assert x.grad is not None
    assert all(
        not parameter.requires_grad
        for parameter in pretrained.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in pretrained.parameters()
    )

    representation.train()

    assert not representation.training
    assert not pretrained.training

    unfrozen = make_representation(
        freeze=False
    )

    assert all(
        parameter.requires_grad
        for parameter in unfrozen.parameters()
    )


def test_vector_representation_preprocessing() -> None:
    pretrained = DummyDescriptorCV()

    representation = MLColvarRepresentation(
        pretrained,
        freeze=True,
    )

    head = FeedForward([
        representation.out_features,
        1,
    ])

    model = RegressionCV(
        model=head,
        preprocessing=representation,
    )

    x = make_input().requires_grad_(True)
    output = model(x)

    assert output.shape == (2, 1)

    output.sum().backward()

    assert x.grad is not None
    assert all(
        parameter.grad is None
        for parameter in pretrained.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.nn.parameters()
    )


def test_vector_representation_torchscript(
    tmp_path,
) -> None:
    representation = make_representation()

    head = FeedForward([
        representation.out_features,
        1,
    ])

    postprocessing = nn.Sequential(
        head,
        nn.Sigmoid(),
    ).eval()

    path = tmp_path / "model.ptc"

    export_representation_torchscript(
        representation=representation,
        postprocessing=postprocessing,
        path=path,
    )

    loaded = torch.jit.load(
        str(path),
        map_location="cpu",
    ).eval()

    output = loaded(
        make_input()
    )

    assert path.exists()
    assert output.shape == (2, 1)