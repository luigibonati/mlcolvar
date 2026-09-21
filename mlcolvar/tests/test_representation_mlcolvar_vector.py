from typing import Optional

import pytest
import torch
from torch import nn

from mlcolvar.representation import (
    MLColvarRepresentation,
    RepresentationModel,
    TaskHead,
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
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0],
                    ]
                )
            )
            self.head.weight.copy_(
                torch.tensor(
                    [[1.0, -1.0]]
                )
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
    return torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 0.0, 1.0],
        ]
    )


def make_representation(
    mode: str = "latent",
    freeze: bool = True,
) -> VectorRepresentation:
    return MLColvarRepresentation(
        DummyDescriptorCV(),
        mode=mode,
        freeze=freeze,
    )


@pytest.mark.parametrize(
    ("mode", "out_features"),
    [
        ("latent", 2),
        ("forward", 1),
        ("output", 1),
    ],
)
def test_vector_modes(
    mode,
    out_features,
) -> None:
    representation = make_representation(
        mode=mode
    )

    output = representation(
        make_input(),
        cell=torch.tensor(0.5),
    )

    assert representation.input_kind == "vector"
    assert representation.in_features == 3
    assert representation.out_features == out_features
    assert output.shape == (2, out_features)


def test_vector_freeze_and_gradients() -> None:
    pretrained = DummyDescriptorCV()

    representation = MLColvarRepresentation(
        pretrained,
        mode="latent",
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


def test_vector_representation_model() -> None:
    pretrained = DummyDescriptorCV()

    representation = MLColvarRepresentation(
        pretrained,
        mode="latent",
        freeze=True,
    )

    model = RepresentationModel(
        representation,
        n_out=1,
        hidden_layers=(),
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
        for parameter in model.head.parameters()
    )


def test_vector_representation_torchscript(
    tmp_path,
) -> None:
    representation = make_representation()

    model = RepresentationModel(
        representation,
        head=TaskHead(
            representation.out_features,
            n_out=1,
            hidden_layers=(),
        ),
    ).eval()

    path = tmp_path / "model.ptc"

    export_representation_torchscript(
        model=model,
        postprocessing=nn.Sigmoid(),
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