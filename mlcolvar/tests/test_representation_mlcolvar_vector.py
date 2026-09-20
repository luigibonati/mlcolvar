from __future__ import annotations

from typing import Optional

import pytest
import torch
from torch import nn

from mlcolvar.representation import (
    MLColvarRepresentation,
    RepresentationModel,
    TaskHead,
    export_representation_torchscript,
)


class AddCellPreprocessing(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = x + 1.0

        if cell is not None:
            output = (
                output
                + cell.reshape(())
            )

        return output


class ScaleNormalization(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return 2.0 * x


class DummyDescriptorCV(nn.Module):
    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.in_features = 3
        self.out_features = 1

        self.preprocessing = (
            AddCellPreprocessing()
        )
        self.norm_in = ScaleNormalization()

        self.nn = nn.Linear(
            3,
            2,
            bias=False,
            dtype=dtype,
        )

        self.head = nn.Linear(
            2,
            1,
            bias=False,
            dtype=dtype,
        )

        with torch.no_grad():
            self.nn.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0],
                    ],
                    dtype=dtype,
                )
            )

            self.head.weight.copy_(
                torch.tensor(
                    [[1.0, -1.0]],
                    dtype=dtype,
                )
            )

    def forward_cv(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.nn(
            self.norm_in(x)
        )
        return self.head(latent)

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


def make_input(
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )


def make_representation(
    mode: str = "latent",
    freeze: bool = True,
) -> MLColvarRepresentation:
    return MLColvarRepresentation(
        DummyDescriptorCV(),
        mode=mode,
        freeze=freeze,
    )


def expected_latent(
    x: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = x + 1.0

    if cell is not None:
        x = x + cell.reshape(())

    x = 2.0 * x

    return torch.stack(
        [
            x[:, 0],
            x[:, 1] + x[:, 2],
        ],
        dim=1,
    )


def expected_output(
    x: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    latent = expected_latent(
        x,
        cell,
    )

    return (
        latent[:, :1]
        - latent[:, 1:2]
    )


@pytest.mark.parametrize(
    ("mode", "out_features", "expected_fn"),
    [
        ("output", 1, expected_output),
        ("forward", 1, expected_output),
        ("latent", 2, expected_latent),
    ],
)
def test_vector_modes(
    mode,
    out_features,
    expected_fn,
) -> None:
    x = make_input()
    cell = torch.tensor(
        0.5,
        dtype=torch.float64,
    )

    representation = make_representation(
        mode=mode
    )

    output = representation(
        x,
        cell=cell,
    )

    assert representation.input_kind == "vector"
    assert representation.in_features == 3
    assert representation.out_features == out_features
    assert output.dtype == torch.float32

    torch.testing.assert_close(
        output,
        expected_fn(
            x,
            cell,
        ).float(),
    )


def test_frozen_vector_representation() -> None:
    pretrained = DummyDescriptorCV()

    representation = MLColvarRepresentation(
        pretrained,
        mode="latent",
        freeze=True,
    )

    x = (
        make_input()
        .requires_grad_(True)
    )

    representation(
        x
    ).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(
        x.grad
    ).all()

    assert all(
        not p.requires_grad
        for p in pretrained.parameters()
    )
    assert all(
        p.grad is None
        for p in pretrained.parameters()
    )

    representation.train()

    assert not representation.training
    assert not pretrained.training

    state = representation.state_dict()

    assert "model.nn.weight" in state
    assert "model.head.weight" in state
    assert "_model_reference" not in state


def test_unfrozen_vector_representation() -> None:
    pretrained = DummyDescriptorCV()

    representation = MLColvarRepresentation(
        pretrained,
        mode="latent",
        freeze=False,
    )

    assert all(
        p.requires_grad
        for p in pretrained.parameters()
    )

    representation.train()

    assert representation.training
    assert pretrained.training


def test_representation_model_trains_only_head() -> None:
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

    x = (
        make_input()
        .requires_grad_(True)
    )

    output = model(x)

    assert output.shape == (2, 1)

    output.sum().backward()

    assert x.grad is not None
    assert pretrained.nn.weight.grad is None
    assert pretrained.head.weight.grad is None

    assert any(
        p.grad is not None
        for p in model.head.parameters()
    )


def test_cached_head_matches_full_model() -> None:
    representation = make_representation()

    head = TaskHead(
        representation.out_features,
        n_out=1,
        hidden_layers=(),
    )

    model = RepresentationModel(
        representation,
        head=head,
    ).eval()

    x = make_input(
        dtype=torch.float32
    )

    with torch.no_grad():
        cached_output = head(
            representation(x)
        )
        raw_output = model(x)

    assert model.head is head

    torch.testing.assert_close(
        raw_output,
        cached_output,
    )


def test_vector_adapter_validates_interfaces() -> None:
    class MissingInput(nn.Module):
        def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
            return x[:, :1]

    class MissingForwardCV(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_features = 3
            self.out_features = 1

        def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
            return x[:, :1]

    class MissingEncoder(
        MissingForwardCV
    ):
        pass

    with pytest.raises(
        (TypeError, ValueError),
    ):
        MLColvarRepresentation(
            MissingInput(),
            mode="output",
        )

    with pytest.raises(
        TypeError,
        match="forward_cv",
    ):
        MLColvarRepresentation(
            MissingForwardCV(),
            mode="forward",
        )

    with pytest.raises(
        TypeError,
        match="latent encoder",
    ):
        MLColvarRepresentation(
            MissingEncoder(),
            mode="latent",
        )


def test_vector_representation_torchscript_roundtrip(
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

    postprocessing = nn.Sigmoid()
    path = (
        tmp_path
        / "vector_representation.ptc"
    )

    export_representation_torchscript(
        model=model,
        postprocessing=postprocessing,
        path=path,
    )

    loaded = torch.jit.load(
        str(path),
        map_location="cpu",
    ).eval()

    x = make_input(
        dtype=torch.float32
    )

    with torch.no_grad():
        expected = postprocessing(
            model(x)
        )
        output = loaded(x)

    assert path.exists()

    torch.testing.assert_close(
        output,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )