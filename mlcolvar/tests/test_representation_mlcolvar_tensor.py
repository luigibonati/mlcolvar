from __future__ import annotations

from typing import Optional

import pytest
import torch
from torch import nn

from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.data import DictDataset
from mlcolvar.representation import (
    CachedRepresentationDerivatives,
    MLColvarRepresentation,
    RepresentationModel,
    TaskHead,
    export_representation_torchscript,
    precompute_committor_cache,
)


class AddCellPreprocessing(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = x + 1.0
        if cell is not None:
            output = output + cell.reshape(())
        return output


class ScaleNormalization(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return 2.0 * x


class DummyDescriptorCV(nn.Module):
    """Minimal descriptor-based CV with preprocessing, encoder and head."""

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.in_features = 3
        self.out_features = 1

        self.preprocessing = AddCellPreprocessing()
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


def make_descriptor_input(
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )


def expected_preprocessed_descriptors(
    x: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    output = x + 1.0
    if cell is not None:
        output = output + cell.reshape(())
    return output


def expected_latent_features(
    x: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = 2.0 * expected_preprocessed_descriptors(
        x,
        cell=cell,
    )
    return torch.stack(
        [
            x[:, 0],
            x[:, 1] + x[:, 2],
        ],
        dim=1,
    )


def expected_cv_output(
    x: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    latent = expected_latent_features(
        x,
        cell=cell,
    )
    return (
        latent[:, :1]
        - latent[:, 1:2]
    )


def test_mlcolvar_tensor_modes() -> None:
    x = make_descriptor_input()
    cell = torch.tensor(
        0.5,
        dtype=torch.float64,
    )

    output_rep = MLColvarRepresentation(
        DummyDescriptorCV(),
        mode="output",
        freeze=True,
    )
    forward_rep = MLColvarRepresentation(
        DummyDescriptorCV(),
        mode="forward",
        freeze=True,
    )
    latent_rep = MLColvarRepresentation(
        DummyDescriptorCV(),
        mode="latent",
        freeze=True,
    )

    output = output_rep(
        x,
        cell=cell,
    )
    forward_output = forward_rep(
        x,
        cell=cell,
    )
    latent = latent_rep(
        x,
        cell=cell,
    )

    assert output_rep.in_features == 3
    assert output_rep.out_features == 1

    assert forward_rep.in_features == 3
    assert forward_rep.out_features == 1

    assert latent_rep.in_features == 3
    assert latent_rep.out_features == 2

    assert output.dtype == torch.float32
    assert forward_output.dtype == torch.float32
    assert latent.dtype == torch.float32

    torch.testing.assert_close(
        output,
        expected_cv_output(
            x,
            cell=cell,
        ).to(torch.float32),
    )

    torch.testing.assert_close(
        forward_output,
        expected_cv_output(
            x,
            cell=cell,
        ).to(torch.float32),
    )

    torch.testing.assert_close(
        latent,
        expected_latent_features(
            x,
            cell=cell,
        ).to(torch.float32),
    )


def test_frozen_mlcolvar_tensor_representation_preserves_gradients() -> None:
    pretrained = DummyDescriptorCV()

    representation = MLColvarRepresentation(
        pretrained,
        mode="latent",
        freeze=True,
    )

    x = (
        make_descriptor_input()
        .requires_grad_(True)
    )

    output = representation(x)
    output.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0

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


def test_unfrozen_mlcolvar_tensor_representation_is_trainable() -> None:
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


def test_mlcolvar_tensor_state_dict_contains_model_not_reference() -> None:
    representation = MLColvarRepresentation(
        DummyDescriptorCV(),
        mode="latent",
        freeze=True,
    )

    state = representation.state_dict()

    assert "model.nn.weight" in state
    assert "model.head.weight" in state
    assert "_model_reference" not in state


def test_representation_model_trains_only_task_head() -> None:
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

    assert model.in_features == 3
    assert model.out_features == 1
    assert model.representation is representation

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    # Linear(2, 1): two weights + one bias.
    assert sum(p.numel() for p in trainable) == 3

    linear = next(
        module
        for module in model.head.modules()
        if isinstance(module, nn.Linear)
    )

    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.zero_()

    x = (
        make_descriptor_input()
        .requires_grad_(True)
    )

    output = model(x)

    assert output.shape == (2, 1)
    assert output.dtype == linear.weight.dtype

    output.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert linear.weight.grad is not None

    assert pretrained.nn.weight.grad is None
    assert pretrained.head.weight.grad is None


def test_cached_latent_head_and_full_model_agree() -> None:
    representation = MLColvarRepresentation(
        DummyDescriptorCV(),
        mode="latent",
        freeze=True,
    )

    head = TaskHead(
        representation.out_features,
        n_out=1,
        hidden_layers=(),
    )

    model = RepresentationModel(
        representation,
        head=head,
    ).eval()

    x = make_descriptor_input(
        dtype=torch.float32,
    )

    with torch.no_grad():
        latent = representation(x)
        cached_output = head(latent)
        raw_output = model(x)

    assert model.head is head
    torch.testing.assert_close(
        raw_output,
        cached_output,
    )


@pytest.mark.parametrize(
    "hidden_layers",
    [
        (0,),
        (-1,),
        (4, 0),
    ],
)
def test_task_head_rejects_invalid_hidden_layers(
    hidden_layers,
) -> None:
    with pytest.raises(
        ValueError,
        match="positive integers",
    ):
        TaskHead(
            2,
            hidden_layers=hidden_layers,
        )


def test_mlcolvar_tensor_adapter_validates_interfaces() -> None:
    class MissingInputDimension(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out_features = 1

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

    class MissingEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_features = 3
            self.out_features = 1

        def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
            return x[:, :1]

    with pytest.raises(
        (TypeError, ValueError),
    ):
        MLColvarRepresentation(
            MissingInputDimension(),
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
        match="`.nn` latent encoder",
    ):
        MLColvarRepresentation(
            MissingEncoder(),
            mode="latent",
        )


class IdentityDescriptorDerivatives(SmartDerivatives):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(
        self,
        gradient_descriptor: torch.Tensor,
        ref_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del ref_idx
        return gradient_descriptor.unsqueeze(1)


def test_tensor_committor_cache_matches_direct_representation() -> None:
    x = make_descriptor_input(
        dtype=torch.float32,
    )

    dataset = DictDataset(
        {
            "data": x,
            "labels": torch.full(
                (len(x),),
                2.0,
            ),
            "weights": torch.ones(len(x)),
            "ref_idx": torch.arange(
                len(x),
                dtype=torch.long,
            ),
        }
    )

    representation = MLColvarRepresentation(
        DummyDescriptorCV(),
        mode="latent",
        freeze=True,
    )

    cached_dataset, cached_derivatives = (
        precompute_committor_cache(
            representation=representation,
            dataset=dataset,
            descriptor_derivatives=(
                IdentityDescriptorDerivatives()
            ),
            batch_size=1,
            device="cpu",
            output_device="cpu",
            separate_boundary_dataset=False,
        )
    )

    with torch.no_grad():
        expected_latent = representation(x)

    torch.testing.assert_close(
        cached_dataset["data"],
        expected_latent,
    )

    assert isinstance(
        cached_derivatives,
        CachedRepresentationDerivatives,
    )

    expected_jacobian_single = torch.tensor(
        [
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [0.0, 2.0],
            ]
        ],
        dtype=torch.float32,
    )

    expected_jacobian = (
        expected_jacobian_single
        .unsqueeze(0)
        .repeat(len(x), 1, 1, 1)
    )

    torch.testing.assert_close(
        cached_derivatives.jacobian,
        expected_jacobian,
    )


def test_tensor_representation_torchscript_roundtrip(
    tmp_path,
) -> None:
    representation = MLColvarRepresentation(
        DummyDescriptorCV(),
        mode="latent",
        freeze=True,
    )

    head = TaskHead(
        representation.out_features,
        n_out=1,
        hidden_layers=(),
    )

    model = RepresentationModel(
        representation,
        head=head,
    ).eval()

    postprocessing = nn.Sigmoid()
    path = tmp_path / "tensor_representation.ptc"

    export_representation_torchscript(
        model=model,
        postprocessing=postprocessing,
        path=path,
    )

    loaded = torch.jit.load(
        str(path),
        map_location="cpu",
    ).eval()

    x = make_descriptor_input(
        dtype=torch.float32,
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
