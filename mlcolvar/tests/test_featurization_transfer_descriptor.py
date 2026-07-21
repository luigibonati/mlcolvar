from __future__ import annotations

from typing import Optional

import pytest
import torch
from torch import nn

from mlcolvar.featurization.transfer import (
    CVForwardFeaturizer,
    CVLatentFeaturizer,
    CVOutputFeaturizer,
    CVReadoutModel,
)


class AddCellPreprocessing(nn.Module):
    """Add one and, when supplied, a scalar cell-dependent shift."""

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
    """Simple deterministic input-normalization layer."""

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return 2.0 * x


class DummyDescriptorCV(nn.Module):
    """Minimal descriptor-based CV with preprocessing, encoder, and head."""

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
    """Create a two-sample fixed-length descriptor batch."""

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
    """Apply the expected preprocessing transformation."""

    output = x + 1.0

    if cell is not None:
        output = output + cell.reshape(())

    return output


def expected_latent_features(
    x: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the expected two-dimensional latent representation."""

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
    """Compute the expected scalar CV output."""

    latent = expected_latent_features(
        x,
        cell=cell,
    )

    return (
        latent[:, :1]
        - latent[:, 1:2]
    )


def test_descriptor_featurizers_use_expected_model_stage() -> None:
    """Output, forward-CV, and latent wrappers expose the intended stage."""

    x = make_descriptor_input()

    cell = torch.tensor(
        0.5,
        dtype=torch.float64,
    )

    output_model = DummyDescriptorCV()
    forward_model = DummyDescriptorCV()
    latent_model = DummyDescriptorCV()

    output_featurizer = CVOutputFeaturizer(
        model=output_model,
        freeze=True,
    )

    forward_featurizer = CVForwardFeaturizer(
        model=forward_model,
        freeze=True,
    )

    latent_featurizer = CVLatentFeaturizer(
        model=latent_model,
        freeze=True,
    )

    output = output_featurizer(
        x,
        cell=cell,
    )

    forward_output = forward_featurizer(
        x,
        cell=cell,
    )

    latent = latent_featurizer(
        x,
        cell=cell,
    )

    assert output_featurizer.in_features == 3
    assert output_featurizer.out_features == 1

    assert forward_featurizer.in_features == 3
    assert forward_featurizer.out_features == 1

    assert latent_featurizer.in_features == 3
    assert latent_featurizer.out_features == 2

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


def test_frozen_descriptor_featurizer_preserves_input_gradients() -> None:
    """Freezing model parameters must not disable descriptor gradients."""

    model = DummyDescriptorCV()

    featurizer = CVLatentFeaturizer(
        model=model,
        freeze=True,
    )

    x = (
        make_descriptor_input()
        .requires_grad_(True)
    )

    output = featurizer(x)
    output.sum().backward()

    gradient = x.grad

    assert gradient is not None
    assert gradient.shape == x.shape
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0

    assert all(
        not parameter.requires_grad
        for parameter in model.parameters()
    )

    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )

    featurizer.train()

    assert featurizer.training
    assert not model.training


def test_unfrozen_descriptor_featurizer_keeps_model_trainable() -> None:
    """Allow fine-tuning when ``freeze=False`` is selected."""

    model = DummyDescriptorCV()

    featurizer = CVLatentFeaturizer(
        model=model,
        freeze=False,
    )

    assert all(
        parameter.requires_grad
        for parameter in model.parameters()
    )

    featurizer.train()

    assert featurizer.training
    assert model.training


def test_descriptor_featurizer_state_dict_contains_model_not_reference() -> None:
    """Save the pretrained model while excluding the dtype/device sentinel."""

    featurizer = CVLatentFeaturizer(
        model=DummyDescriptorCV(),
        freeze=True,
    )

    state = featurizer.state_dict()

    assert "model.nn.weight" in state
    assert "model.head.weight" in state
    assert "_model_reference" not in state


def test_cv_readout_model_trains_only_new_head() -> None:
    """Compose a frozen descriptor encoder with a trainable linear probe."""

    pretrained_model = DummyDescriptorCV()

    featurizer = CVLatentFeaturizer(
        model=pretrained_model,
        freeze=True,
    )

    model = CVReadoutModel(
        featurizer=featurizer,
        n_out=1,
        hidden_layers=(),
    )

    assert model.in_features == 3
    assert model.out_features == 1
    assert model.featurizer is featurizer

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    # Linear(2, 1): two weights and one bias.
    assert sum(
        parameter.numel()
        for parameter in trainable_parameters
    ) == 3

    assert all(
        not parameter.requires_grad
        for parameter in pretrained_model.parameters()
    )

    linear = next(
        module
        for module in model.nn.modules()
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
    assert torch.count_nonzero(x.grad) > 0

    assert linear.weight.grad is not None

    assert pretrained_model.nn.weight.grad is None
    assert pretrained_model.head.weight.grad is None

    model.train()

    assert model.training
    assert model.featurizer.training
    assert not pretrained_model.training


def test_descriptor_readout_can_be_traced() -> None:
    """Trace the frozen descriptor encoder and readout together."""

    model = CVReadoutModel(
        featurizer=CVLatentFeaturizer(
            model=DummyDescriptorCV(),
            freeze=True,
        ),
        n_out=1,
        hidden_layers=(),
    ).eval()

    x = make_descriptor_input()

    expected = model(x)

    traced = torch.jit.trace(
        model,
        example_inputs=(x,),
        check_trace=True,
    )

    output = traced(x)

    torch.testing.assert_close(
        output,
        expected,
    )


@pytest.mark.parametrize(
    "hidden_layers",
    [
        (0,),
        (-1,),
        (4, 0),
    ],
)
def test_descriptor_readout_rejects_invalid_hidden_layers(
    hidden_layers,
) -> None:
    """Reject hidden layers with non-positive dimensions."""

    featurizer = CVLatentFeaturizer(
        model=DummyDescriptorCV(),
    )

    with pytest.raises(
        ValueError,
        match="must contain only positive integers",
    ):
        CVReadoutModel(
            featurizer=featurizer,
            hidden_layers=hidden_layers,
        )


def test_descriptor_featurizers_validate_required_interfaces() -> None:
    """Reject models that do not provide the requested CV interface."""

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
        AttributeError,
        match="does not define `in_features`",
    ):
        CVOutputFeaturizer(
            MissingInputDimension()
        )

    with pytest.raises(
        TypeError,
        match="does not implement",
    ):
        CVForwardFeaturizer(
            MissingForwardCV()
        )

    with pytest.raises(
        TypeError,
        match="does not define an `nn` block",
    ):
        CVLatentFeaturizer(
            MissingEncoder()
        )