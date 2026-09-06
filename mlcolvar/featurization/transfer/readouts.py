from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from mlcolvar.core import BaseGNN, FeedForward

from ._utils import (
    _as_positive_int,
    _module_reference_tensor,
)
from .featurizers import (
    _BaseGraphFeaturizer,
    _BaseTensorFeaturizer,
)


__all__ = ["TransferModel"]


def _validate_readout(
    n_out: int,
    hidden_layers: Sequence[int],
) -> Tuple[int, Tuple[int, ...]]:
    """Validate common readout options."""

    n_out = _as_positive_int(n_out, "n_out")
    hidden_layers = tuple(int(x) for x in hidden_layers)

    if any(x <= 0 for x in hidden_layers):
        raise ValueError(
            "`hidden_layers` must contain positive integers."
        )

    return n_out, hidden_layers


def _prepare_features(
    features: torch.Tensor,
    expected_features: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Validate and cast latent features for the readout."""

    if features.ndim < 2:
        raise ValueError(
            "`features` must include a batch dimension."
        )

    if features.shape[-1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} latent features, "
            f"found {features.shape[-1]}."
        )

    return features.to(
        dtype=reference.dtype,
        device=reference.device,
    )


class _TensorTransferModel(FeedForward):
    """Frozen tensor featurizer followed by a trainable readout."""

    def __init__(
        self,
        featurizer: _BaseTensorFeaturizer,
        n_out: int = 1,
        hidden_layers: Sequence[int] = (32, 32),
        options: Optional[Dict[str, Any]] = None,
        cached_input: bool = False,
    ) -> None:
        n_out, hidden_layers = _validate_readout(
            n_out,
            hidden_layers,
        )

        super().__init__(
            layers=[
                featurizer.out_features,
                *hidden_layers,
                n_out,
            ],
            **({} if options is None else dict(options)),
        )

        self.featurizer = featurizer
        self.cached_input = bool(cached_input)

        self.raw_in_features = _as_positive_int(
            featurizer.in_features,
            "featurizer.in_features",
        )
        self.latent_features = _as_positive_int(
            featurizer.out_features,
            "featurizer.out_features",
        )

        self.in_features = (
            self.latent_features
            if self.cached_input
            else self.raw_in_features
        )
        self.out_features = n_out

        self.register_buffer(
            "_readout_reference",
            _module_reference_tensor(self.nn),
            persistent=False,
        )

    def forward_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply only the trainable readout."""

        features = _prepare_features(
            features,
            self.latent_features,
            self._readout_reference,
        )

        return self.nn(features)

    def forward_raw(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the frozen featurizer followed by the readout."""

        if x.ndim < 2:
            raise ValueError(
                "`x` must include a batch dimension."
            )

        if x.shape[-1] != self.raw_in_features:
            raise ValueError(
                f"Expected {self.raw_in_features} raw features, "
                f"found {x.shape[-1]}."
            )

        return self.forward_features(
            self.featurizer(
                x,
                cell=cell,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.cached_input:
            return self.forward_features(x)

        return self.forward_raw(
            x,
            cell=cell,
        )


class _GraphTransferModel(BaseGNN):
    """Frozen graph featurizer followed by a trainable readout."""

    def __init__(
        self,
        featurizer: _BaseGraphFeaturizer,
        n_out: int = 1,
        hidden_layers: Sequence[int] = (32, 32),
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        n_out, hidden_layers = _validate_readout(
            n_out,
            hidden_layers,
        )

        super().__init__(
            n_out=n_out,
            dataset_for_initialization=None,
            pooling_operation=None,
            cutoff=float(
                featurizer.cutoff.detach().cpu().item()
            ),
            buffer=float(
                featurizer.buffer.detach().cpu().item()
            ),
            long_range_cutoff=float(
                featurizer.long_range_cutoff
                .detach()
                .cpu()
                .item()
            ),
            atomic_numbers=(
                featurizer.atomic_numbers
                .detach()
                .cpu()
                .tolist()
            ),
        )

        # Keep BaseGNN compatibility for graph metadata and deployment.
        # Neighborhood construction is handled by the frozen GNN.
        self._modules.pop(
            "_radial_embedding",
            None,
        )

        self.featurizer = featurizer
        self.raw_in_features = None

        self.latent_features = _as_positive_int(
            featurizer.out_features,
            "featurizer.out_features",
        )

        # Required by transfer inference/export.
        self.transfer_out_features = n_out

        self.readout = FeedForward(
            layers=[
                self.latent_features,
                *hidden_layers,
                n_out,
            ],
            **({} if options is None else dict(options)),
        )

        self.register_buffer(
            "_readout_reference",
            _module_reference_tensor(
                self.readout
            ),
            persistent=False,
        )

    def forward_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply only the trainable readout."""

        features = _prepare_features(
            features,
            self.latent_features,
            self._readout_reference,
        )

        return self.readout(features)

    def forward_raw(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the frozen graph featurizer followed by the readout."""

        return self.forward_features(
            self.featurizer(
                data,
                cell=cell,
            )
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_raw(
            data,
            cell=cell,
        )


def TransferModel(
    featurizer: nn.Module,
    n_out: int = 1,
    hidden_layers: Sequence[int] = (32, 32),
    options: Optional[Dict[str, Any]] = None,
    cached_input: bool = False,
) -> nn.Module:
    """Create a transfer-learning model.

    The implementation is selected automatically from the featurizer type.

    Parameters
    ----------
    featurizer
        Featurizer created with :func:`TransferFeaturizer`.
    n_out
        Number of downstream outputs.
    hidden_layers
        Hidden dimensions of the trainable readout.
    options
        Optional ``FeedForward`` configuration.
    cached_input
        If ``True``, tensor models expect precomputed latent features
        instead of raw inputs.
    """

    if isinstance(featurizer, _BaseTensorFeaturizer):
        return _TensorTransferModel(
            featurizer=featurizer,
            n_out=n_out,
            hidden_layers=hidden_layers,
            options=options,
            cached_input=cached_input,
        )

    if isinstance(featurizer, _BaseGraphFeaturizer):
        if cached_input:
            raise ValueError(
                "`cached_input=True` is currently supported only "
                "for tensor transfer models."
            )

        return _GraphTransferModel(
            featurizer=featurizer,
            n_out=n_out,
            hidden_layers=hidden_layers,
            options=options,
        )

    raise TypeError(
        "`featurizer` must be created with `TransferFeaturizer`. "
        f"Found {type(featurizer)}."
    )