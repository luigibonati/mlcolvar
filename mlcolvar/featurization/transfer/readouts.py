from typing import Any, Dict, Optional, Sequence

import torch

from mlcolvar.core import BaseGNN, FeedForward

from ._utils import _as_positive_int, _module_reference_tensor
from .featurizers import (
    _BaseGraphCVFeaturizer,
    _BaseTensorCVFeaturizer,
)


__all__ = [
    "CVReadoutModel",
    "CVGraphReadoutModel",
]


class CVReadoutModel(FeedForward):
    """Tensor featurizer followed by a trainable readout."""

    def __init__(
        self,
        featurizer: _BaseTensorCVFeaturizer,
        n_out: int = 1,
        hidden_layers: Sequence[int] = (32, 32),
        options: Optional[Dict[str, Any]] = None,
        cached_input: bool = False,
    ) -> None:
        if not isinstance(featurizer, _BaseTensorCVFeaturizer):
            raise TypeError(
                "`CVReadoutModel` requires a tensor featurizer."
            )

        n_out = _as_positive_int(n_out, "n_out")
        hidden_layers = tuple(int(x) for x in hidden_layers)

        if any(x <= 0 for x in hidden_layers):
            raise ValueError(
                "`hidden_layers` must contain positive integers."
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

        # Explicit dimensions.
        self.raw_in_features = featurizer.in_features
        self.latent_features = featurizer.out_features
        self.out_features = n_out

        # Dimension expected by forward() and by downstream CV models.
        self.in_features = (
            self.latent_features
            if self.cached_input
            else self.raw_in_features
        )

        self.register_buffer(
            "_readout_reference",
            _module_reference_tensor(self.nn),
            persistent=False,
        )

    def forward_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the readout on latent features."""

        if features.ndim < 2:
            raise ValueError(
                "`features` must include a batch dimension."
            )

        if features.shape[-1] != self.latent_features:
            raise ValueError(
                f"Expected {self.latent_features} latent features, "
                f"found {features.shape[-1]}."
            )

        features = features.to(
            dtype=self._readout_reference.dtype,
            device=self._readout_reference.device,
        )

        return self.nn(features)

    def forward_raw(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the frozen featurizer followed by the readout."""

        if x.ndim < 2:
            raise ValueError(
                "`x` must include a batch dimension."
            )

        if x.shape[-1] != self.raw_in_features:
            raise ValueError(
                f"Expected {self.raw_in_features} raw features, "
                f"found {x.shape[-1]}."
            )

        latent = self.featurizer(
            x,
            cell=cell,
        )

        return self.forward_features(latent)

    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run cached head-only or complete raw-input inference."""

        if self.cached_input:
            return self.forward_features(x)

        return self.forward_raw(
            x,
            cell=cell,
        )


class CVGraphReadoutModel(BaseGNN):
    """Graph featurizer followed by a trainable readout."""

    def __init__(
        self,
        featurizer: _BaseGraphCVFeaturizer,
        n_out: int = 1,
        hidden_layers: Sequence[int] = (32, 32),
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(featurizer, _BaseGraphCVFeaturizer):
            raise TypeError(
                "`CVGraphReadoutModel` requires a graph featurizer."
            )

        n_out = _as_positive_int(n_out, "n_out")
        hidden_layers = tuple(int(x) for x in hidden_layers)

        if any(x <= 0 for x in hidden_layers):
            raise ValueError(
                "`hidden_layers` must contain positive integers."
            )

        super().__init__(
            n_out=n_out,
            dataset_for_initialization=None,
            pooling_operation=None,
            cutoff=float(
                featurizer.cutoff.detach().cpu()
            ),
            buffer=float(
                featurizer.buffer.detach().cpu()
            ),
            long_range_cutoff=float(
                featurizer.long_range_cutoff.detach().cpu()
            ),
            atomic_numbers=(
                featurizer.atomic_numbers
                .detach()
                .cpu()
                .tolist()
            ),
        )

        # The frozen graph featurizer constructs its own radial representation.
        self._modules.pop(
            "_radial_embedding",
            None,
        )

        self.featurizer = featurizer

        # Do not overwrite BaseGNN.in_features or BaseGNN.out_features:
        # they are read-only properties.
        self.raw_in_features = None
        self.latent_features = int(
            featurizer.out_features
        )
        self.transfer_out_features = int(n_out)

        self.readout = FeedForward(
            layers=[
                self.latent_features,
                *hidden_layers,
                self.transfer_out_features,
            ],
            **({} if options is None else dict(options)),
        )

        self.register_buffer(
            "_readout_reference",
            _module_reference_tensor(self.readout),
            persistent=False,
        )

    def forward_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the readout on graph-level latent features."""

        if features.ndim < 2:
            raise ValueError(
                "`features` must include a batch dimension."
            )

        if features.shape[-1] != self.latent_features:
            raise ValueError(
                f"Expected {self.latent_features} graph features, "
                f"found {features.shape[-1]}."
            )

        features = features.to(
            dtype=self._readout_reference.dtype,
            device=self._readout_reference.device,
        )

        return self.readout(features)

    def forward_raw(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the frozen graph featurizer followed by the readout."""

        latent = self.featurizer(
            data,
            cell=cell,
        )

        return self.forward_features(latent)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run complete graph-input inference."""

        return self.forward_raw(
            data,
            cell=cell,
        )