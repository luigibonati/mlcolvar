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
    """FFNN transfer model: pretrained tensor featurizer + trainable readout."""

    def __init__(
        self,
        featurizer: _BaseTensorCVFeaturizer,
        n_out: int = 1,
        hidden_layers: Sequence[int] = (32, 32),
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(featurizer, _BaseTensorCVFeaturizer):
            raise TypeError(
                "`CVReadoutModel` requires a tensor-based CV featurizer. "
                "For GNN models use `CVGraphReadoutModel`."
            )

        n_out = _as_positive_int(n_out, "n_out")
        hidden_layers = tuple(int(size) for size in hidden_layers)

        if any(size <= 0 for size in hidden_layers):
            raise ValueError(
                "`hidden_layers` must contain only positive integers."
            )

        readout_options = {} if options is None else dict(options)

        super().__init__(
            layers=[
                featurizer.out_features,
                *hidden_layers,
                n_out,
            ],
            **readout_options,
        )

        self.featurizer = featurizer

        # BaseCV should see the original descriptor dimension.
        self.in_features = featurizer.in_features
        self.out_features = n_out

        self.register_buffer(
            "_readout_reference",
            _module_reference_tensor(self.nn),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.featurizer(x, cell=cell)
        features = features.to(
            dtype=self._readout_reference.dtype,
            device=self._readout_reference.device,
        )
        return self.nn(features)


class CVGraphReadoutModel(BaseGNN):
    """GNN transfer model: pretrained graph featurizer + trainable readout.

    Inheriting from BaseGNN is necessary because mlcolvar CV training steps
    use ``isinstance(model, BaseGNN)`` to select graph batches.
    """

    def __init__(
        self,
        featurizer: _BaseGraphCVFeaturizer,
        n_out: int = 1,
        hidden_layers: Sequence[int] = (32, 32),
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(featurizer, _BaseGraphCVFeaturizer):
            raise TypeError(
                "`CVGraphReadoutModel` requires a graph-based CV featurizer."
            )

        n_out = _as_positive_int(n_out, "n_out")
        hidden_layers = tuple(int(size) for size in hidden_layers)

        if any(size <= 0 for size in hidden_layers):
            raise ValueError(
                "`hidden_layers` must contain only positive integers."
            )

        super().__init__(
            n_out=n_out,
            dataset_for_initialization=None,
            pooling_operation=None,
            cutoff=float(featurizer.cutoff.detach().cpu().item()),
            buffer=float(featurizer.buffer.detach().cpu().item()),
            long_range_cutoff=float(
                featurizer.long_range_cutoff.detach().cpu().item()
            ),
            atomic_numbers=(
                featurizer.atomic_numbers.detach().cpu().tolist()
            ),
        )

        # The pretrained GNN creates its own graph representation; this
        # wrapper does not use BaseGNN's radial embedding.
        self._modules.pop("_radial_embedding", None)

        self.featurizer = featurizer
        readout_options = {} if options is None else dict(options)
        self.readout = FeedForward(
            layers=[
                featurizer.out_features,
                *hidden_layers,
                n_out,
            ],
            **readout_options,
        )

        self.register_buffer(
            "_readout_reference",
            _module_reference_tensor(self.readout),
            persistent=False,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.featurizer(data, cell=cell)
        features = features.to(
            dtype=self._readout_reference.dtype,
            device=self._readout_reference.device,
        )
        return self.readout(features)
