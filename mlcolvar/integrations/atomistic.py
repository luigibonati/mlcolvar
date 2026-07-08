from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from mlcolvar.core import BaseGNN, FeedForward
from mlcolvar.integrations.utils import (
    align_node_attrs,
    infer_num_graphs,
)

__all__ = [
    "BaseAtomisticBackbone",
    "AtomisticFeaturizer",
    "AtomisticModel",
]


_METADATA_BUFFERS = (
    "feature_dim",
    "atomic_numbers",
    "cutoff",
    "buffer",
    "long_range_cutoff",
)


def _copy_backbone_metadata(
    module: nn.Module,
    backbone: nn.Module,
) -> None:
    for name in _METADATA_BUFFERS:
        module.register_buffer(
            name,
            getattr(backbone, name).detach().clone(),
        )


class BaseAtomisticBackbone(nn.Module):
    """Base class for pretrained atomistic-model backbones."""

    __constants__ = [
        "out_features",
        "sample_kind",
        "full_neighbor_list",
    ]

    def __init__(
        self,
        out_features: int,
        atomic_numbers: Sequence[int],
        cutoff: float,
        sample_kind: str = "atom",
        buffer: float = 0.0,
        long_range_cutoff: float = -1.0,
        full_neighbor_list: bool = True,
    ) -> None:
        super().__init__()

        atomic_numbers = [int(number) for number in atomic_numbers]

        if out_features <= 0:
            raise ValueError("`out_features` must be positive.")

        if len(atomic_numbers) == 0:
            raise ValueError("`atomic_numbers` cannot be empty.")

        if len(set(atomic_numbers)) != len(atomic_numbers):
            raise ValueError("`atomic_numbers` must not contain duplicates.")

        if any(number <= 0 for number in atomic_numbers):
            raise ValueError(
                "`atomic_numbers` must contain positive integers."
            )

        if cutoff <= 0.0:
            raise ValueError("`cutoff` must be positive.")

        if buffer < 0.0:
            raise ValueError("`buffer` must be non-negative.")

        if long_range_cutoff >= 0.0 and long_range_cutoff <= cutoff:
            raise ValueError(
                "`long_range_cutoff` must be negative or larger than "
                "`cutoff`."
            )

        if sample_kind not in ("atom", "system"):
            raise ValueError(
                "`sample_kind` must be either 'atom' or 'system'."
            )

        self.out_features = int(out_features)
        self.sample_kind = sample_kind
        self.full_neighbor_list = bool(full_neighbor_list)

        self.register_buffer(
            "feature_dim",
            torch.tensor(out_features, dtype=torch.int64),
        )
        self.register_buffer(
            "atomic_numbers",
            torch.tensor(atomic_numbers, dtype=torch.int64),
        )
        self.register_buffer(
            "cutoff",
            torch.tensor(cutoff, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "buffer",
            torch.tensor(buffer, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "long_range_cutoff",
            torch.tensor(
                long_range_cutoff,
                dtype=torch.get_default_dtype(),
            ),
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class AtomisticFeaturizer(nn.Module):
    """Convert an atomistic backbone into graph-level features."""

    __constants__ = [
        "out_features",
        "sample_kind",
        "pooling",
        "freeze",
        "full_neighbor_list",
    ]

    def __init__(
        self,
        backbone: BaseAtomisticBackbone,
        pooling: str = "mean",
        freeze: bool = True,
    ) -> None:
        super().__init__()

        if pooling not in ("mean", "sum"):
            raise ValueError("`pooling` must be either 'mean' or 'sum'.")

        self.backbone = backbone
        self.out_features = backbone.out_features
        self.sample_kind = backbone.sample_kind
        self.pooling = pooling
        self.freeze = bool(freeze)
        self.full_neighbor_list = backbone.full_neighbor_list

        _copy_backbone_metadata(self, backbone)

        if self.freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

            self.backbone.eval()

    @torch.jit.unused
    def align_dataset(
        self,
        dataset,
    ):
        return align_node_attrs(
            dataset=dataset,
            target_atomic_numbers=self.atomic_numbers,
        )

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)

        if self.freeze:
            self.backbone.eval()

        return self

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.backbone(
            data,
            cell=cell,
        )

        if self.sample_kind == "system":
            return features

        return self._pool_node_features(
            node_features=features,
            data=data,
        )

    def _pool_node_features(
        self,
        node_features: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch = data["batch"].to(
            device=node_features.device,
            dtype=torch.long,
        )

        n_graphs = infer_num_graphs(data)

        if "system_masks" in data:
            mask = data["system_masks"].reshape(-1, 1).to(
                device=node_features.device,
                dtype=node_features.dtype,
            )
        else:
            mask = node_features.new_ones(
                (node_features.size(0), 1)
            )

        output = node_features.new_zeros(
            (n_graphs, node_features.size(-1))
        )

        output.index_add_(
            0,
            batch,
            node_features * mask,
        )

        if self.pooling == "sum":
            return output

        counts = node_features.new_zeros((n_graphs, 1))

        counts.index_add_(
            0,
            batch,
            mask,
        )

        return output / counts.clamp_min(1.0)


class AtomisticModel(BaseGNN):
    """Atomistic featurizer followed by a trainable CV readout."""

    def __init__(
        self,
        featurizer: AtomisticFeaturizer,
        n_out: int = 1,
        hidden_layers: Tuple[int, ...] = (30, 30),
    ) -> None:
        if n_out <= 0:
            raise ValueError("`n_out` must be positive.")

        if any(size <= 0 for size in hidden_layers):
            raise ValueError(
                "`hidden_layers` must contain positive integers."
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
            atomic_numbers=featurizer.atomic_numbers.detach().cpu().tolist(),
        )

        # Unused for external atomistic backbones.
        self._modules.pop("_radial_embedding", None)

        self.featurizer = featurizer
        self.readout = FeedForward(
            layers=[
                featurizer.out_features,
                *hidden_layers,
                n_out,
            ],
        )

        readout_parameter = next(self.readout.parameters())

        self.register_buffer(
            "_readout_dtype_reference",
            torch.zeros(
                (),
                device=readout_parameter.device,
                dtype=readout_parameter.dtype,
            ),
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.featurizer(
            data,
            cell=cell,
        )

        features = features.to(
            device=self._readout_dtype_reference.device,
            dtype=self._readout_dtype_reference.dtype,
        )

        return self.readout(features)