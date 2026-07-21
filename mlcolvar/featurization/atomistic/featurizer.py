from typing import Dict, Optional

import torch
from torch import nn

from .backbones.base import BaseAtomisticBackbone
from .alignment import align_node_attrs
from .graph import infer_num_graphs


__all__ = [
    "AtomisticFeaturizer",
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
    backbone: BaseAtomisticBackbone,
) -> None:
    """Copy standardized backbone metadata to another module."""

    for name in _METADATA_BUFFERS:
        module.register_buffer(
            name,
            getattr(backbone, name).detach().clone(),
        )


class AtomisticFeaturizer(nn.Module):
    """Extract features from a pretrained atomistic backbone.

    The default :meth:`forward` method converts atom-level features into
    system-level features using mean or sum pooling.

    Raw atom-level features can be obtained through
    :meth:`forward_features`.
    """

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
    def align_dataset(self, dataset):
        """Align dataset node attributes with the model element table."""

        return align_node_attrs(
            dataset=dataset,
            target_atomic_numbers=self.atomic_numbers,
        )

    def train(self, mode: bool = True):
        """Keep a frozen backbone in evaluation mode."""

        super().train(mode)

        if self.freeze:
            self.backbone.eval()

        return self

    def forward_features(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return raw atom- or system-level backbone features."""

        return self.backbone(data, cell=cell)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return system-level features using the configured pooling."""

        features = self.forward_features(data, cell=cell)

        if self.sample_kind == "system":
            return features

        return self.pool_node_features(
            node_features=features,
            data=data,
        )

    def pool_node_features(
        self,
        node_features: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pool node-level values into system-level values.

        If ``system_masks`` is present, only nodes with non-zero mask
        values contribute to the pooling operation.
        """

        if "batch" not in data:
            raise KeyError("Graph data must contain `batch` for node pooling.")

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
            mask = node_features.new_ones((node_features.size(0), 1))

        output = node_features.new_zeros(
            (n_graphs, node_features.size(-1))
        )
        output.index_add_(0, batch, node_features * mask)

        if self.pooling == "sum":
            return output

        counts = node_features.new_zeros((n_graphs, 1))
        counts.index_add_(0, batch, mask)

        return output / counts.clamp_min(1.0)
