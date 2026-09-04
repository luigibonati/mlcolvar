from typing import Dict, Optional

import torch
from torch import nn

from .alignment import align_node_attrs
from .backbones import BaseAtomisticBackbone
from .graph import infer_num_graphs


__all__ = ["AtomisticFeaturizer"]


_METADATA = (
    "feature_dim",
    "atomic_numbers",
    "cutoff",
    "buffer",
    "long_range_cutoff",
)


class AtomisticFeaturizer(nn.Module):
    """Pretrained atomistic feature extractor."""

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
            raise ValueError(
                "`pooling` must be 'mean' or 'sum'."
            )

        self.backbone = backbone
        self.out_features = backbone.out_features
        self.sample_kind = backbone.sample_kind
        self.full_neighbor_list = backbone.full_neighbor_list

        self.pooling = pooling
        self.freeze = freeze

        for name in _METADATA:
            self.register_buffer(
                name,
                getattr(backbone, name).detach().clone(),
            )

        if freeze:
            backbone.requires_grad_(False)
            backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze:
            self.backbone.eval()

        return self

    @torch.jit.unused
    def align_dataset(self, dataset):
        return align_node_attrs(
            dataset,
            self.atomic_numbers,
        )

    def forward_features(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.backbone(data, cell=cell)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.forward_features(data, cell)

        if self.sample_kind == "system":
            return features

        return self.pool(features, data)

    def pool(
        self,
        features: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        if "batch" not in data:
            raise KeyError(
                "Graph data must contain `batch` for pooling."
            )

        batch = data["batch"].to(
            device=features.device,
            dtype=torch.long,
        )

        n_systems = infer_num_graphs(data)

        mask = (
            data["system_masks"].reshape(-1, 1).to(features)
            if "system_masks" in data
            else features.new_ones((features.size(0), 1))
        )

        output = features.new_zeros(
            n_systems,
            features.size(-1),
        )

        output.index_add_(
            0,
            batch,
            features * mask,
        )

        if self.pooling == "sum":
            return output

        counts = features.new_zeros(n_systems, 1)
        counts.index_add_(0, batch, mask)

        return output / counts.clamp_min(1)