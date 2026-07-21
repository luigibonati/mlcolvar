from typing import Dict, Optional, Tuple

import torch
from torch import nn

from mlcolvar.integrations.atomistic import BaseAtomisticBackbone

from ._utils import to_float, to_int, to_int_list


__all__ = ["MACEBackbone"]


def _resolve_num_layers(
    model: nn.Module,
    num_layers: Optional[int],
) -> int:
    """Resolve the number of selected MACE interaction layers."""

    if hasattr(model, "num_interactions"):
        available_layers = to_int(
            model.num_interactions,
            name="model.num_interactions",
        )

        if available_layers <= 0:
            raise ValueError(
                "MACE `num_interactions` must be positive, "
                f"found {available_layers}."
            )

        selected_layers = (
            available_layers
            if num_layers is None
            else to_int(num_layers, name="num_layers")
        )

        if selected_layers > available_layers:
            raise ValueError(
                f"Requested {selected_layers} MACE layers, but the "
                f"model contains only {available_layers}."
            )

    elif num_layers is None:
        raise ValueError(
            "Could not infer the number of MACE interaction layers. "
            "Pass `num_layers` explicitly."
        )

    else:
        selected_layers = to_int(
            num_layers,
            name="num_layers",
        )

    if selected_layers <= 0:
        raise ValueError(
            "`num_layers` must be positive, "
            f"found {selected_layers}."
        )

    return selected_layers


def _infer_descriptor_layout(
    model: nn.Module,
) -> Tuple[int, int]:
    """Infer ``num_features`` and ``l_max`` from a MACE model."""

    try:
        irreps_out = model.products[0].linear.irreps_out
        descriptor_dim = int(irreps_out.dim)
        l_max = int(irreps_out.lmax)

    except (
        AttributeError,
        IndexError,
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise ValueError(
            "Could not infer the MACE descriptor layout from "
            "`model.products[0].linear.irreps_out`."
        ) from exc

    if descriptor_dim <= 0:
        raise ValueError(
            "The MACE descriptor dimension must be positive, "
            f"found {descriptor_dim}."
        )

    if l_max < 0:
        raise ValueError(
            "The MACE descriptor `l_max` must be non-negative, "
            f"found {l_max}."
        )

    angular_size = (l_max + 1) ** 2

    if descriptor_dim % angular_size != 0:
        raise ValueError(
            "The MACE descriptor dimension is incompatible with "
            f"`l_max`: descriptor_dim={descriptor_dim}, "
            f"l_max={l_max}."
        )

    return descriptor_dim // angular_size, l_max


def _resolve_descriptor_layout(
    model: nn.Module,
    num_features: Optional[int],
    l_max: Optional[int],
) -> Tuple[int, int]:
    """Resolve the MACE descriptor layout."""

    if num_features is None or l_max is None:
        inferred_num_features, inferred_l_max = _infer_descriptor_layout(model)

        if num_features is None:
            num_features = inferred_num_features

        if l_max is None:
            l_max = inferred_l_max

    num_features = to_int(
        num_features,
        name="num_features",
    )

    l_max = to_int(
        l_max,
        name="l_max",
    )

    if num_features <= 0:
        raise ValueError(
            "`num_features` must be positive, "
            f"found {num_features}."
        )

    if l_max < 0:
        raise ValueError(
            "`l_max` must be non-negative, "
            f"found {l_max}."
        )

    return num_features, l_max


class MACEBackbone(BaseAtomisticBackbone):
    """Extract invariant atom-level features from a pretrained MACE model."""

    __constants__ = [
        "num_layers",
        "num_features",
        "l_max",
        "layer_size",
        "required_input_features",
    ]
    
    def __init__(
        self,
        model: nn.Module,
        num_layers: Optional[int] = None,
        num_features: Optional[int] = None,
        l_max: Optional[int] = None,
        buffer: float = 0.0,
        long_range_cutoff: float = -1.0,
    ) -> None:
        if not hasattr(model, "atomic_numbers"):
            raise ValueError(
                "The MACE model does not expose `atomic_numbers`."
            )

        if not hasattr(model, "r_max"):
            raise ValueError(
                "The MACE model does not expose `r_max`."
            )

        num_layers = _resolve_num_layers(
            model=model,
            num_layers=num_layers,
        )

        num_features, l_max = _resolve_descriptor_layout(
            model=model,
            num_features=num_features,
            l_max=l_max,
        )

        super().__init__(
            out_features=num_layers * num_features,
            atomic_numbers=to_int_list(
                model.atomic_numbers,
                name="model.atomic_numbers",
            ),
            cutoff=to_float(
                model.r_max,
                name="model.r_max",
            ),
            sample_kind="atom",
            buffer=buffer,
            long_range_cutoff=long_range_cutoff,
            full_neighbor_list=True,
        )

        self.num_layers = num_layers
        self.num_features = num_features
        self.l_max = l_max
        self.layer_size = (l_max + 1) ** 2 * num_features

        self.required_input_features = (
            (num_layers - 1) * self.layer_size + num_features
        )

        self.model = model

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = cell

        output = self.model(
            data,
            training=self.training,
            compute_force=False,
        )

        node_features = output["node_feats"]

        if node_features is None:
            raise RuntimeError(
                "The MACE model returned `node_feats=None`."
            )

        return self._extract_invariant_features(node_features)

    def _extract_invariant_features(
        self,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the leading scalar block from each selected MACE layer."""

        return torch.cat(
            [
                node_features[
                    :,
                    layer_index * self.layer_size:
                    layer_index * self.layer_size + self.num_features,
                ]
                for layer_index in range(self.num_layers)
            ],
            dim=-1,
        )   