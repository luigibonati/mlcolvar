from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from ..base import GraphRepresentation
from ._utils import to_float, to_int, to_int_list


__all__ = ["MACERepresentation"]


def _resolve_num_layers(
    model: nn.Module,
    num_layers: Optional[int],
) -> int:
    """Resolve the number of selected MACE interaction layers."""

    if hasattr(model, "num_interactions"):
        available = to_int(
            model.num_interactions,
            name="model.num_interactions",
        )

        if available <= 0:
            raise ValueError(
                "MACE `num_interactions` must be positive."
            )

        selected = (
            available
            if num_layers is None
            else to_int(num_layers, name="num_layers")
        )

        if selected > available:
            raise ValueError(
                f"Requested {selected} MACE layers, "
                f"but the model contains only {available}."
            )

    elif num_layers is None:
        raise ValueError(
            "Could not infer the number of MACE interaction layers. "
            "Pass `num_layers` explicitly."
        )

    else:
        selected = to_int(
            num_layers,
            name="num_layers",
        )

    if selected <= 0:
        raise ValueError(
            "`num_layers` must be positive."
        )

    return selected


def _infer_descriptor_layout(
    model: nn.Module,
) -> Tuple[int, int]:
    """Infer ``num_features`` and ``l_max``."""

    try:
        irreps = model.products[0].linear.irreps_out
        descriptor_dim = int(irreps.dim)
        l_max = int(irreps.lmax)

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
            "The MACE descriptor dimension must be positive."
        )

    if l_max < 0:
        raise ValueError(
            "`l_max` must be non-negative."
        )

    angular_size = (l_max + 1) ** 2

    if descriptor_dim % angular_size != 0:
        raise ValueError(
            "The MACE descriptor dimension is incompatible "
            f"with l_max={l_max}."
        )

    return descriptor_dim // angular_size, l_max


def _resolve_descriptor_layout(
    model: nn.Module,
    num_features: Optional[int],
    l_max: Optional[int],
) -> Tuple[int, int]:
    """Resolve the MACE descriptor layout."""

    if num_features is None or l_max is None:
        inferred_features, inferred_lmax = (
            _infer_descriptor_layout(model)
        )

        if num_features is None:
            num_features = inferred_features

        if l_max is None:
            l_max = inferred_lmax

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
            "`num_features` must be positive."
        )

    if l_max < 0:
        raise ValueError(
            "`l_max` must be non-negative."
        )

    return num_features, l_max


class MACERepresentation(GraphRepresentation):
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
        freeze: bool = True,
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
            model,
            num_layers,
        )

        num_features, l_max = _resolve_descriptor_layout(
            model,
            num_features,
            l_max,
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
            output_kind="atom",
            buffer=buffer,
            long_range_cutoff=long_range_cutoff,
            full_neighbor_list=True,
            freeze=freeze,
        )

        self.num_layers = num_layers
        self.num_features = num_features
        self.l_max = l_max

        self.layer_size = (
            (l_max + 1) ** 2
            * num_features
        )

        self.required_input_features = (
            (num_layers - 1)
            * self.layer_size
            + num_features
        )

        self.model = model
        self._freeze_module(self.model)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract invariant atom-level MACE features."""

        del cell

        output = self.model(
            data,
            training=(
                self.training
                and not self.freeze
            ),
            compute_force=False,
        )

        if "node_feats" not in output:
            raise RuntimeError(
                "The MACE model output does not contain `node_feats`."
            )

        node_features = output["node_feats"]

        if node_features is None:
            raise RuntimeError(
                "The MACE model returned `node_feats=None`."
            )

        return self._extract_invariant_features(
            node_features
        )

    def _extract_invariant_features(
        self,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """Extract scalar features from selected MACE layers."""

        blocks = torch.jit.annotate(
            List[torch.Tensor],
            [],
        )

        for i in range(self.num_layers):
            start = i * self.layer_size
            end = start + self.num_features

            blocks.append(
                node_features[:, start:end]
            )

        return torch.cat(
            blocks,
            dim=-1,
        )

    @torch.jit.unused
    def prepare_for_torchscript(
        self,
    ) -> None:
        """Prepare the native MACE model for TorchScript export."""

        if isinstance(
            self.model,
            torch.jit.ScriptModule,
        ):
            return

        try:
            from e3nn.util.jit import (
                script as e3nn_script,
            )
        except ImportError as exc:
            raise ImportError(
                "Exporting a MACE representation requires e3nn."
            ) from exc

        self.model = e3nn_script(
            self.model.eval(),
            in_place=False,
        )