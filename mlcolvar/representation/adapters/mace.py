from typing import List

import torch
from torch import nn

from ..base import Representation
from ._utils import to_float, to_int, to_int_list

__all__ = ["MACERepresentation"]


def _resolve_num_layers(
    model: nn.Module,
    num_layers: int | None,
) -> int:
    """Resolve the number of MACE interaction layers used by the representation."""
    if hasattr(model, "num_interactions"):
        available = to_int(
            model.num_interactions,
            name="model.num_interactions",
        )
        if available <= 0:
            raise ValueError("MACE `num_interactions` must be positive.")

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
        selected = to_int(num_layers, name="num_layers")

    if selected <= 0:
        raise ValueError("`num_layers` must be positive.")
    return selected


def _infer_descriptor_layout(model: nn.Module) -> tuple[int, int]:
    """Infer the scalar feature count and maximum angular degree from MACE."""
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
        raise ValueError("The MACE descriptor dimension must be positive.")
    if l_max < 0:
        raise ValueError("`l_max` must be non-negative.")

    angular_size = (l_max + 1) ** 2
    if descriptor_dim % angular_size:
        raise ValueError(
            "The MACE descriptor dimension is incompatible "
            f"with l_max={l_max}."
        )

    return descriptor_dim // angular_size, l_max


def _resolve_descriptor_layout(
    model: nn.Module,
    num_features: int | None,
    l_max: int | None,
) -> tuple[int, int]:
    """Resolve the number of scalar features and maximum angular degree."""
    if num_features is None or l_max is None:
        inferred_features, inferred_lmax = _infer_descriptor_layout(model)
        num_features = (
            inferred_features
            if num_features is None
            else num_features
        )
        l_max = inferred_lmax if l_max is None else l_max

    num_features = to_int(num_features, name="num_features")
    l_max = to_int(l_max, name="l_max")

    if num_features <= 0:
        raise ValueError("`num_features` must be positive.")
    if l_max < 0:
        raise ValueError("`l_max` must be non-negative.")

    return num_features, l_max


class MACERepresentation(Representation):
    """Extract invariant features from a pretrained MACE model.

    Scalar features are collected from selected MACE interaction layers and
    concatenated into a reusable atom-level representation. Only invariant
    scalar components are retained. Optional pooling reduces atom-level
    features to a system-level representation.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained MACE model exposing ``atomic_numbers``, ``r_max``, and
        interaction-layer node features through ``node_feats``.
    pooling_operation : {"mean", "sum"}, optional
        Optional atom-to-system reduction. If ``None``, atom-level features
        are returned.
    num_layers : int, optional
        Number of MACE interaction layers used. If omitted, inferred from
        ``model.num_interactions``.
    num_features : int, optional
        Number of invariant scalar features extracted from each selected layer.
    l_max : int, optional
        Maximum angular degree of the MACE feature layout.
    buffer : float, default=0.0
        Buffer added to the graph neighbor-list cutoff.
    long_range_cutoff : float, default=-1.0
        Optional long-range interaction cutoff.
    freeze : bool, default=True
        Whether to freeze the pretrained MACE model.

    Notes
    -----
    The output dimension before pooling is ``num_layers * num_features``.
    """

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
        *,
        pooling_operation: str | None = None,
        num_layers: int | None = None,
        num_features: int | None = None,
        l_max: int | None = None,
        buffer: float = 0.0,
        long_range_cutoff: float = -1.0,
        freeze: bool = True,
    ) -> None:
        if not hasattr(model, "atomic_numbers"):
            raise ValueError(
                "The MACE model does not expose `atomic_numbers`."
            )
        if not hasattr(model, "r_max"):
            raise ValueError("The MACE model does not expose `r_max`.")

        num_layers = _resolve_num_layers(model, num_layers)
        num_features, l_max = _resolve_descriptor_layout(
            model,
            num_features,
            l_max,
        )

        super().__init__(
            out_features=num_layers * num_features,
            input_kind="graph",
            atomic_numbers=to_int_list(
                model.atomic_numbers,
                name="model.atomic_numbers",
            ),
            cutoff=to_float(model.r_max, name="model.r_max"),
            pooling_operation=pooling_operation,
            buffer=buffer,
            long_range_cutoff=long_range_cutoff,
            full_neighbor_list=True,
            freeze=freeze,
        )

        self.num_layers = num_layers
        self.num_features = num_features
        self.l_max = l_max
        self.layer_size = (l_max + 1) ** 2 * num_features
        self.required_input_features = (
            (num_layers - 1) * self.layer_size + num_features
        )

        self.model = model
        self._freeze_module(model)

    def forward(
        self,
        data: dict[str, torch.Tensor],
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the MACE representation."""
        del cell

        output = self.model(
            data,
            training=self.training and not self.freeze,
            compute_force=False,
        )
        if "node_feats" not in output:
            raise RuntimeError(
                "The MACE model output does not contain `node_feats`."
            )

        node_features = output["node_feats"]
        if node_features is None:
            raise RuntimeError("The MACE model returned `node_feats=None`.")

        features = self._extract_invariant_features(node_features)
        return self.pooling(features, data)

    def _extract_invariant_features(
        self,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """Extract invariant scalar components from selected MACE layers."""
        if node_features.dim() != 2:
            raise RuntimeError(
                "MACE `node_feats` must be a rank-two tensor."
            )
        if node_features.size(1) < self.required_input_features:
            raise RuntimeError(
                "MACE `node_feats` is incompatible with the configured "
                "descriptor layout: expected at least "
                f"{self.required_input_features} features, found "
                f"{node_features.size(1)}."
            )

        blocks = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(self.num_layers):
            start = i * self.layer_size
            blocks.append(
                node_features[:, start : start + self.num_features]
            )

        return torch.cat(blocks, dim=-1)

    @torch.jit.unused
    def prepare_for_torchscript(self) -> None:
        """Prepare the wrapped MACE model for TorchScript export."""
        if isinstance(self.model, torch.jit.ScriptModule):
            return

        try:
            from e3nn.util.jit import script as e3nn_script
        except ImportError as exc:
            raise ImportError(
                "Exporting a MACE representation requires e3nn."
            ) from exc

        self.model = e3nn_script(
            self.model.eval(),
            in_place=False,
        )