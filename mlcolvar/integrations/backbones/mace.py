"""MACE integration for pretrained atomistic representations."""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from mlcolvar.integrations.atomistic import BaseAtomisticBackbone

from ._utils import to_float, to_int, to_int_list


__all__ = ["MACEBackbone"]


def _infer_num_layers(model: nn.Module) -> int:
    """Infer the number of interaction layers from a MACE model.

    Parameters
    ----------
    model
        Pretrained MACE model exposing ``num_interactions``.

    Returns
    -------
    int
        Number of interaction layers.

    Raises
    ------
    ValueError
        If the model does not expose a valid positive
        ``num_interactions`` value.
    """
    if not hasattr(model, "num_interactions"):
        raise ValueError(
            "Could not infer the number of MACE interaction layers "
            "because the model does not expose `num_interactions`."
        )

    num_layers = to_int(
        model.num_interactions,
        name="model.num_interactions",
    )

    if num_layers <= 0:
        raise ValueError(
            "MACE `num_interactions` must be positive, "
            f"found {num_layers}."
        )

    return num_layers


def _infer_descriptor_layout(model: nn.Module) -> Tuple[int, int]:
    """Infer the layout of the MACE node descriptors.

    MACE stores the equivariant representation associated with an
    interaction layer in::

        model.products[0].linear.irreps_out

    For a representation containing angular momenta from ``l = 0`` to
    ``l = l_max``, each feature channel contains

    .. math::

        \\sum_{l=0}^{l_{\\max}} (2l + 1)
        = (l_{\\max} + 1)^2

    angular components.

    Parameters
    ----------
    model
        Pretrained MACE model.

    Returns
    -------
    num_features
        Number of feature channels associated with each angular component.
    l_max
        Maximum angular momentum represented in the descriptor.

    Raises
    ------
    ValueError
        If the descriptor layout cannot be inferred or is inconsistent.
    """
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
            "`model.products[0].linear.irreps_out`. Pass both "
            "`num_features` and `l_max` explicitly."
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

    num_features = descriptor_dim // angular_size

    if num_features <= 0:
        raise ValueError(
            "Could not infer a positive number of MACE feature "
            f"channels, found {num_features}."
        )

    return num_features, l_max


def _resolve_num_layers(
    model: nn.Module,
    num_layers: Optional[int],
) -> int:
    """Resolve and validate the number of selected MACE layers."""
    if hasattr(model, "num_interactions"):
        available_layers = _infer_num_layers(model)

        if num_layers is None:
            selected_layers = available_layers
        else:
            selected_layers = to_int(
                num_layers,
                name="num_layers",
            )

        if selected_layers > available_layers:
            raise ValueError(
                f"Requested {selected_layers} MACE layers, but the "
                f"model contains only {available_layers}."
            )
    else:
        if num_layers is None:
            raise ValueError(
                "Could not infer the number of MACE interaction layers "
                "because the model does not expose `num_interactions`. "
                "Pass `num_layers` explicitly."
            )

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


def _resolve_descriptor_layout(
    model: nn.Module,
    num_features: Optional[int],
    l_max: Optional[int],
) -> Tuple[int, int]:
    """Resolve and validate the MACE descriptor layout.

    When the layout can be inferred from the model, explicitly provided
    values are checked for consistency. If inference is unavailable, both
    values must be provided explicitly.
    """
    provided_num_features = (
        None
        if num_features is None
        else to_int(
            num_features,
            name="num_features",
        )
    )

    provided_l_max = (
        None
        if l_max is None
        else to_int(
            l_max,
            name="l_max",
        )
    )

    try:
        inferred_num_features, inferred_l_max = (
            _infer_descriptor_layout(model)
        )
    except ValueError:
        if provided_num_features is None or provided_l_max is None:
            raise

        resolved_num_features = provided_num_features
        resolved_l_max = provided_l_max
    else:
        if (
            provided_num_features is not None
            and provided_num_features != inferred_num_features
        ):
            raise ValueError(
                "`num_features` does not match the MACE descriptor "
                f"layout: expected {inferred_num_features}, "
                f"found {provided_num_features}."
            )

        if (
            provided_l_max is not None
            and provided_l_max != inferred_l_max
        ):
            raise ValueError(
                "`l_max` does not match the MACE descriptor layout: "
                f"expected {inferred_l_max}, "
                f"found {provided_l_max}."
            )

        resolved_num_features = (
            inferred_num_features
            if provided_num_features is None
            else provided_num_features
        )

        resolved_l_max = (
            inferred_l_max
            if provided_l_max is None
            else provided_l_max
        )

    if resolved_num_features <= 0:
        raise ValueError(
            "`num_features` must be positive, "
            f"found {resolved_num_features}."
        )

    if resolved_l_max < 0:
        raise ValueError(
            "`l_max` must be non-negative, "
            f"found {resolved_l_max}."
        )

    return resolved_num_features, resolved_l_max


class MACEBackbone(BaseAtomisticBackbone):
    """Extract invariant atom-level features from a pretrained MACE model.

    The wrapped MACE or ScaleShiftMACE model must return ``node_feats`` in
    its output dictionary. The leading scalar block from each selected
    interaction layer is extracted and returned through the common
    atomistic-backbone interface.

    This implementation assumes that each complete interaction-layer
    representation stores its ``l = 0`` scalar channels first, followed by
    the higher-order equivariant components.

    Pooling, masking, parameter freezing, and graph-level output are handled
    by :class:`mlcolvar.integrations.atomistic.AtomisticFeaturizer`.

    Parameters
    ----------
    model
        Pretrained MACE or ScaleShiftMACE model.
    num_layers
        Number of consecutive interaction layers to use, starting from the
        first layer. All available layers are used when omitted. This value
        must be provided when the model does not expose
        ``num_interactions``.
    num_features
        Number of scalar invariant channels contributed by each interaction
        layer. It is inferred from the MACE model when omitted.
    l_max
        Maximum angular momentum in the MACE descriptor layout. It is
        inferred from the MACE model when omitted.
    buffer
        Additional environment buffer used during graph construction.
    long_range_cutoff
        Optional cutoff for long-range edges. A negative value disables
        long-range edges.
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
        num_layers: Optional[int] = None,
        num_features: Optional[int] = None,
        l_max: Optional[int] = None,
        buffer: float = 0.0,
        long_range_cutoff: float = -1.0,
    ) -> None:
        selected_layers = _resolve_num_layers(
            model=model,
            num_layers=num_layers,
        )

        resolved_num_features, resolved_l_max = (
            _resolve_descriptor_layout(
                model=model,
                num_features=num_features,
                l_max=l_max,
            )
        )

        if not hasattr(model, "atomic_numbers"):
            raise ValueError(
                "The MACE model does not expose `atomic_numbers`."
            )

        if not hasattr(model, "r_max"):
            raise ValueError(
                "The MACE model does not expose `r_max`."
            )

        atomic_numbers = to_int_list(
            model.atomic_numbers,
            name="model.atomic_numbers",
        )

        cutoff = to_float(
            model.r_max,
            name="model.r_max",
        )

        # Each selected layer contributes one scalar invariant block.
        out_features = selected_layers * resolved_num_features

        # Initialize nn.Module before registering the MACE model as a child
        # module.
        super().__init__(
            out_features=out_features,
            atomic_numbers=atomic_numbers,
            cutoff=cutoff,
            sample_kind="atom",
            buffer=buffer,
            long_range_cutoff=long_range_cutoff,
            full_neighbor_list=True,
        )

        self.num_layers = selected_layers
        self.num_features = resolved_num_features
        self.l_max = resolved_l_max

        # A complete equivariant feature channel contains
        #
        #     sum_l (2l + 1) = (l_max + 1)^2
        #
        # angular components.
        self.layer_size = (
            (self.l_max + 1) ** 2
            * self.num_features
        )

        # Earlier interaction layers contain complete equivariant
        # representations, whereas the final layer may contain only its
        # scalar invariant block.
        self.required_input_features = (
            (self.num_layers - 1) * self.layer_size
            + self.num_features
        )

        self.model = model

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute invariant atom-level MACE features.

        Parameters
        ----------
        data
            mlcolvar graph dictionary containing the fields required by
            MACE, typically ``positions``, ``node_attrs``, ``edge_index``,
            ``shifts``, ``batch``, ``ptr``, and ``cell``.
        cell
            Optional cell passed separately by the mlcolvar CV interface.
            MACE expects the cell to be stored in the graph dictionary, so
            this argument is accepted only for interface compatibility.

        Returns
        -------
        torch.Tensor
            Atom-level invariant features with shape
            ``[n_atoms, num_layers * num_features]``.
        """
        # MACE reads the simulation cell directly from the graph dictionary.
        _ = cell

        output = self.model(
            data,
            training=self.training,
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

        # Avoid tensor-to-Python shape conversions while scripting or
        # tracing.
        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
        ):
            self._validate_node_features(node_features)

        return self._extract_invariant_features(node_features)

    @torch.jit.unused
    def _validate_node_features(
        self,
        node_features: torch.Tensor,
    ) -> None:
        """Validate the MACE node-feature tensor in eager mode."""
        if not isinstance(node_features, torch.Tensor):
            raise RuntimeError(
                "MACE `node_feats` must be a torch.Tensor."
            )

        if node_features.dim() != 2:
            raise RuntimeError(
                "Expected MACE `node_feats` to be a rank-2 tensor "
                "with shape [n_atoms, n_features], but found shape "
                f"{tuple(node_features.shape)}."
            )

        if node_features.size(1) < self.required_input_features:
            raise RuntimeError(
                "The MACE `node_feats` tensor is too small for the "
                "requested descriptor layout: expected at least "
                f"{self.required_input_features} features, but found "
                f"{node_features.size(1)}. Check `num_layers`, "
                "`num_features`, and `l_max`."
            )

    def _extract_invariant_features(
        self,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the leading scalar block from each selected MACE layer.

        The descriptor layout is assumed to place the ``l = 0`` scalar
        channels at the beginning of every complete layer representation.
        """
        invariant_blocks = torch.jit.annotate(
            List[torch.Tensor],
            [],
        )

        for layer_index in range(self.num_layers):
            start = layer_index * self.layer_size
            end = start + self.num_features

            invariant_blocks.append(
                node_features[:, start:end]
            )

        return torch.cat(
            invariant_blocks,
            dim=-1,
        )