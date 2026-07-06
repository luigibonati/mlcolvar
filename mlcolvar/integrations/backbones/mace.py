from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from mlcolvar.integrations.atomistic import BaseAtomisticBackbone


__all__ = ["MACEBackbone"]


def _to_int(
    value,
    name: str,
) -> int:
    """Convert a scalar tensor or Python scalar to an integer."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{name} must contain exactly one value."
            )

        return int(
            value.detach().cpu().item()
        )

    return int(value)


def _to_float(
    value,
    name: str,
) -> float:
    """Convert a scalar tensor or Python scalar to a float."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{name} must contain exactly one value."
            )

        return float(
            value.detach().cpu().item()
        )

    return float(value)


def _to_atomic_numbers(
    value,
) -> List[int]:
    """Convert MACE atomic-number metadata to a Python list."""
    if isinstance(value, torch.Tensor):
        values = (
            value.detach()
            .cpu()
            .reshape(-1)
            .tolist()
        )
    else:
        values = list(value)

    return [
        int(number)
        for number in values
    ]


def _infer_num_layers(
    model: nn.Module,
) -> int:
    """Infer the number of interaction layers from a MACE model."""
    if not hasattr(model, "num_interactions"):
        raise ValueError(
            "Could not infer the number of MACE interaction layers "
            "because the model does not expose `num_interactions`. "
            "Pass `num_layers` explicitly."
        )

    num_layers = _to_int(
        model.num_interactions,
        name="model.num_interactions",
    )

    if num_layers <= 0:
        raise ValueError(
            "MACE `num_interactions` must be positive, "
            f"found {num_layers}."
        )

    return num_layers


def _infer_descriptor_layout(
    model: nn.Module,
) -> Tuple[int, int]:
    """Infer the scalar-channel layout of MACE node descriptors.

    MACE stores the equivariant descriptor representation in

    ``model.products[0].linear.irreps_out``.

    Returns
    -------
    num_features
        Number of feature channels associated with each angular component.
    l_max
        Maximum angular momentum of the representation.
    """
    try:
        irreps_out = (
            model.products[0]
            .linear
            .irreps_out
        )

        descriptor_dim = int(
            irreps_out.dim
        )

        l_max = int(
            irreps_out.lmax
        )

    except (
        AttributeError,
        IndexError,
        TypeError,
    ) as exc:
        raise ValueError(
            "Could not infer the MACE descriptor layout. "
            "Pass `num_features` and `l_max` explicitly."
        ) from exc

    angular_size = (
        l_max + 1
    ) ** 2

    if descriptor_dim % angular_size != 0:
        raise ValueError(
            "The MACE descriptor dimension is incompatible with "
            f"`l_max`: descriptor_dim={descriptor_dim}, "
            f"l_max={l_max}."
        )

    num_features = (
        descriptor_dim
        // angular_size
    )

    if num_features <= 0:
        raise ValueError(
            "Could not infer a positive number of MACE feature "
            f"channels, found {num_features}."
        )

    return num_features, l_max


class MACEBackbone(BaseAtomisticBackbone):
    """Extract invariant atom-level features from a pretrained MACE model.

    The wrapped MACE model must return ``node_feats`` in its output
    dictionary. Scalar invariant channels from the selected interaction
    layers are extracted and returned using the common atomistic-backbone
    interface.

    Pooling, masking, parameter freezing, and graph-level output are handled
    by :class:`mlcolvar.integrations.atomistic.AtomisticFeaturizer`.

    Parameters
    ----------
    model
        Pretrained MACE or ScaleShiftMACE model.
    num_layers
        Number of MACE interaction layers to use. All available layers are
        used when omitted.
    num_features
        Number of scalar invariant channels per interaction layer. It is
        inferred from the MACE model when omitted.
    l_max
        Maximum angular momentum in the MACE descriptor layout. It is
        inferred from the MACE model when omitted.
    buffer
        Additional environment buffer used by graph construction.
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
        # ============================================================
        # Infer and validate the selected MACE interaction layers
        # ============================================================
        available_layers = _infer_num_layers(
            model
        )

        if num_layers is None:
            selected_layers = available_layers
        else:
            selected_layers = int(
                num_layers
            )

        if selected_layers <= 0:
            raise ValueError(
                "`num_layers` must be positive, "
                f"found {selected_layers}."
            )

        if selected_layers > available_layers:
            raise ValueError(
                f"Requested {selected_layers} MACE layers, but the "
                f"model contains only {available_layers}."
            )

        # ============================================================
        # Infer and validate the descriptor layout
        # ============================================================
        if (
            num_features is None
            or l_max is None
        ):
            (
                inferred_features,
                inferred_l_max,
            ) = _infer_descriptor_layout(
                model
            )

            if num_features is None:
                num_features = inferred_features

            if l_max is None:
                l_max = inferred_l_max

        num_features = int(
            num_features
        )

        l_max = int(
            l_max
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

        # ============================================================
        # Read metadata required by the shared graph interface
        # ============================================================
        if not hasattr(
            model,
            "atomic_numbers",
        ):
            raise ValueError(
                "The MACE model does not expose `atomic_numbers`."
            )

        if not hasattr(
            model,
            "r_max",
        ):
            raise ValueError(
                "The MACE model does not expose `r_max`."
            )

        atomic_numbers = _to_atomic_numbers(
            model.atomic_numbers
        )

        cutoff = _to_float(
            model.r_max,
            name="model.r_max",
        )

        # Each selected interaction layer contributes one scalar block.
        out_features = (
            selected_layers
            * num_features
        )

        # Initialize nn.Module before registering the MACE child module.
        super().__init__(
            out_features=out_features,
            atomic_numbers=atomic_numbers,
            cutoff=cutoff,
            sample_kind="atom",
            buffer=buffer,
            long_range_cutoff=long_range_cutoff,
            full_neighbor_list=True,
        )

        # ============================================================
        # Store MACE-specific descriptor metadata
        # ============================================================
        self.num_layers = selected_layers
        self.num_features = num_features
        self.l_max = l_max

        # For one feature channel, the complete equivariant
        # representation contains
        #
        # sum_l (2l + 1) = (l_max + 1)^2
        #
        # components.
        self.layer_size = (
            (self.l_max + 1) ** 2
            * self.num_features
        )

        # Earlier layers contain complete equivariant representations.
        # The final layer may contain only the scalar invariant block.
        self.required_input_features = (
            (self.num_layers - 1)
            * self.layer_size
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
            ``shifts``, ``batch``, ``ptr`` and ``cell``.
        cell
            Optional cell passed separately by the mlcolvar CV interface.
            It is ignored because MACE reads the cell from the graph
            dictionary.

        Returns
        -------
        torch.Tensor
            Atom-level invariant features with shape

            ``[n_atoms, num_layers * num_features]``.
        """
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

        node_features = output[
            "node_feats"
        ]

        if node_features is None:
            raise RuntimeError(
                "The MACE model returned `node_feats=None`."
            )

        # Skip Python shape checks during TorchScript compilation and
        # tracing to avoid tensor-to-Python conversion warnings.
        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
        ):
            if node_features.dim() != 2:
                raise RuntimeError(
                    "Expected MACE `node_feats` to be a rank-2 "
                    "tensor with shape [n_atoms, n_features]."
                )

            if (
                node_features.size(1)
                < self.required_input_features
            ):
                raise RuntimeError(
                    "The MACE `node_feats` tensor is too small for "
                    "the requested descriptor layout. Check "
                    "`num_layers`, `num_features`, and `l_max`."
                )

        return self._extract_invariant_features(
            node_features
        )

    def _extract_invariant_features(
        self,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """Extract scalar invariant channels from selected MACE layers."""
        invariant_blocks = torch.jit.annotate(
            List[torch.Tensor],
            [],
        )

        for layer_index in range(
            self.num_layers
        ):
            start = (
                layer_index
                * self.layer_size
            )

            end = (
                start
                + self.num_features
            )

            invariant_blocks.append(
                node_features[
                    :,
                    start:end,
                ]
            )

        return torch.cat(
            invariant_blocks,
            dim=-1,
        )