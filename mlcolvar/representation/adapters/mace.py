import torch
from torch import nn

from .._utils import as_float, as_int, as_positive_int
from ..base import Representation

__all__ = ["MACERepresentation"]


def _infer_descriptor_layout(model: nn.Module) -> tuple[int, int]:
    """Infer the invariant feature count and maximum angular degree."""
    try:
        irreps = model.products[0].linear.irreps_out
        descriptor_dim = int(irreps.dim)
        l_max = int(irreps.lmax)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "Could not infer the MACE descriptor layout from "
            "`model.products[0].linear.irreps_out`."
        ) from exc

    if descriptor_dim <= 0 or l_max < 0:
        raise ValueError("Invalid MACE descriptor layout.")

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
    """Resolve the invariant feature count and maximum angular degree."""
    if num_features is None or l_max is None:
        inferred_features, inferred_lmax = _infer_descriptor_layout(model)
        num_features = inferred_features if num_features is None else num_features
        l_max = inferred_lmax if l_max is None else l_max

    num_features = as_positive_int(num_features, "num_features")
    l_max = as_int(l_max, "l_max")
    if l_max < 0:
        raise ValueError("`l_max` must be non-negative.")
    return num_features, l_max


class MACERepresentation(Representation):
    """Extract invariant features from a pretrained MACE model."""

    __constants__ = ["required_input_features"]

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
        if not hasattr(model, "atomic_numbers") or not hasattr(model, "r_max"):
            raise ValueError(
                "The MACE model must expose `atomic_numbers` and `r_max`."
            )
        if not hasattr(model, "num_interactions"):
            raise ValueError(
                "The MACE model must expose `num_interactions`."
            )

        available = as_positive_int(
            model.num_interactions,
            "model.num_interactions",
        )
        num_layers = (
            available
            if num_layers is None
            else as_positive_int(num_layers, "num_layers")
        )
        if num_layers > available:
            raise ValueError(
                f"Requested {num_layers} MACE layers, "
                f"but the model contains only {available}."
            )

        num_features, l_max = _resolve_descriptor_layout(
            model,
            num_features,
            l_max,
        )
        layer_size = (l_max + 1) ** 2 * num_features
        indices = [
            i * layer_size + j
            for i in range(num_layers)
            for j in range(num_features)
        ]

        super().__init__(
            out_features=len(indices),
            input_kind="graph",
            atomic_numbers=model.atomic_numbers,
            cutoff=as_float(model.r_max, "model.r_max"),
            pooling_operation=pooling_operation,
            buffer=buffer,
            long_range_cutoff=long_range_cutoff,
            freeze=freeze,
        )

        self.required_input_features = indices[-1] + 1
        self.register_buffer(
            "feature_indices",
            torch.tensor(indices, dtype=torch.long),
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
        if "node_feats" not in output or output["node_feats"] is None:
            raise RuntimeError(
                "The MACE model output does not contain valid `node_feats`."
            )

        node_features = output["node_feats"]
        if node_features.dim() != 2:
            raise RuntimeError("MACE `node_feats` must be a rank-two tensor.")
        if node_features.size(1) < self.required_input_features:
            raise RuntimeError(
                "MACE `node_feats` is incompatible with the configured "
                "descriptor layout: expected at least "
                f"{self.required_input_features} features, found "
                f"{node_features.size(1)}."
            )

        features = node_features.index_select(1, self.feature_indices)
        return self.pooling(features, data)

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
        self.model = e3nn_script(self.model.eval(), in_place=False)