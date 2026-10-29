
import torch
from torch import nn

from ._utils import as_float, as_int, as_positive_int
from .base import Representation

__all__ = ["MACERepresentation"]


def _descriptor_layout(
    model: nn.Module,
    num_features: int | None,
    l_max: int | None,
) -> tuple[int, int]:
    """Resolve the number of invariant channels and maximum angular degree."""
    if num_features is None or l_max is None:
        try:
            irreps = model.products[0].linear.irreps_out
            inferred_lmax = int(irreps.lmax)
            dim = int(irreps.dim)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                "Cannot infer MACE descriptor layout from "
                "`model.products[0].linear.irreps_out`."
            ) from exc

        angular_size = (inferred_lmax + 1) ** 2
        if inferred_lmax < 0 or dim <= 0 or dim % angular_size:
            raise ValueError("Invalid MACE descriptor layout.")

        if num_features is None:
            num_features = dim // angular_size
        if l_max is None:
            l_max = inferred_lmax

    num_features = as_positive_int(num_features, "num_features")
    l_max = as_int(l_max, "l_max")

    if l_max < 0:
        raise ValueError("`l_max` must be non-negative.")

    return num_features, l_max


class MACERepresentation(Representation):
    """Extract invariant features from a pretrained MACE model.

    Parameters
    ----------
    model : nn.Module
        Pretrained MACE model.
    pooling_operation : {"mean", "sum"} or None, optional
        Pool atom-level features into system-level features.
    num_layers : int, optional
        Number of interaction layers. Defaults to all layers.
    num_features : int, optional
        Number of invariant channels per layer, inferred by default.
    l_max : int, optional
        Maximum angular degree, inferred by default.
    freeze : bool, default=True
        Freeze pretrained model parameters.
    """

    __constants__ = ["required_input_features"]

    def __init__(
        self,
        model: nn.Module,
        *,
        pooling_operation: str | None = None,
        num_layers: int | None = None,
        num_features: int | None = None,
        l_max: int | None = None,
        freeze: bool = True,
    ) -> None:
        if not all(
            hasattr(model, attr)
            for attr in ("atomic_numbers", "r_max", "num_interactions")
        ):
            raise ValueError(
                "MACE model must expose `atomic_numbers`, "
                "`r_max`, and `num_interactions`."
            )

        available = as_positive_int(
            model.num_interactions, "model.num_interactions"
        )
        num_layers = (
            available if num_layers is None
            else as_positive_int(num_layers, "num_layers")
        )
        if num_layers > available:
            raise ValueError(
                f"Requested {num_layers} layers, but only {available} are available."
            )

        num_features, l_max = _descriptor_layout(
            model, num_features, l_max
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
            output_kind="atom" if pooling_operation is None else "system",
            atomic_numbers=model.atomic_numbers,
            cutoff=as_float(model.r_max, "model.r_max"),
            pooling_operation=pooling_operation,
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
        """Return selected invariant MACE features."""
        del cell

        output = self.model(
            data,
            training=self.training and not self.freeze,
            compute_force=False,
        )
        features = output.get("node_feats")

        if features is None or features.ndim != 2:
            raise RuntimeError("MACE must return a 2D `node_feats` tensor.")
        if features.shape[1] < self.required_input_features:
            raise RuntimeError(
                "MACE `node_feats` is incompatible with the configured "
                "descriptor layout."
            )

        features = features.index_select(1, self.feature_indices)
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
