from collections.abc import Sequence

import torch
from torch import nn

from mlcolvar.utils import _code

from ._utils import _as_atomic_number_list, align_node_attrs, as_positive_int

__all__ = ["Representation"]


class Representation(nn.Module):
    """Base class for reusable vector- or graph-based representations.

    A representation maps raw inputs to a latent feature space that can be reused by downstream
    collective-variable models. ``input_kind`` describes the input format, while ``output_kind``
    specifies whether the representation returns atom- or system-level features.

    Parameters
    ----------
    out_features : int
        Number of output features, per atom for atom-level representations.
    input_kind : {"vector", "graph"}
        Type of input consumed by the representation.
    output_kind : {"atom", "system"}
        Whether the representation returns atom- or system-level features.
    in_features : int, optional
        Number of input features for vector representations. Required for ``input_kind="vector"``.
    atomic_numbers : Sequence[int] or torch.Tensor, optional
        Atomic numbers supported by graph representations. Required for ``input_kind="graph"``.
    cutoff : float, optional
        Short-range cutoff used by graph representations. Required for ``input_kind="graph"``.
    pooling_operation : {"mean", "sum"} or None, optional
        Optional aggregation of atom-level graph features.
    buffer : float, optional
        Neighbor-list buffer for graph representations. Default is ``0.0``.
    long_range_cutoff : float, optional
        Optional long-range cutoff. Negative values disable it. Default is ``-1.0``.
    freeze : bool, optional
        Whether wrapped pretrained modules are frozen and kept in evaluation mode. Default is ``True``.
    """

    __constants__ = [
        "input_kind",
        "output_kind",
        "in_features",
        "out_features",
        "pooling_operation",
        "freeze",
    ]

    def __init__(
        self,
        *,
        out_features: int,
        input_kind: str,
        output_kind: str,
        in_features: int | None = None,
        atomic_numbers: Sequence[int] | torch.Tensor | None = None,
        cutoff: float | None = None,
        pooling_operation: str | None = None,
        buffer: float = 0.0,
        long_range_cutoff: float = -1.0,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        if input_kind not in {"vector", "graph"}:
            raise ValueError("`input_kind` must be 'vector' or 'graph'.")
        if output_kind not in {"atom", "system"}:
            raise ValueError("`output_kind` must be 'atom' or 'system'.")
        if pooling_operation not in {None, "mean", "sum"}:
            raise ValueError("`pooling_operation` must be 'mean', 'sum', or None.")

        if input_kind == "vector":
            if in_features is None:
                raise ValueError("`in_features` is required for vector inputs.")
            if output_kind != "system":
                raise ValueError("Vector representations must return system-level features.")
            if pooling_operation is not None:
                raise ValueError("`pooling_operation` is only available for graph inputs.")
        else:
            if atomic_numbers is None or cutoff is None:
                raise ValueError("`atomic_numbers` and `cutoff` are required for graph inputs.")
            atomic_numbers = _as_atomic_number_list(atomic_numbers, "representation")
            if cutoff <= 0:
                raise ValueError("`cutoff` must be positive.")
            if buffer < 0:
                raise ValueError("`buffer` must be non-negative.")
            if 0 <= long_range_cutoff <= cutoff:
                raise ValueError("`long_range_cutoff` must be negative or larger than `cutoff`.")
            if pooling_operation is not None and output_kind != "system":
                raise ValueError("Graph pooling requires `output_kind='system'`.")

        self.input_kind = input_kind
        self.output_kind = output_kind
        self.in_features = (
            as_positive_int(in_features, "in_features")
            if in_features is not None
            else None
        )
        self.out_features = as_positive_int(out_features, "out_features")
        self.pooling_operation = pooling_operation
        self.freeze = bool(freeze)

        if input_kind == "graph":
            dtype = torch.get_default_dtype()
            self.register_buffer(
                "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.long)
            )
            self.register_buffer("cutoff", torch.tensor(cutoff, dtype=dtype))
            self.register_buffer("buffer", torch.tensor(buffer, dtype=dtype))
            self.register_buffer(
                "long_range_cutoff",
                torch.tensor(long_range_cutoff, dtype=dtype),
            )

    def _freeze_module(self, module: nn.Module) -> None:
        """Freeze a wrapped module when requested."""
        if not self.freeze:
            return
        if not isinstance(module, torch.jit.ScriptModule):
            module.requires_grad_(False)
        module.eval()

    def train(self, mode: bool = True):
        """Set the representation training mode."""
        return super().train(False if self.freeze else mode)

    def pooling(
        self,
        features: torch.Tensor,
        data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pool atom-level graph features over each system."""
        if self.pooling_operation is None:
            return features

        if "system_masks" in data:
            features = features * data["system_masks"]

        if self.pooling_operation == "mean":
            if "system_masks" not in data:
                return _code.scatter_mean(features, data["batch"], dim=0)
            return _code.scatter_sum(features, data["batch"], dim=0) / data["n_system"]

        return _code.scatter_sum(features, data["batch"], dim=0)

    @torch.jit.unused
    def align_dataset(self, dataset):
        """Align graph node attributes with the representation species."""
        if self.input_kind != "graph":
            return dataset
        return align_node_attrs(dataset, self.atomic_numbers)