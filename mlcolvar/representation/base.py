
from collections.abc import Sequence

import torch
from torch import nn

from mlcolvar.utils import _code
from ._utils import _as_atomic_number_list, align_node_attrs, as_positive_int

__all__ = ["Representation"]


class Representation(nn.Module):
    """Base class for reusable vector- and graph-based representations.

    Parameters
    ----------
    out_features : int
        Number of output features.
    input_kind : {"vector", "graph"}
        Input data type.
    output_kind : {"atom", "system"}
        Output granularity.
    in_features : int, optional
        Number of input features for vector representations.
    atomic_numbers : Sequence[int] or torch.Tensor, optional
        Supported atomic numbers for graph representations.
    cutoff : float, optional
        Graph neighbor cutoff.
    pooling_operation : {"mean", "sum"} or None, optional
        Graph-level pooling operation.
    buffer : float, default=0.0
        Neighbor-list buffer.
    long_range_cutoff : float, default=-1.0
        Optional long-range cutoff.
    freeze : bool, default=True
        Whether to freeze the pretrained representation.
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
            raise ValueError("Invalid `pooling_operation`.")

        if input_kind == "vector":
            if in_features is None:
                raise ValueError("`in_features` is required for vector inputs.")
            if output_kind != "system":
                raise ValueError("Vector outputs must be system-level.")
            if pooling_operation is not None:
                raise ValueError("Pooling requires graph inputs.")
        else:
            if atomic_numbers is None or cutoff is None:
                raise ValueError("Graph inputs require `atomic_numbers` and `cutoff`.")
            atomic_numbers = _as_atomic_number_list(
                atomic_numbers, "representation"
            )
            if cutoff <= 0 or buffer < 0:
                raise ValueError("Invalid graph cutoff or buffer.")
            if 0 <= long_range_cutoff <= cutoff:
                raise ValueError("Invalid long-range cutoff.")
            if pooling_operation is not None and output_kind != "system":
                raise ValueError("Pooling requires system-level outputs.")

        self.input_kind = input_kind
        self.output_kind = output_kind
        self.in_features = (
            as_positive_int(in_features, "in_features")
            if in_features is not None else None
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
        if self.freeze:
            if not isinstance(module, torch.jit.ScriptModule):
                module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True):
        """Keep frozen representations in evaluation mode."""
        return super().train(mode and not self.freeze)

    def pooling(
        self,
        features: torch.Tensor,
        data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pool atom-level features into system-level features."""
        if self.pooling_operation is None:
            return features

        if "system_masks" not in data and self.pooling_operation == "mean":
            return _code.scatter_mean(features, data["batch"], dim=0)

        if "system_masks" in data:
            features = features * data["system_masks"]

        output = _code.scatter_sum(features, data["batch"], dim=0)
        if self.pooling_operation == "mean":
            output = output / data["n_system"]

        return output

    @torch.jit.unused
    def align_dataset(self, dataset):
        """Align graph node attributes with the representation species."""
        if self.input_kind == "graph":
            return align_node_attrs(dataset, self.atomic_numbers)
        return dataset
