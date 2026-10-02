from contextlib import contextmanager

import torch

from mlcolvar.data import DictDataset

from .._utils import module_reference_tensor
from ..base import Representation


@contextmanager
def temporary_eval(
    representation: Representation,
    device: torch.device,
):
    """Temporarily evaluate a representation on another device."""
    original_device = module_reference_tensor(representation).device
    training = representation.training

    representation.to(device).eval()
    try:
        yield
    finally:
        representation.to(original_device).train(training)


def resolve_devices(
    representation: Representation,
    device=None,
    output_device="cpu",
):
    device = torch.device(
        device or module_reference_tensor(representation).device
    )
    return device, torch.device(output_device)


def require_graph_dataset(dataset: DictDataset) -> None:
    if dataset.metadata.get("data_type") != "graphs":
        raise TypeError("Expected a graph dataset.")