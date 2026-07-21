from typing import Sequence, Union

import torch


__all__ = ["align_node_attrs"]


def _as_atomic_number_list(
    atomic_numbers: Union[Sequence[int], torch.Tensor],
    name: str,
) -> list[int]:
    numbers = (
        torch.as_tensor(atomic_numbers, dtype=torch.long)
        .detach()
        .cpu()
        .reshape(-1)
    )

    values = [int(number) for number in numbers.tolist()]

    if len(values) == 0:
        raise ValueError(f"The {name} atomic-number table cannot be empty.")

    if any(number <= 0 for number in values):
        raise ValueError(
            f"The {name} atomic-number table must contain positive integers."
        )

    if len(set(values)) != len(values):
        raise ValueError(
            f"The {name} atomic-number table contains duplicates: {values}."
        )

    return values


def align_node_attrs(
    dataset,
    target_atomic_numbers: Union[Sequence[int], torch.Tensor],
):
    """Align graph node attributes with an external model element table."""

    if not hasattr(dataset, "metadata"):
        raise TypeError("The dataset must expose a `metadata` attribute.")

    if "atomic_numbers" not in dataset.metadata:
        raise KeyError("The dataset metadata must contain `atomic_numbers`.")

    data_list = dataset["data_list"]

    source_values = _as_atomic_number_list(
        dataset.metadata["atomic_numbers"],
        name="source",
    )
    target_values = _as_atomic_number_list(
        target_atomic_numbers,
        name="target",
    )

    if source_values == target_values:
        return dataset

    target_indices = {
        atomic_number: index
        for index, atomic_number in enumerate(target_values)
    }

    missing = [
        atomic_number
        for atomic_number in source_values
        if atomic_number not in target_indices
    ]

    if missing:
        raise ValueError(
            "The target model does not support atomic numbers "
            f"{missing}."
        )

    source_to_target = torch.tensor(
        [target_indices[number] for number in source_values],
        dtype=torch.long,
    )

    for graph in data_list:
        old_node_attrs = graph["node_attrs"]

        if old_node_attrs.size(1) != len(source_values):
            raise ValueError(
                "`node_attrs` width does not match "
                "`dataset.metadata['atomic_numbers']`."
            )

        target_species = source_to_target.to(
            device=old_node_attrs.device,
        )[old_node_attrs.argmax(dim=-1)]

        new_node_attrs = old_node_attrs.new_zeros(
            old_node_attrs.size(0),
            len(target_values),
        )

        new_node_attrs.scatter_(
            dim=1,
            index=target_species.reshape(-1, 1),
            value=1,
        )

        graph["node_attrs"] = new_node_attrs

    dataset.metadata["atomic_numbers"] = target_values

    return dataset
