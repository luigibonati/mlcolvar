"""Utilities shared by external-model integrations."""

from typing import Sequence, Union

import torch


__all__ = [
    "align_node_attrs",
]


def align_node_attrs(
    dataset,
    target_atomic_numbers: Union[
        Sequence[int],
        torch.Tensor,
    ],
):
    """Align graph node attributes with an external model element table.

    Graph datasets encode atomic species as one-hot node attributes. The
    meaning of each column is determined by the ordering of
    ``dataset.metadata["atomic_numbers"]``.

    This function rebuilds all ``node_attrs`` tensors using the atomic-number
    ordering required by an external pretrained model.

    The dataset is modified in place and returned for convenience.

    Parameters
    ----------
    dataset
        mlcolvar graph dataset containing ``data_list`` and metadata with an
        ``atomic_numbers`` entry.
    target_atomic_numbers
        Atomic-number ordering required by the external model.

    Returns
    -------
    dataset
        Dataset with aligned ``node_attrs`` and updated metadata.
    """

    if not hasattr(
        dataset,
        "metadata",
    ):
        raise TypeError(
            "The dataset must expose a `metadata` attribute."
        )

    if "atomic_numbers" not in dataset.metadata:
        raise KeyError(
            "The dataset metadata must contain `atomic_numbers`."
        )

    try:
        data_list = dataset["data_list"]
    except (KeyError, TypeError) as error:
        raise KeyError(
            "The dataset must contain `data_list`."
        ) from error

    source_atomic_numbers = torch.as_tensor(
        dataset.metadata["atomic_numbers"],
        dtype=torch.long,
    ).detach().cpu().reshape(-1)

    target_atomic_numbers = torch.as_tensor(
        target_atomic_numbers,
        dtype=torch.long,
    ).detach().cpu().reshape(-1)

    if source_atomic_numbers.numel() == 0:
        raise ValueError(
            "The source atomic-number table cannot be empty."
        )

    if target_atomic_numbers.numel() == 0:
        raise ValueError(
            "The target atomic-number table cannot be empty."
        )

    if torch.any(
        source_atomic_numbers <= 0
    ):
        raise ValueError(
            "The source atomic-number table must contain "
            "positive integers."
        )

    if torch.any(
        target_atomic_numbers <= 0
    ):
        raise ValueError(
            "The target atomic-number table must contain "
            "positive integers."
        )

    source_values = [
        int(number)
        for number in source_atomic_numbers.tolist()
    ]

    target_values = [
        int(number)
        for number in target_atomic_numbers.tolist()
    ]

    if source_values == target_values:
        return dataset

    if len(set(source_values)) != len(source_values):
        raise ValueError(
            "The source atomic-number table contains duplicates: "
            f"{source_values}."
        )

    if len(set(target_values)) != len(target_values):
        raise ValueError(
            "The target atomic-number table contains duplicates: "
            f"{target_values}."
        )

    target_indices = {
        atomic_number: index
        for index, atomic_number in enumerate(
            target_values
        )
    }

    missing_atomic_numbers = [
        atomic_number
        for atomic_number in source_values
        if atomic_number not in target_indices
    ]

    if missing_atomic_numbers:
        raise ValueError(
            "The target model does not support atomic numbers "
            f"{missing_atomic_numbers}."
        )

    source_to_target = torch.tensor(
        [
            target_indices[atomic_number]
            for atomic_number in source_values
        ],
        dtype=torch.long,
    )

    for graph_index, graph in enumerate(
        data_list
    ):
        if "node_attrs" not in graph:
            raise KeyError(
                f"Graph {graph_index} does not contain "
                "`node_attrs`."
            )

        old_node_attrs = graph[
            "node_attrs"
        ]

        if old_node_attrs.dim() != 2:
            raise ValueError(
                f"Graph {graph_index} `node_attrs` must be "
                "rank 2, but found shape "
                f"{tuple(old_node_attrs.shape)}."
            )

        if (
            old_node_attrs.size(1)
            != source_atomic_numbers.numel()
        ):
            raise ValueError(
                f"Graph {graph_index} contains "
                f"{old_node_attrs.size(1)} node-attribute columns, "
                "but the dataset metadata contains "
                f"{source_atomic_numbers.numel()} atomic numbers."
            )

        device = old_node_attrs.device

        local_species = old_node_attrs.argmax(
            dim=-1
        )

        target_species = source_to_target.to(
            device=device
        )[local_species]

        new_node_attrs = old_node_attrs.new_zeros(
            (
                old_node_attrs.size(0),
                target_atomic_numbers.numel(),
            )
        )

        new_node_attrs.scatter_(
            dim=1,
            index=target_species.reshape(
                -1,
                1,
            ),
            value=1,
        )

        graph["node_attrs"] = (
            new_node_attrs
        )

    dataset.metadata[
        "atomic_numbers"
    ] = target_values

    return dataset