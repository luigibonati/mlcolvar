import numpy as np
import torch
from typing import List
import mdtraj
from warnings import warn
from mlcolvar.data.graph.atomic import AtomicNumberTable


__all__ = ["_as_torch_if_array",
           "_to_torch_tensor",
           "_get_selected_frame_indices",
           "_normalize_trajectory_labels",
           "_broadcast_trajectory_to_graph_labels",
           "_normalize_frame_level_labels",
           "_normalize_graph_target_inputs",
           "_check_atom_selection",
           "_update_atomic_numbers_from_configurations"]

def _as_torch_if_array(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x)
    return x


def _to_torch_tensor(x, dtype=torch.get_default_dtype()):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def _get_selected_frame_indices(n_frames: int, load_arg: dict = None) -> List[int]:
    if load_arg is None:
        return list(range(0, n_frames, 1))
    start = load_arg.get('start', 0)
    stop = load_arg.get('stop', None)
    stride = load_arg.get('stride', 1)
    if stop is None:
        stop = n_frames
    return list(range(start, stop, stride))


def _normalize_trajectory_labels(trajectory_labels, n_traj: int):
    if trajectory_labels is None:
        return [torch.tensor([[i]], dtype=torch.get_default_dtype()) for i in range(n_traj)]

    trajectory_labels = _as_torch_if_array(trajectory_labels)

    if isinstance(trajectory_labels, torch.Tensor):
        if trajectory_labels.ndim == 0:
            trajectory_labels = trajectory_labels.reshape(1).repeat(n_traj)
            trajectory_labels = trajectory_labels.tolist()
        elif trajectory_labels.ndim >= 1 and trajectory_labels.shape[0] == n_traj:
            trajectory_labels = [trajectory_labels[i] for i in range(n_traj)]
        else:
            if n_traj == 1:
                raise ValueError(
                    f"trajectory_labels has length {trajectory_labels.shape[0]} for a single trajectory. Use graph_labels for per-frame targets."
                )
            raise ValueError(
                f"trajectory_labels first dimension ({trajectory_labels.shape[0]}) must match number of trajectories ({n_traj})."
            )

    if not isinstance(trajectory_labels, (list, tuple)):
        trajectory_labels = [trajectory_labels]

    if len(trajectory_labels) != n_traj:
        raise ValueError(
            f"Number of trajectory labels ({len(trajectory_labels)}) must match number of trajectories ({n_traj})."
        )

    normalized = []
    for item in trajectory_labels:
        item = _as_torch_if_array(item)
        if isinstance(item, torch.Tensor):
            if item.ndim == 0:
                normalized.append(item.reshape(1, 1).to(dtype=torch.get_default_dtype()))
            else:
                normalized.append(item.reshape(-1, 1).to(dtype=torch.get_default_dtype()))
        elif np.isscalar(item):
            normalized.append(torch.tensor([[item]], dtype=torch.get_default_dtype()))
        else:
            arr = _to_torch_tensor(item)
            if arr.ndim == 0:
                normalized.append(arr.reshape(1, 1))
            else:
                normalized.append(arr.reshape(-1, 1))
    return normalized


def _broadcast_trajectory_to_graph_labels(trajectory_labels, frame_counts: List[int]):
    trajectory_labels = _normalize_trajectory_labels(trajectory_labels, len(frame_counts))
    broadcast = []
    for i, n_frames in enumerate(frame_counts):
        y = trajectory_labels[i].reshape(1, -1)
        broadcast.append(y.repeat(n_frames, 1))
    return broadcast

def _normalize_frame_level_labels(labels, frame_counts: List[int], name: str):
    n_traj = len(frame_counts)

    if labels is None:
        return [None for _ in range(n_traj)]

    labels = _as_torch_if_array(labels)

    if n_traj == 1:
        n_frames = frame_counts[0]
        if isinstance(labels, torch.Tensor) and labels.ndim >= 1 and labels.shape[0] == n_frames:
            labels = [labels]
        elif (
            isinstance(labels, (list, tuple))
            and len(labels) == n_frames
            and len(labels) != n_traj
        ):
            warn(
                f"`{name}` was passed as a flat list/tuple of length {n_frames} for a single trajectory; "
                "interpreting it as frame-level labels. To avoid ambiguity, prefer passing a tensor/array "
                "or wrapping per-trajectory labels as [labels]."
            )
            labels = [_to_torch_tensor(labels)]

    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"{name} must be a list/tuple with one entry per trajectory.")

    if len(labels) != n_traj:
        raise ValueError(
            f"{name} has {len(labels)} entries but number of trajectories is {n_traj}."
        )

    normalized = []
    for i, item in enumerate(labels):
        n_frames = frame_counts[i]
        if item is None:
            normalized.append(None)
            continue

        item = _as_torch_if_array(item)
        arr = _to_torch_tensor(item)

        if arr.ndim == 0:
            arr = arr.reshape(1, 1).repeat(n_frames, 1)
        elif arr.ndim == 1:
            if arr.shape[0] != n_frames:
                raise ValueError(
                    f"{name}[{i}] length ({arr.shape[0]}) must match selected frames ({n_frames})."
                )
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 2:
            if arr.shape[0] != n_frames:
                raise ValueError(
                    f"{name}[{i}] first dimension ({arr.shape[0]}) must match selected frames ({n_frames})."
                )
        elif arr.ndim == 3 and name == 'node_labels':
            if arr.shape[0] != n_frames:
                raise ValueError(
                    f"{name}[{i}] first dimension ({arr.shape[0]}) must match selected frames ({n_frames})."
                )
        else:
            raise ValueError(f"Unsupported shape for {name}[{i}]: {tuple(arr.shape)}.")

        normalized.append(arr)

    return normalized

def _normalize_graph_target_inputs(
    trajectories: List,
    load_args: list,
    trajectory_labels=None,
    graph_labels=None,
    node_labels=None,
):
    n_traj = len(trajectories)
    frame_counts = [
        len(_get_selected_frame_indices(
            n_frames=len(traj),
            load_arg=load_args[i] if load_args is not None else None,
        ))
        for i, traj in enumerate(trajectories)
    ]

    if trajectory_labels is not None and graph_labels is not None:
        raise ValueError("Only one of `trajectory_labels` or `graph_labels` can be provided.")

    if trajectory_labels is None and graph_labels is None:
        trajectory_labels = [i for i in range(n_traj)]

    if graph_labels is None:
        graph_labels = _broadcast_trajectory_to_graph_labels(
            trajectory_labels=trajectory_labels,
            frame_counts=frame_counts,
        )
    else:
        graph_labels = _normalize_frame_level_labels(graph_labels, frame_counts, name='graph_labels')

    node_labels = _normalize_frame_level_labels(node_labels, frame_counts, name='node_labels')

    return graph_labels, node_labels


def _check_atom_selection(system_selection : str,
                          environment_selection : str,
                          subsystem_selection : str,
                          buffer : float = 0,
                          long_range_cutoff : float = -1.0):
    """Check compatibility of selection keywords combinations. 
    NOTE: This doesn't check if the selection is correct."""

    if environment_selection is not None:
        if system_selection is None:
            raise ValueError('The `environment_selection` argument requires the `system_selection` argument to be defined!')
     
    if environment_selection is None:
        assert buffer == 0, ('The `buffer` argument is only valid when `environment_selection` is provided!')
    
    if (subsystem_selection is not None) and (long_range_cutoff <= 0):
        raise ValueError('The `subsystem_selection` argument requires a positive `long_range_cutoff` argument!')

def _update_atomic_numbers_from_configurations(configurations, 
                                               atomic_numbers):
    if isinstance(atomic_numbers, AtomicNumberTable):
        atomic_numbers = atomic_numbers.zs
    
    for configuration in configurations:
        aux = np.unique(np.array(configuration.atomic_numbers))
        check = [j not in atomic_numbers for j in aux]

        if any(check):
            atomic_numbers.extend(iter([int(k) for k in aux[check]]))
    
    atomic_numbers = AtomicNumberTable.from_zs(atomic_numbers)
    return atomic_numbers