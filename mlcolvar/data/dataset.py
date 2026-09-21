import numpy as np
import torch
import torch_geometric

from typing import List, Union
from torch.utils.data import Dataset

from mlcolvar.core.transform.utils import Statistics


__all__ = ["DictDataset"]


class DictDataset(Dataset):
    """Define a torch dataset from a dictionary of arrays or tensors.

    Examples
    --------
    ``{"data": torch.Tensor(...), "labels": ..., "weights": ...}``
    """

    def __init__(
        self,
        dictionary: dict = None,
        feature_names=None,
        metadata: dict = None,
        data_type: str = "descriptors",
        create_ref_idx: bool = False,
        **kwargs,
    ):
        """Create a dataset from a dictionary or keyword arguments.

        Parameters
        ----------
        dictionary : dict, optional
            Dictionary containing dataset fields.
        feature_names : array-like, optional
            Feature names associated with descriptor data.
        metadata : dict, optional
            Metadata shared across the whole dataset.
        data_type : {"descriptors", "graphs"}, optional
            Type of data stored in the dataset.
        create_ref_idx : bool, optional
            Whether to add reference indices to the dataset.
        **kwargs
            Additional dataset fields.
        """
        if dictionary is not None and not isinstance(dictionary, dict):
            raise TypeError(
                f"DictDataset requires a dictionary, not {type(dictionary)}."
            )

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(
                f"DictDataset metadata requires a dictionary, not {type(metadata)}."
            )

        if data_type not in {"descriptors", "graphs"}:
            raise TypeError(
                "data_type must be either 'descriptors' or 'graphs', "
                f"found {data_type!r}."
            )

        if dictionary is None:
            dictionary = {}

        dictionary = {**dictionary, **kwargs}

        if not dictionary:
            raise ValueError("Empty datasets are not supported")

        if metadata is None:
            metadata = {}

        if "data_type" in metadata:
            if metadata["data_type"] != data_type:
                raise ValueError(
                    "Two different data_type values specified. "
                    f"Found {metadata['data_type']!r} in metadata and "
                    f"{data_type!r} as keyword."
                )
        else:
            metadata["data_type"] = data_type

        for key, value in dictionary.items():
            if not isinstance(value, torch.Tensor):
                if key in {"data_list", "data_list_lag"}:
                    dictionary[key] = value
                else:
                    dictionary[key] = torch.Tensor(value)

        self._dictionary = dictionary
        self.feature_names = feature_names
        self.metadata = metadata

        values = iter(dictionary.values())
        self.length = len(next(values))

        if not all(len(value) == self.length for value in values):
            raise ValueError(
                "Not all arrays in dictionary have the same length."
            )

        if create_ref_idx and "ref_idx" not in self._dictionary:
            self._dictionary["ref_idx"] = torch.arange(
                len(self),
                dtype=torch.int,
            )

    @classmethod
    def from_colvars(
        cls,
        file_names: Union[List[str], str],
        folder: str = None,
        create_labels: bool = None,
        load_args: List[dict] = None,
        filter_args: dict = None,
        modifier_function=None,
        return_dataframe: bool = False,
        verbose: bool = True,
        start: int = 0,
        stop: int = None,
        stride: int = 1,
        delete_download: bool = True,
        read_csv_kwargs: dict = None,
    ):
        """Create a descriptor dataset from COLVAR-like files.

        Parameters
        ----------
        file_names : str or list[str]
            File name or list of file names.
        folder : str, optional
            Common folder containing the files.
        create_labels : bool, optional
            Whether to assign one label to each input file.
        load_args : list[dict], optional
            Per-file loading arguments.
        filter_args : dict, optional
            Arguments passed to ``DataFrame.filter``.
        modifier_function : callable, optional
            Function applied to the descriptor dataframe.
        return_dataframe : bool, optional
            If True, also return the loaded dataframe.
        verbose : bool, optional
            Whether to print information about the loaded data.
        start, stop, stride : int, optional
            Global slicing options.
        delete_download : bool, optional
            Whether to delete temporary downloaded files after loading.
        read_csv_kwargs : dict, optional
            Additional arguments passed to ``pandas.read_csv``.

        Returns
        -------
        DictDataset
            Dataset constructed from the input files.
        tuple[DictDataset, pandas.DataFrame]
            Dataset and dataframe when ``return_dataframe=True``.
        """
        from mlcolvar.io.colvar import _prepare_dataset_from_colvars

        dataset_kwargs, dataframe = _prepare_dataset_from_colvars(
            file_names=file_names,
            folder=folder,
            create_labels=create_labels,
            load_args=load_args,
            filter_args=filter_args,
            modifier_function=modifier_function,
            verbose=verbose,
            start=start,
            stop=stop,
            stride=stride,
            delete_download=delete_download,
            read_csv_kwargs=read_csv_kwargs,
        )

        dataset = cls(**dataset_kwargs)

        if return_dataframe:
            return dataset, dataframe

        return dataset

    @classmethod
    def graph_from_configurations(
        cls,
        config,
        atomic_numbers,
        cutoff: float,
        buffer: float = 0.0,
        long_range_cutoff: float = -1.0,
        atom_names: List = None,
        remove_isolated_nodes: bool = False,
        show_progress: bool = True,
    ):
        """Create a graph dataset from atomic configurations.

        Parameters
        ----------
        config
            Atomic configurations used to construct the graph dataset.
        atomic_numbers
            Atomic number table defining the chemical species.
        cutoff : float
            Cutoff distance used to construct graph edges.
        buffer : float, optional
            Buffer distance used when selecting environment atoms.
        long_range_cutoff : float, optional
            Cutoff distance for long-range subsystem edges. If negative,
            long-range edges are not constructed.
        atom_names : list, optional
            Names of the system atoms.
        remove_isolated_nodes : bool, optional
            Whether to remove isolated nodes from the generated graphs.
        show_progress : bool, optional
            Whether to display graph-construction progress.

        Returns
        -------
        DictDataset
            Graph dataset constructed from the configurations.
        """
        from mlcolvar.data.graph.utils import (
            _prepare_dataset_from_configurations,
        )

        dataset_kwargs = _prepare_dataset_from_configurations(
            config=config,
            atomic_numbers=atomic_numbers,
            cutoff=cutoff,
            buffer=buffer,
            long_range_cutoff=long_range_cutoff,
            atom_names=atom_names,
            remove_isolated_nodes=remove_isolated_nodes,
            show_progress=show_progress,
        )

        return cls(**dataset_kwargs)

    @classmethod
    def graph_from_trajectories(
        cls,
        trajectories: Union[List[str], str],
        cutoff: float,
        topologies: Union[List[str], str, None] = None,
        load_args: List[dict] = None,
        folder: str = None,
        trajectory_labels: list = None,
        graph_labels: list = None,
        node_labels: list = None,
        system_selection=None,
        environment_selection=None,
        buffer: float = 0.0,
        subsystem_selection=None,
        long_range_cutoff: float = -1.0,
        return_trajectories: bool = False,
        remove_isolated_nodes: bool = True,
        show_progress: bool = False,
        atom_names: List = None,
        lengths_conversion: float = None,
        delete_download: bool = True,
        backend: str = "mdtraj",
    ):
        """Create a graph dataset directly from trajectory files.

        Parameters
        ----------
        trajectories : str or list[str]
            Path or paths to trajectory files.
        cutoff : float
            Cutoff distance used to construct graph edges in Angstroms.
        topologies : str or list[str], optional
            Topology file or files used by the MDTraj backend.
        load_args : list[dict], optional
            Per-trajectory loading options containing ``start``, ``stop``,
            and ``stride``.
        folder : str, optional
            Common directory containing trajectory and topology files.
        trajectory_labels : list, optional
            Labels assigned to trajectories and broadcast to selected frames.
        graph_labels : list, optional
            Frame-level graph labels.
        node_labels : list, optional
            Node-level labels.
        system_selection : optional
            Backend-specific selection defining system atoms.
        environment_selection : optional
            Backend-specific selection defining environment atoms.
        buffer : float, optional
            Buffer distance used when selecting environment atoms.
        subsystem_selection : optional
            Backend-specific selection defining atoms used for long-range
            edges.
        long_range_cutoff : float, optional
            Cutoff distance for long-range subsystem edges. If negative,
            long-range edges are not constructed.
        return_trajectories : bool, optional
            If True, also return the loaded trajectory objects.
        remove_isolated_nodes : bool, optional
            Whether to remove isolated nodes from the generated graphs.
        show_progress : bool, optional
            Whether to display graph-construction progress.
        atom_names : list, optional
            Names of the system atoms. If not provided, they are inferred
            when possible.
        lengths_conversion : float, optional
            Conversion factor applied to trajectory coordinates. If None,
            the backend default is used.
        delete_download : bool, optional
            Whether temporary downloaded files are deleted after loading.
        backend : {"mdtraj", "ase"}, optional
            Backend used to load trajectory files.

        Returns
        -------
        DictDataset
            Graph dataset constructed from the trajectories.
        tuple[DictDataset, list]
            Dataset and loaded trajectories when
            ``return_trajectories=True``.
        """
        from mlcolvar.io.graphs._utils import (
            _check_atom_selection,
            _normalize_graph_target_inputs,
        )
        from mlcolvar.io.graphs.common import _load_trajectories

        if backend not in {"mdtraj", "ase"}:
            raise ValueError(
                f"Unknown backend {backend!r}. Expected 'mdtraj' or 'ase'."
            )

        if trajectory_labels is not None and graph_labels is not None:
            raise ValueError(
                "Only one of `trajectory_labels` or `graph_labels` "
                "can be provided."
            )

        _check_atom_selection(
            system_selection=system_selection,
            environment_selection=environment_selection,
            subsystem_selection=subsystem_selection,
            buffer=buffer,
            long_range_cutoff=long_range_cutoff,
        )

        loaded_trajectories = _load_trajectories(
            trajectories=trajectories,
            topologies=topologies,
            load_args=load_args,
            folder=folder,
            delete_download=delete_download,
            backend=backend,
        )

        graph_labels, node_labels = _normalize_graph_target_inputs(
            trajectories=loaded_trajectories,
            trajectory_labels=trajectory_labels,
            graph_labels=graph_labels,
            node_labels=node_labels,
        )

        if lengths_conversion is None:
            lengths_conversion = 10 if backend == "mdtraj" else 1

        if backend == "mdtraj":
            from mlcolvar.io.graphs.mdtraj_ import (
                _prepare_configurations_from_mdtraj_trajectories,
            )

            configurations, atomic_numbers, atom_names = (
                _prepare_configurations_from_mdtraj_trajectories(
                    trajectories=loaded_trajectories,
                    graph_labels=graph_labels,
                    node_labels=node_labels,
                    system_selection=system_selection,
                    environment_selection=environment_selection,
                    subsystem_selection=subsystem_selection,
                    lengths_conversion=lengths_conversion,
                    atom_names=atom_names,
                )
            )

        else:
            from mlcolvar.io.graphs.ase_ import (
                _prepare_configurations_from_ase_trajectories,
            )

            configurations, atomic_numbers, atom_names = (
                _prepare_configurations_from_ase_trajectories(
                    trajectories=loaded_trajectories,
                    graph_labels=graph_labels,
                    node_labels=node_labels,
                    system_selection=system_selection,
                    environment_selection=environment_selection,
                    subsystem_selection=subsystem_selection,
                    lengths_conversion=lengths_conversion,
                    atom_names=atom_names,
                )
            )

        dataset = cls.graph_from_configurations(
            config=configurations,
            atomic_numbers=atomic_numbers,
            cutoff=cutoff,
            buffer=buffer,
            long_range_cutoff=long_range_cutoff,
            atom_names=atom_names,
            remove_isolated_nodes=remove_isolated_nodes,
            show_progress=show_progress,
        )

        if return_trajectories:
            return dataset, loaded_trajectories

        return dataset

    def __getitem__(self, index):
        """Return a field, one sample, or a sliced DictDataset."""

        if isinstance(index, str):
            return self._dictionary[index]

        is_scalar = (
            isinstance(index, (int, np.integer))
            or (
                isinstance(index, (torch.Tensor, np.ndarray))
                and index.ndim == 0
            )
        )

        if is_scalar:
            if isinstance(index, (torch.Tensor, np.ndarray)):
                index = index.item()

            return {
                key: value[index]
                for key, value in self._dictionary.items()
            }

        if isinstance(index, slice):
            sliced = {
                key: value[index]
                for key, value in self._dictionary.items()
            }

        else:
            if isinstance(index, torch.Tensor):
                indices = (
                    torch.where(index)[0].tolist()
                    if index.dtype == torch.bool
                    else index.flatten().tolist()
                )

            elif isinstance(index, np.ndarray):
                indices = (
                    np.flatnonzero(index).tolist()
                    if index.dtype == bool
                    else index.flatten().tolist()
                )

            else:
                indices = list(index)

                if indices and all(
                    isinstance(item, (bool, np.bool_))
                    for item in indices
                ):
                    indices = [
                        i
                        for i, selected in enumerate(indices)
                        if selected
                    ]

            sliced = {
                key: (
                    value[indices]
                    if isinstance(value, (torch.Tensor, np.ndarray))
                    else [value[i] for i in indices]
                )
                for key, value in self._dictionary.items()
            }

        return type(self)(
            dictionary=sliced,
            feature_names=self.feature_names,
            metadata=self.metadata.copy(),
            data_type=self.metadata.get("data_type", "descriptors"),
        )

    def __setitem__(self, index, value):
        if isinstance(index, str):
            if len(value) != len(self):
                raise ValueError(
                    f"length of value ({len(value)}) != "
                    f"length of dataset ({len(self)})."
                )

            self._dictionary[index] = value
            return

        raise NotImplementedError(
            f"Only string indexes can be set, {type(index)} is not supported."
        )

    def __len__(self):
        value = next(iter(self._dictionary.values()))
        return len(value)

    def get_stats(self):
        """Compute statistics of the dataset.

        Returns
        -------
        dict
            Dictionary containing statistics for each dataset field.
        """
        if self.metadata.get("data_type") == "graphs":
            raise ValueError(
                "Method get_stats is not supported for graph-based datasets."
            )

        stats = {}

        for key in self.keys:
            print("KEY: ", key, end="\n\n\n")

            if key != "ref_idx":
                stats[key] = Statistics(
                    self._dictionary[key]
                ).to_dict()

        return stats

    def __repr__(self) -> str:
        parts = ["DictDataset("]

        for key, value in self._dictionary.items():
            if key in {"data_list", "data_list_lag"}:
                parts.append(f' "{key}": {len(value)},')
            else:
                parts.append(f' "{key}": {list(value.shape)},')

        if self.metadata:
            parts.append("\n\t    metadata={")

            for key, value in self.metadata.items():
                parts.append(
                    f'"{key}": {value},\n\t\t      '
                )

            if parts[-1].endswith(",\n\t\t      "):
                parts[-1] = parts[-1][:-10]

            parts.append(" },")

        if parts[-1].endswith(","):
            parts[-1] = parts[-1][:-1]

        parts.append(" )")

        return "".join(parts)

    @property
    def keys(self):
        return tuple(self._dictionary.keys())

    @property
    def feature_names(self):
        """Feature names."""
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value):
        self._feature_names = (
            np.asarray(value, dtype=str)
            if value is not None
            else None
        )

    def get_graph_inputs(self):
        """Return the complete graph dataset as a single batch."""
        assert self.metadata["data_type"] == "graphs", (
            "Graph inputs can only be generated for graph-based datasets"
        )

        loader = torch_geometric.loader.DataLoader(
            self,
            batch_size=len(self),
            shuffle=False,
        )

        return next(iter(loader))["data_list"]