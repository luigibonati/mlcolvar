import numpy as np
import torch
import os
from typing import Union, List, Tuple
import mdtraj

from mlcolvar.utils.io._utils import _download_temp_file
from mlcolvar.utils.io.graphs._utils import *
from mlcolvar.utils.io.graphs.mdtraj_ import *
from mlcolvar.utils.io.graphs.ase_ import *

from mlcolvar.data import DictDataset
from mlcolvar.data.graph.atomic import AtomicNumberTable

__all__ = ["create_dataset_from_trajectories"]


def create_dataset_from_trajectories(
    trajectories: Union[List[str], str],
    topologies: Union[List[str], str, None],
    cutoff: float,
    buffer: float = 0.0,
    long_range_cutoff: float = -1.0,
    atomic_numbers: AtomicNumberTable = None,
    load_args: list = None,
    folder: str = None,
    labels: list = None,
    trajectory_labels: list = None,
    graph_labels: list = None,
    node_labels: list = None,
    system_selection: str = None,
    environment_selection: str = None,
    subsystem_selection: str = None,
    return_trajectories: bool = False,
    remove_isolated_nodes: bool = True,
    show_progress: bool = False,
    save_names=True,
    lengths_conversion : float = 10.0,
    delete_download: bool = True,
) -> Union[
    DictDataset,
    Tuple[
        DictDataset,
        Union[List[List[mdtraj.Trajectory]], List[mdtraj.Trajectory]]
    ]
]:
    """
    Create a dataset from a set of trajectory files.

    Parameters
    ----------
    trajectories: Union[List[str], str]
        Paths to trajectories files.
    topologies: Union[List[str], str, None]
        Path to topology files. Only for .xyz files it can be set to None or empty to generate automatically a topology file.
    cutoff: float (units: Ang)
        The graph cutoff radius.
    buffer: float
        Buffer size used in finding active environment atoms.
    long_range_cutoff : float
            Cutoff radius for the long-range edges defined on subsystem atoms. 
            If negative, no long-range interactions are considered, by default -1.0. 
            This option should be defined with the `subsystem_selection` option.
    atomic_numbers: mlcolvar.graph.data.atomic.AtomicNumberTable
        The atomic number table used to build the node attributes. If not
        given, it will be created from the given trajectories.
    load_args: list[dict], optional
        List of dictionaries for loading options for each file (keys: start,stop,stride), by default None
    folder: str
        Common path for the files to be imported. If set, filenames become
        `folder/file_name`.
    labels: list
        Deprecated alias for `trajectory_labels`.
    trajectory_labels: list
        One label (or vector of labels) per trajectory. It is broadcast to all
        selected frames and saved as `graph_labels`.
    graph_labels: list
        One label (or vector of labels) per selected frame of each trajectory.
        Mutually exclusive with `trajectory_labels`.
    node_labels: list
        Optional node-level labels per selected frame and trajectory.
    system_selection: str
        MDTraj style atom selections [1] of the system atoms. If given, only
        selected atoms will be loaded from the trajectories. This option may
        increase the speed of building graphs.
    environment_selection: str
        MDTraj style atom selections [1] of the environment atoms. If given,
        only the system atoms and [the environment atoms within the cutoff
        radius of the system atoms] will be kept in the graph.
    subsystem_selection: str
        MDTraj style atom selections of the subsystem atoms. If given, long
        edges will be put between subsystem atoms. This option should be
        defined along with the `long_range_cutoff` option. Besides, all atoms selected
        by this option should also be selected by the `system_selection`.
    return_trajectories: bool
        If also return the loaded trajectory objects.
    remove_isolated_nodes: bool
        If remove isolated nodes from the dataset.
    show_progress: bool
        If show the progress bar, by default False.
    save_names: bool
        If to save names from topology file, by default True
    lengths_conversion: float,
        Conversion factor for length units, by default 10.
        MDTraj uses nanometers, the default sends to Angstroms.
    delete_download: bool, optinal
        whether to delete the downloaded file after it has been loaded, default True.    

    Returns
    -------
    dataset: mlcolvar.graph.data.GraphDataSet
        The graph dataset.
    trajectories: Union[List[List[mdtraj.Trajectory]], List[mdtraj.Trajectory]]
        The loaded trajectory objects.

    Notes
    -----
    The login behind this method is like the follows:
        1. If only `system_selection` is given, the method will only load atoms
        selected by this selection, from the trajectories.
        2. If both `system_selection` and `environment_selection` are given,
        the method will load the atoms select by both selections, but will
        build graphs using [the system atoms] and [the environment atoms within
        the cutoff radius of the system atoms].

    References
    ----------
    .. [1] https://www.mdtraj.org/1.9.8.dev0/atom_selection.html
    """

    # ======================================= Initial checks =======================================

    # ensure trajectories is a list
    if isinstance(trajectories, str):
        trajectories = [trajectories]

    # backward compatibility: labels is an alias for trajectory_labels
    if labels is not None and trajectory_labels is not None:
        raise ValueError("Only one of `labels` or `trajectory_labels` can be provided.")
    if labels is not None:
        trajectory_labels = labels

    if trajectory_labels is not None and graph_labels is not None:
        raise ValueError("Only one of `trajectory_labels` or `graph_labels` can be provided.")

    # check atoms selection makes sense and get the total selection
    _setup_atom_selection(system_selection=system_selection,
                          environment_selection=environment_selection,
                          subsystem_selection=subsystem_selection,
                          buffer=buffer,
                          long_range_cutoff=long_range_cutoff,
                          return_selection=False)


    # ================================== Topology files handling ===================================

    # check topologies if given, with xyz it can be None
    if topologies is not None:
        assert len(trajectories) == len(topologies) or len(topologies)==1 or isinstance(topologies, str), (
            'Either a single topology file or as many as the trajectory files must be provided!'
        )
    
    # Allow topology to be None or empty. In that case, create a list of empty strings.
    shared_top = True
    if isinstance(topologies, str):
        topologies = [topologies for _ in trajectories]
    elif topologies is None or (isinstance(topologies, list) and len(topologies) == 0):
        topologies = ["" for _ in trajectories]
    elif len(topologies) == 1 and len(trajectories) > 1:
        topologies = [topologies for _ in trajectories]
    else: 
        shared_top = False


    # ========================================= Load files =========================================

    trajectories_in_memory = []
    for i in range(len(trajectories)):
        # ============================== PREPARATION ==============================
        assert isinstance(trajectories[i], str)

        # check if folder is given
        if folder is not None:
            trajectories[i] = os.path.join(folder, trajectories[i])
            if topologies[i]:
                topologies[i] = os.path.join(folder, topologies[i])
        
        # check if trajectories[i] is an url
        download_traj = False
        if "http" in trajectories[i]:
            download_traj = True
            url_traj = trajectories[i]
            temp_traj, trajectories[i] = _download_temp_file(file_url=url_traj, 
                                                             delete_download=delete_download, 
                                                             append_suffix=True, 
                                                             return_name=True
                                                            )

        # check if topologies[i] is an url
        download_top = False
        if "http" in topologies[i]:
            download_top = True
            # check if it is really needed to download or top is shared
            if shared_top and i > 0: 
                topologies[i] = topologies[0]
            else:
                url_top = topologies[i]
                temp_top, topologies[i] = _download_temp_file(file_url=url_top, 
                                                              delete_download=delete_download, 
                                                              append_suffix=True, 
                                                              return_name=True
                                                            )

        # check extension of file, if .xyz create topology file through ASE
        _, ext = os.path.splitext(trajectories[i])
        if (ext.lower() == ".xyz") and (not topologies[i]):
            pdb_file = trajectories[i].replace(ext, '_top.pdb')
            topologies[i] = create_pdb_from_xyz(trajectories[i], pdb_file)


        # ============================== LOADING ==============================

        traj = load_traj_with_mdtraj(trajectory=trajectories[i],
                                     topology=topologies[i],
                                     )

        trajectories_in_memory.append(traj)

        # remove temporary files from dowload if needed
        if download_traj:
            if delete_download:
                temp_traj.close()
            else:
                print(f"downloaded file ({url_traj}) saved as ({trajectories[i]}).")

        if download_top:
            if not shared_top or (shared_top and i == len(trajectories)):
                if delete_download:
                    temp_top.close()
                else:
                    print(f"downloaded file ({url_top}) saved as ({topologies[i]}).")
    #endfor i in range(len(trajectories)):

    # TODO make it work with save_names, possibly moving into dataset_from_mdtraj_trajectories
    atom_names = None

    graph_labels, node_labels = _normalize_graph_target_inputs(trajectories=trajectories_in_memory,
                                                               load_args=load_args,
                                                               trajectory_labels=trajectory_labels,
                                                               graph_labels=graph_labels,
                                                               node_labels=node_labels,
                                                               )

    dataset = dataset_from_mdtraj_trajectories(trajectories=trajectories_in_memory,
                                               graph_labels=graph_labels,
                                               node_labels=node_labels,
                                               cutoff=cutoff, 
                                               atomic_numbers=atomic_numbers,
                                               system_selection=system_selection,
                                               environment_selection=environment_selection,
                                               subsystem_selection=subsystem_selection,
                                               load_args=load_args,
                                               lengths_conversion=lengths_conversion,
                                               buffer=buffer,
                                               long_range_cutoff=long_range_cutoff,
                                               atom_names=atom_names,
                                               remove_isolated_nodes=remove_isolated_nodes,
                                               show_progress=show_progress)

    if return_trajectories:
        return dataset, trajectories_in_memory
    else:
        return dataset



def test_datasesetFromTrajectories():
    from mlcolvar.tests import data_dir

    with data_dir() as data_folder:
        create_dataset_from_trajectories(
            trajectories=['r.dcd',
                        'p.dcd'],
            topologies=['r.pdb', 
                        'p.pdb'],
            folder=data_folder,
            cutoff=8.0,  # Ang
            labels=None,
            system_selection='all and not type H',
            show_progress=False,
        )

        dataset = create_dataset_from_trajectories(
                    trajectories=['r.dcd',
                                'p.dcd'],
                    topologies=['r.pdb', 
                                'p.pdb'],
                    folder=data_folder,
                    cutoff=8.0,  # Ang
                    labels=[0,1],
                    system_selection='all and not type H',
                    show_progress=False,
                    load_args=[{'start' : 0, 'stop' : 10, 'stride' : 1},
                            {'start' : 6, 'stop' : 10, 'stride' : 2}]
                )
        assert(len(dataset)==12)

        dataset = create_dataset_from_trajectories(
                    trajectories=['r.dcd', 'r.dcd',
                                'p.dcd', 'p.dcd'],
                    topologies=['r.pdb', 'r.pdb', 
                                'p.pdb', 'p.pdb'],
                    folder=data_folder,
                    cutoff=8.0,  # Ang
                    labels=[0,1,2,3],
                    system_selection='all and not type H',
                    show_progress=False,
                    load_args=[{'start' : 0, 'stop' : 10, 'stride' : 1}, {'start' : 0, 'stop' : 10, 'stride' : 1},
                            {'start' : 6, 'stop' : 10, 'stride' : 2}, {'start' : 6, 'stop' : 10, 'stride' : 2}]
                )
        assert(len(dataset)==24)


def test_create_dataset_from_trajectories(text: str = """
CRYST1    2.000    2.000    2.000  90.00  90.00  90.00 P 1           1
ATOM      1  OH2 TIP3W   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  H1  TIP3W   1       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      3  H2  TIP3W   1       0.700  -0.700   0.000  1.00  0.00      WT1  H
ENDMODEL
ATOM      1  OH2 TIP3W   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  H1  TIP3W   1       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      3  H2  TIP3W   1       0.700  -0.700   0.000  1.00  0.00      WT1  H
END
""", 
system_selection: str = None
) -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dataset_path = "test_dataset.pdb"
        test_dataset_path = os.path.join(tmpdir, test_dataset_path)
        with open(test_dataset_path, 'w') as fp:
            print(text, file=fp)

        dataset, trajectories = create_dataset_from_trajectories(
            trajectories=[test_dataset_path, test_dataset_path, test_dataset_path],
            topologies=[test_dataset_path, test_dataset_path, test_dataset_path],
            cutoff=1.0,
            system_selection=system_selection,
            return_trajectories=True,
            show_progress=False
        )

        assert len(dataset) == 6
        assert dataset.metadata["cutoff"] == 1.0
        assert dataset.metadata["atomic_numbers"] == [1, 8]
        assert len(trajectories[0]) == 2
        assert len(trajectories[1]) == 2
        assert len(trajectories[2]) == 2

        assert dataset[0]["data_list"]['graph_labels'] == torch.tensor([[0.0]])
        assert dataset[1]["data_list"]['graph_labels'] == torch.tensor([[0.0]])
        assert dataset[2]["data_list"]['graph_labels'] == torch.tensor([[1.0]])
        assert dataset[3]["data_list"]['graph_labels'] == torch.tensor([[1.0]])
        assert dataset[4]["data_list"]['graph_labels'] == torch.tensor([[2.0]])
        assert dataset[5]["data_list"]['graph_labels'] == torch.tensor([[2.0]])

        dataset, trajectories = create_dataset_from_trajectories(
            trajectories=[test_dataset_path, test_dataset_path, test_dataset_path],
            topologies=test_dataset_path,
            cutoff=1.0,
            labels=None,
            system_selection=system_selection,
            return_trajectories=True,
            show_progress=False
        )

        assert dataset[0]["data_list"]['graph_labels'] == torch.tensor([[0.0]])
        assert dataset[1]["data_list"]['graph_labels'] == torch.tensor([[0.0]])
        assert dataset[2]["data_list"]['graph_labels'] == torch.tensor([[1.0]])
        assert dataset[3]["data_list"]['graph_labels'] == torch.tensor([[1.0]])
        assert dataset[4]["data_list"]['graph_labels'] == torch.tensor([[2.0]])
        assert dataset[5]["data_list"]['graph_labels'] == torch.tensor([[2.0]])

        dataset = create_dataset_from_trajectories(
            trajectories=[test_dataset_path, test_dataset_path, test_dataset_path],
            topologies=[test_dataset_path, test_dataset_path, test_dataset_path],
            cutoff=1.0,
            trajectory_labels=[10.0, 20.0, 30.0],
            system_selection=system_selection,
            show_progress=False,
        )
        assert dataset[0]["data_list"]["graph_labels"] == torch.tensor([[10.0]])
        assert dataset[1]["data_list"]["graph_labels"] == torch.tensor([[10.0]])
        assert dataset[2]["data_list"]["graph_labels"] == torch.tensor([[20.0]])
        assert dataset[3]["data_list"]["graph_labels"] == torch.tensor([[20.0]])
        assert dataset[4]["data_list"]["graph_labels"] == torch.tensor([[30.0]])
        assert dataset[5]["data_list"]["graph_labels"] == torch.tensor([[30.0]])

        dataset = create_dataset_from_trajectories(
            trajectories=[test_dataset_path, test_dataset_path, test_dataset_path],
            topologies=[test_dataset_path, test_dataset_path, test_dataset_path],
            cutoff=1.0,
            graph_labels=[np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])],
            system_selection=system_selection,
            show_progress=False,
        )
        assert torch.allclose(dataset[0]["data_list"]["graph_labels"], torch.tensor([[1.0]]))
        assert torch.allclose(dataset[1]["data_list"]["graph_labels"], torch.tensor([[2.0]]))
        assert torch.allclose(dataset[2]["data_list"]["graph_labels"], torch.tensor([[3.0]]))
        assert torch.allclose(dataset[3]["data_list"]["graph_labels"], torch.tensor([[4.0]]))
        assert torch.allclose(dataset[4]["data_list"]["graph_labels"], torch.tensor([[5.0]]))
        assert torch.allclose(dataset[5]["data_list"]["graph_labels"], torch.tensor([[6.0]]))

        def check_data_1(data) -> None:
            assert(torch.allclose(data["data_list"]['edge_index'], torch.tensor([[0, 0, 1, 1, 2, 2],
                                                                                [2, 1, 0, 2, 1, 0]])
                                    )
                    )
            assert(torch.allclose(data["data_list"]['shifts'], torch.tensor([[0.0, 0.0, 0.0],
                                                                            [0.0, 0.0, 0.0],
                                                                            [0.0, 0.0, 0.0],
                                                                            [0.0, 2.0, 0.0],
                                                                            [0.0, -2.0, 0.0],
                                                                            [0.0, 0.0, 0.0]])
                                        )
                    )
            assert(torch.allclose(data["data_list"]['unit_shifts'], torch.tensor([[0.0, 0.0, 0.0],
                                                                                [0.0, 0.0, 0.0],
                                                                                [0.0, 0.0, 0.0],
                                                                                [0.0, 1.0, 0.0],
                                                                                [0.0, -1.0, 0.0],
                                                                                [0.0, 0.0, 0.0]])
                                )
                    )
            assert(torch.allclose(data["data_list"]['positions'], torch.tensor([[0.0, 0.0, 0.0],
                                                                                [0.7, 0.7, 0.0],
                                                                                [0.7, -0.7, 0.0]])
                                )
                    )
            assert(torch.allclose(data["data_list"]['cell'], torch.tensor([[2.0, 0.0, 0.0],
                                                                        [0.0, 2.0, 0.0],
                                                                        [0.0, 0.0, 2.0]])
                            )
                    )
            assert(torch.allclose(data["data_list"]['node_attrs'], torch.tensor([[0.0, 1.0], 
                                                                                [1.0, 0.0], 
                                                                                [1.0, 0.0]])
                            )
                    )

        for i in range(6):
            check_data_1(dataset[i])

        if system_selection is not None:

            dataset = create_dataset_from_trajectories(
                trajectories=[test_dataset_path, test_dataset_path, test_dataset_path],
                topologies=[test_dataset_path, test_dataset_path, test_dataset_path],
                cutoff=1.0,
                system_selection='type O and {:s}'.format(system_selection),
                environment_selection='type H and {:s}'.format(system_selection),
                show_progress=False
            )

            for i in range(6):
                check_data_1(dataset[i])

            dataset = create_dataset_from_trajectories(
                trajectories=[test_dataset_path, test_dataset_path, test_dataset_path],
                topologies=[test_dataset_path, test_dataset_path, test_dataset_path],
                cutoff=1.0,
                system_selection='name H1 and {:s}'.format(system_selection),
                environment_selection='name H2 and {:s}'.format(system_selection),
                show_progress=False
            )

            def check_data_2(data) -> None:
                assert(torch.allclose(data["data_list"]['edge_index'], torch.tensor([[0, 1], [1, 0]])))
                assert(torch.allclose(data["data_list"]['shifts'], torch.tensor([[0.0, 2.0, 0.0], 
                                                                                [0.0, -2.0, 0.0]])
                                        )
                        )
                assert(torch.allclose(data["data_list"]['unit_shifts'], torch.tensor([[0.0, 1.0, 0.0], 
                                                                                    [0.0, -1.0, 0.0]])
                                    )
                        )
                assert(torch.allclose(data["data_list"]['positions'], torch.tensor([[0.7, 0.7, 0.0], 
                                                                                    [0.7, -0.7, 0.0]])
                                    )
                        )
                assert(torch.allclose(data["data_list"]['cell'], torch.tensor([[2.0, 0.0, 0.0],
                                                                            [0.0, 2.0, 0.0],
                                                                            [0.0, 0.0, 2.0]])
                                    )
                        )
                print(data["data_list"]['node_attrs'])
                assert(torch.allclose(data["data_list"]['node_attrs'], torch.tensor([[1.0], 
                                                                                    [1.0]])
                                    )
                        )

            for i in range(6):
                check_data_2(dataset[i])


def test_graph_and_node_labels_syntax(text: str = """
CRYST1    2.000    2.000    2.000  90.00  90.00  90.00 P 1           1
ATOM      1  OH2 TIP3W   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  H1  TIP3W   1       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      3  H2  TIP3W   1       0.700  -0.700   0.000  1.00  0.00      WT1  H
ENDMODEL
ATOM      1  OH2 TIP3W   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  H1  TIP3W   1       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      3  H2  TIP3W   1       0.700  -0.700   0.000  1.00  0.00      WT1  H
END
""") -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_dataset_path = os.path.join(tmpdir, "test_labels_syntax.pdb")
        with open(test_dataset_path, 'w') as fp:
            print(text, file=fp)

        # Valid graph-level labels from torch tensor [n_frames]
        dataset = create_dataset_from_trajectories(
            trajectories=[test_dataset_path],
            topologies=[test_dataset_path],
            cutoff=1.0,
            graph_labels=torch.tensor([1.0, 2.0]),
            show_progress=False,
        )
        assert torch.allclose(dataset[0]["data_list"]["graph_labels"], torch.tensor([[1.0]]))
        assert torch.allclose(dataset[1]["data_list"]["graph_labels"], torch.tensor([[2.0]]))

        # Valid graph-level multitarget labels [n_frames, n_targets]
        dataset = create_dataset_from_trajectories(
            trajectories=[test_dataset_path],
            topologies=[test_dataset_path],
            cutoff=1.0,
            graph_labels=torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
            show_progress=False,
        )
        assert torch.allclose(dataset[0]["data_list"]["graph_labels"], torch.tensor([[1.0], [10.0]]))
        assert torch.allclose(dataset[1]["data_list"]["graph_labels"], torch.tensor([[2.0], [20.0]]))

        # Valid node-level labels [n_frames, n_nodes]
        dataset = create_dataset_from_trajectories(
            trajectories=[test_dataset_path],
            topologies=[test_dataset_path],
            cutoff=1.0,
            graph_labels=torch.tensor([0.0, 1.0]),
            node_labels=torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            show_progress=False,
        )
        assert torch.allclose(dataset[0]["data_list"]["node_labels"], torch.tensor([[0.0], [1.0], [2.0]]))
        assert torch.allclose(dataset[1]["data_list"]["node_labels"], torch.tensor([[3.0], [4.0], [5.0]]))

        # Invalid graph-level labels: wrong number of frames
        try:
            create_dataset_from_trajectories(
                trajectories=[test_dataset_path],
                topologies=[test_dataset_path],
                cutoff=1.0,
                graph_labels=torch.tensor([1.0, 2.0, 3.0]),
                show_progress=False,
            )
            assert False
        except ValueError:
            pass

        # Invalid node-level labels: wrong number of frames
        try:
            create_dataset_from_trajectories(
                trajectories=[test_dataset_path],
                topologies=[test_dataset_path],
                cutoff=1.0,
                graph_labels=torch.tensor([1.0, 2.0]),
                node_labels=torch.tensor([[0.0, 1.0, 2.0]]),
                show_progress=False,
            )
            assert False
        except ValueError:
            pass


def test_dataset_from_xyz():
    from mlcolvar.tests import data_dir

    with data_dir() as data_folder:
        # load single file
        load_args = [{'start' : 0, 'stop' : 2, 'stride' : 1}]
        dataset = create_dataset_from_trajectories(trajectories="Cu.xyz",
                                                folder=data_folder,
                                                topologies=None,
                                                cutoff=3.5,  # Ang
                                                labels=None,
                                                system_selection="index 0",
                                                environment_selection="not index 0",
                                                show_progress=False,
                                                load_args=load_args,
                                                buffer=1,
                                            )
        
        print(dataset)

        # load multiple files
        load_args = [{'start' : 0, 'stop' : 2, 'stride' : 1},
                    {'start' : 0, 'stop' : 4, 'stride' : 2}]
        dataset = create_dataset_from_trajectories(trajectories=["Cu.xyz", "Cu.xyz"],
                                                folder=data_folder,
                                                topologies=None,
                                                cutoff=3.5,  # Ang
                                                labels=None,
                                                system_selection="index 0 or index 1",
                                                environment_selection="not index 0 and not index 1",
                                                show_progress=False,
                                                load_args=load_args,
                                                buffer=1,
                                                )
        print(dataset)