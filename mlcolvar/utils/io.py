"""Input/output functions."""
try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "pandas is required to use the i/o utils (mlcolvar.utils.io)\n", e
    )

import numpy as np
import torch
import os
import urllib.request
from typing import Union, List, Tuple
import mdtraj

from mlcolvar.data import graph as gdata

from mlcolvar.data import DictDataset

__all__ = ["load_dataframe", "plumed_to_pandas", "create_dataset_from_files"]


def is_plumed_file(filename):
    """
    Check if given file is in PLUMED format.

    Parameters
    ----------
    filename : string, optional
        PLUMED output file

    Returns
    -------
    bool
        wheter is a plumed output file
    """
    headers = pd.read_csv(filename, sep=" ", skipinitialspace=True, nrows=0)
    is_plumed = True if " ".join(headers.columns[:2]) == "#! FIELDS" else False
    return is_plumed


def plumed_to_pandas(filename="./COLVAR"):
    """
    Load a PLUMED file and save it to a dataframe.

    Parameters
    ----------
    filename : string, optional
        PLUMED output file

    Returns
    -------
    df : DataFrame
        Collective variables dataframe
    """
    skip_rows = 1
    # Read header
    headers = pd.read_csv(filename, sep=" ", skipinitialspace=True, nrows=0)
    # Discard #! FIELDS
    headers = headers.columns[2:]
    # Load dataframe and use headers for columns names
    df = pd.read_csv(
        filename,
        sep=" ",
        skipinitialspace=True,
        header=None,
        skiprows=range(skip_rows),
        names=headers,
        comment="#",
    )

    return df


def load_dataframe(
    file_names, start=0, stop=None, stride=1, delete_download=True, **kwargs
):
    """Load dataframe(s) from file(s). It can be used also to open files from internet (if the string contains http).
    In case of PLUMED colvar files automatically handles the column names, otherwise it is just a wrapper for pd.load_csv function.

    Parameters
    ----------
    filenames : str or list[str]
        filenames to be loaded
    start: int, optional
        read from this row, default 0
    stop: int, optional
        read until this row, default None
    stride: int, optional
        read every this number, default 1
    delete_download: bool, optinal
        whether to delete the downloaded file after it has been loaded, default True.
    kwargs:
        keyword arguments passed to pd.load_csv function

    Returns
    -------
    pandas.DataFrame
        Dataframe

    Raises
    ------
    TypeError
        if data is not a valid type
    """

    # if it is a single string
    if type(file_names) == str:
        file_names = [file_names]
    elif type(file_names) != list:
        raise TypeError(
            f"only strings or list of strings are supported, not {type(file_names)}."
        )

    # list of file_names
    df_list = []
    for i, filename in enumerate(file_names):
        # check if filename is an url
        download = False
        if "http" in filename:
            download = True
            url = filename
            filename = "tmp_" + filename.split("/")[-1]
            urllib.request.urlretrieve(url, filename)

        # check if file is in PLUMED format
        if is_plumed_file(filename):
            df_tmp = plumed_to_pandas(filename)
            df_tmp["walker"] = [i for _ in range(len(df_tmp))]
            df_tmp = df_tmp.iloc[start:stop:stride, :]
            df_list.append(df_tmp)

        # else use read_csv with optional kwargs
        else:
            df_tmp = pd.read_csv(filename, **kwargs)
            df_tmp["walker"] = [i for _ in range(len(df_tmp))]
            df_tmp = df_tmp.iloc[start:stop:stride, :]
            df_list.append(df_tmp)

        # delete temporary data if necessary
        if download:
            if delete_download:
                os.remove(filename)
            else:
                print(f"downloaded file ({url}) saved as ({filename}).")

        # concatenate
        df = pd.concat(df_list)
        df.reset_index(drop=True, inplace=True)

    return df


def create_dataset_from_files(
    file_names: Union[list, str],
    folder: str = None,
    create_labels: bool = None,
    load_args: list = None,
    filter_args: dict = None,
    modifier_function=None,
    return_dataframe: bool = False,
    verbose: bool = True,
    **kwargs,
):
    """
    Initialize a dataset from (a list of) files. Suitable for supervised/unsupervised tasks.

    Parameters
    ----------
    file_names : list
        Names of files from which import the data
    folder : str, optional
        Common path for the files to be imported, by default None. If set, filenames become 'folder/file_name'.
    create_labels: bool, optional
        Assign a label to each file, default True if more than a file is given, otherwise False
    load_args: list[dict], optional
        List of dictionaries with the arguments passed to load_dataframe function for each file (keys: start,stop,stride and pandas.read_csv options), by default None
    filter_args: dict, optional
        Dictionary of arguments which are passed to df.filter() to select descriptors (keys: items, like, regex), by default None
        Note that 'time' and '*.bias' columns are always discarded.
    return_dataframe : bool, optional
        Return also the imported Pandas dataframe for convenience, by default False
    modifier_function : function, optional
        Function to be applied to the input data, by default None.
    verbose : bool, optional
        Print info on the datasets, by default True
    kwargs : optional
        args passed to mlcolvar.utils.io.load_dataframe

    Returns
    -------
    torch.Dataset
        Torch labeled dataset of the given data
    optional, pandas.Dataframe
        Pandas dataframe of the given data #TODO improve

    See also
    --------
    mlcolvar.utils.io.load_dataframe
        Function that is used to load the files

    """
    if isinstance(file_names, str):
        file_names = [file_names]

    num_files = len(file_names)

    # set file paths
    if folder is not None:
        file_names = [os.path.join(folder, fname) for fname in file_names]

    # check if per file args are given, otherwise set to {}
    if load_args is None:
        load_args = [{} for _ in file_names]
    else:
        if (not isinstance(load_args, list)) or (len(file_names) != len(load_args)):
            raise TypeError(
                "load_args should be a list of dictionaries of arguments of same length as file_names. If you want to use the same args for all file pass them directly as **kwargs."
            )

    # check if create_labels if given, otherwise set it to True if more than one file is given
    if create_labels is None:
        create_labels = False if len(file_names) == 1 else True

    # initialize pandas dataframe
    df = pd.DataFrame()

    # load data
    for i in range(num_files):
        df_tmp = load_dataframe(file_names[i], **load_args[i], **kwargs)

        # add label in the dataframe
        if create_labels:
            df_tmp["labels"] = i
        if verbose:
            print(f"Class {i} dataframe shape: ", np.shape(df_tmp))

        # update collective dataframe
        df = pd.concat([df, df_tmp], ignore_index=True)

    # filter inputs
    df_data = df.filter(**filter_args) if filter_args is not None else df.copy()
    df_data = df_data.filter(regex="^(?!.*labels)^(?!.*time)^(?!.*bias)^(?!.*walker)")

    if verbose:
        print(f"\n - Loaded dataframe {df.shape}:", list(df.columns))
        print(f" - Descriptors {df_data.shape}:", list(df_data.columns))

    # apply transformation
    if modifier_function is not None:
        df_data = df_data.apply(modifier_function)

    # create DictDataset
    dictionary = {"data": torch.Tensor(df_data.values)}
    if create_labels:
        dictionary["labels"] = torch.Tensor(df["labels"].values)
    dataset = DictDataset(dictionary, feature_names=df_data.columns.values)

    if return_dataframe:
        return dataset, df
    else:
        return dataset

def create_dataset_from_trajectories(
    trajectories: Union[List[List[str]], List[str], str],
    top: Union[List[List[str]], List[str], str],
    cutoff: float,
    buffer: float = 0.0,
    z_table: gdata.atomic.AtomicNumberTable = None,
    folder: str = None,
    create_labels: bool = True,
    system_selection: str = None,
    environment_selection: str = None,
    return_trajectories: bool = False,
    remove_isolated_nodes: bool = True,
    show_progress: bool = True
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
    trajectories: Union[List[List[str]], List[str], str]
        Names of trajectories files.
    top: Union[List[List[str]], List[str], str]
        Names of topology files.
    cutoff: float (units: Ang)
        The graph cutoff radius.
    buffer: float
        Buffer size used in finding active environment atoms.
    z_table: mlcolvar.graph.data.atomic.AtomicNumberTable
        The atomic number table used to build the node attributes. If not
        given, it will be created from the given trajectories.
    folder: str
        Common path for the files to be imported. If set, filenames become
        `folder/file_name`.
    create_labels: bool
        Assign a label to each file according to the total number of files.
        If False, labels of all files will be `-1`.
    system_selection: str
        MDTraj style atom selections [1] of the system atoms. If given, only
        selected atoms will be loaded from the trajectories. This option may
        increase the speed of building graphs.
    environment_selection: str
        MDTraj style atom selections [1] of the environment atoms. If given,
        only the system atoms and [the environment atoms within the cutoff
        radius of the system atoms] will be kept in the graph.
    return_trajectories: bool
        If also return the loaded trajectory objects.
    remove_isolated_nodes: bool
        If remove isolated nodes from the dataset.
    show_progress: bool
        If show the progress bar.

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

    if environment_selection is not None:
        assert system_selection is not None, (
            'the `environment_selection` argument requires the'
            + '`system_selection` argument to be defined!'
        )
        selection = '({:s}) or ({:s})'.format(
            system_selection, environment_selection
        )
    elif system_selection is not None:
        selection = system_selection
    else:
        selection = None

    if environment_selection is None:
        assert buffer == 0, (
            'Not `environment_selection` given! Cannot define buffer size!'
        )

    # fmt: off
    assert type(trajectories) is type(top), (
        'The `trajectories` and `top` parameters should have the same type!'
    )
    if isinstance(trajectories, str):
        trajectories = [trajectories]
        top = [top]
    assert len(trajectories) == len(top), (
        'Numbers of trajectories and topology files should be the same!'
    )
    # fmt: on

    for i in range(len(trajectories)):
        assert type(trajectories[i]) is type(top[i]), (
            'Each element of `trajectories` and `top` parameters '
            + 'should have the same type!'
        )
        if isinstance(trajectories[i], list):
            assert len(trajectories[i]) == len(top[i]), (
                'Numbers of trajectories and topology files should be '
                + 'the same!'
            )
            for j in range(len(trajectories[i])):
                if folder is not None:
                    trajectories[i][j] = folder + '/' + trajectories[i][j]
                    top[i][j] = folder + '/' + top[i][j]
                assert isinstance(trajectories[i][j], str)
                assert isinstance(top[i][j], str)
        else:
            if folder is not None:
                trajectories[i] = folder + '/' + trajectories[i]
                top[i] = folder + '/' + top[i]
            assert isinstance(trajectories[i], str)
            assert isinstance(top[i], str)

    topologies = []
    trajectories_in_memory = []

    for i in range(len(trajectories)):
        if isinstance(trajectories[i], list):
            traj = [
                mdtraj.load(trajectories[i][j], top=top[i][j])
                for j in range(len(trajectories[i]))
            ]
            for t in traj:
                t.top = mdtraj.core.trajectory.load_topology(top[i][j])
            if selection is not None:
                for j in range(len(traj)):
                    subset = traj[j].top.select(selection)
                    assert len(subset) > 0, (
                        'No atoms will be selected with selection string '
                        + '"{:s}"!'.format(selection)
                    )
                    traj[j] = traj[j].atom_slice(subset)
            trajectories_in_memory.append(traj)
            topologies.extend([t.top for t in traj])
        else:
            traj = mdtraj.load(trajectories[i], top=top[i])
            traj.top = mdtraj.core.trajectory.load_topology(top[i])
            if selection is not None:
                subset = traj.top.select(selection)
                assert len(subset) > 0, (
                    'No atoms will be selected with selection string '
                    + '"{:s}"!'.format(selection)
                )
                traj = traj.atom_slice(subset)
            trajectories_in_memory.append(traj)
            topologies.append(traj.top)

    if z_table is None:
        z_table = _z_table_from_top(topologies)

    configurations = []
    for i in range(len(trajectories_in_memory)):
        if isinstance(trajectories_in_memory[i], list):
            for j in range(len(trajectories_in_memory[i])):
                configuration = _configures_from_trajectory(
                    trajectories_in_memory[i][j],
                    i if create_labels else -1,  # NOTE: all these configurations have a label `i`
                    system_selection,
                    environment_selection,
                )
                configurations.extend(configuration)
        else:
            configuration = _configures_from_trajectory(
                trajectories_in_memory[i],
                i if create_labels else -1,
                system_selection,
                environment_selection,
            )
            configurations.extend(configuration)

    dataset = gdata.dataset.create_dataset_from_configurations(
        configurations,
        z_table,
        cutoff,
        buffer,
        remove_isolated_nodes,
        show_progress
    )

    if return_trajectories:
        return dataset, trajectories_in_memory
    else:
        return dataset
    

def _z_table_from_top(
    top: List[mdtraj.Topology]
) -> gdata.atomic.AtomicNumberTable:
    """
    Create an atomic number table from the topologies.

    Parameters
    ----------
    top: List[mdtraj.Topology]
        The topology objects.
    """
    atomic_numbers = []
    for t in top:
        atomic_numbers.extend([a.element.number for a in t.atoms])
    # atomic_numbers = np.array(atomic_numbers, dtype=int)
    z_table = gdata.atomic.AtomicNumberTable.from_zs(atomic_numbers)
    return z_table


def _configures_from_trajectory(
    trajectory: mdtraj.Trajectory,
    label: int = None,
    system_selection: str = None,
    environment_selection: str = None,
) -> gdata.atomic.Configurations:
    """
    Create configurations from one trajectory.

    Parameters
    ----------
    trajectory: mdtraj.Trajectory
        The MDTraj Trajectory object.
    label: int
        The graph label.
    system_selection: str
        MDTraj style atom selections of the system atoms. If given, only
        selected atoms will be loaded from the trajectories. This option may
        increase the speed of building graphs.
    environment_selection: str
        MDTraj style atom selections of the environment atoms. If given,
        only the system atoms and [the environment atoms within the cutoff
        radius of the system atoms] will be kept in the graph.
    """
    if label is not None:
        label = np.array([[label]])

    if system_selection is not None and environment_selection is not None:
        system_atoms = trajectory.top.select(system_selection)
        assert len(system_atoms) > 0, (
            'No atoms will be selected with `system_selection`: '
            + '"{:s}"!'.format(system_selection)
        )
        environment_atoms = trajectory.top.select(environment_selection)
        assert len(environment_atoms) > 0, (
            'No atoms will be selected with `environment_selection`: '
            + '"{:s}"!'.format(environment_selection)
        )
    else:
        system_atoms = None
        environment_atoms = None

    atomic_numbers = [a.element.number for a in trajectory.top.atoms]
    if trajectory.unitcell_vectors is not None:
        pbc = [True] * 3
        cell = trajectory.unitcell_vectors
    else:
        pbc = [False] * 3
        cell = [None] * len(trajectory)

    configurations = []
    for i in range(len(trajectory)):
        configuration = gdata.atomic.Configuration(
            atomic_numbers=atomic_numbers,
            positions=trajectory.xyz[i] * 10,
            cell=cell[i] * 10,
            pbc=pbc,
            graph_labels=label,
            node_labels=None,  # TODO: Add supports for per-node labels.
            system=system_atoms,
            environment=environment_atoms
        )
        configurations.append(configuration)

    return configurations

def test_datasetFromFile():
    # Test with unlabeled dataset
    torch_dataset, pd_dataframe = create_dataset_from_files(
        file_names=["state_A.dat", "state_B.dat", "state_C.dat"],
        folder="mlcolvar/tests/data",
        create_labels=False,
        load_args=None,
        filter_args=None,
        return_dataframe=True,
        start=0,  # kwargs to load_dataframe
        stop=5,
        stride=1,
    )

    # Test no regex on two states
    create_dataset_from_files(
        file_names=["state_A.dat", "state_B.dat"],
        folder="mlcolvar/tests/data",
        create_labels=True,
        load_args=None,
        filter_args=None,
        return_dataframe=True,
        start=0,  # kwargs to load_dataframe
        stop=5,
        stride=1,
    )

    # Test with filter regex on two states
    dataset = create_dataset_from_files(
        file_names=["state_A.dat", "state_B.dat"],
        folder="mlcolvar/tests/data",
        create_labels=True,
        load_args=None,
        filter_args={"regex": "n|o"},
        return_dataframe=False,
        start=0,  # kwargs to load_dataframe
        stop=5,
        stride=1,
    )

    def test_modifier(x):
        return x**2

    # Test with filter regex on two states with modifier
    create_dataset_from_files(
        file_names=["state_A.dat", "state_B.dat"],
        folder="mlcolvar/tests/data",
        create_labels=True,
        load_args=None,
        filter_args={"regex": "n|o"},
        modifier_function=test_modifier,
        return_dataframe=True,
        start=0,  # kwargs to load_dataframe
        stop=5,
        stride=1,
    )


if __name__ == "__main__":
    test_datasetFromFile()