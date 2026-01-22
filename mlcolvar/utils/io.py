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
import tempfile
import urllib.request
from typing import Union, List, Tuple
import mdtraj
from warnings import warn

# Import ASE for xyz to pdb conversion.
try:
    from ase.io import read, write
    from ase import Atoms
except ImportError as e:
    raise ImportError("ASE is required for xyz to pdb conversion.", e)


from mlcolvar.data import DictDataset
from mlcolvar.data.graph.atomic import AtomicNumberTable, Configuration, Configurations
from mlcolvar.data.graph.utils import create_dataset_from_configurations


__all__ = ["load_dataframe", "plumed_to_pandas", "create_dataset_from_files", "create_dataset_from_configurations", "create_dataset_from_trajectories"]


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


def load_dataframe(file_names: Union[str, list],
                   folder: str = None, 
                   start: int = 0, 
                   stop: int = None, 
                   stride: int = 1, 
                   load_args: List[dict] = None, 
                   delete_download: bool = True, 
                   **kwargs,
):
    """Load dataframe(s) from file(s). It can be used also to open files from internet (if the string contains http).
    In case of PLUMED colvar files automatically handles the column names, otherwise it is just a wrapper for pd.load_csv function.

    Parameters
    ----------
    filenames : str or list[str]
        filenames to be loaded
    folder : str, optional
        Common path for the files to be imported, by default None. If set, filenames become 'folder/file_name'.
    start: int, optional
        read from this row, default 0
    stop: int, optional
        read until this row, default None
    stride: int, optional
        read every this number, default 1
    load_args: list[dict], optional
        List of dictionaries with the loading arguments for each file (keys: start,stop,stride and pandas.read_csv options), by default None
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
    default_load_args = {'start' : start, 'stride': stride, 'stop': stop}

    # if it is a single string
    if type(file_names) == str:
        file_names = [file_names]
    elif type(file_names) != list:
        raise TypeError(
            f"only strings or list of strings are supported, not {type(file_names)}."
        )
    
    # set file paths
    if folder is not None:
        file_names = [os.path.join(folder, fname) for fname in file_names]

    # check if per file args are given, otherwise set to {}
    if load_args is None:
        load_args = [default_load_args for _ in file_names]
    else:
        if start != 0 or stride != 1 or stop is not None:
            raise ValueError(
                "Both global and per-file loading parameters have been specified. Either use load_args for per-file parameters or start, stop, stride keywords for global behavior."
            )
        if (not isinstance(load_args, list)) or (len(file_names) != len(load_args)):
            raise TypeError(
                "load_args should be a list of dictionaries of arguments of same length as file_names. If you want to use the same args for all file pass them directly as **kwargs."
            )
        for i,arg in enumerate(load_args):
            for key in default_load_args.keys():
                if key not in arg.keys():
                    load_args[i][key] = default_load_args[key]

    # list of file_names
    df_list = []
    for i, filename in enumerate(file_names):
        # get correct loading args
        start = load_args[i]['start']
        stop = load_args[i]['stop']
        stride = load_args[i]['stride']

        # check if filename is an url
        download = False
        if "http" in filename:
            download = True
            url = filename
            if delete_download:
                temp = tempfile.NamedTemporaryFile()
                filename = temp.name   
            else:
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
                temp.close()
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
    load_args: List[dict] = None,
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
    dataset = DictDataset(dictionary, feature_names=df_data.columns.values, data_type='descriptors')

    if return_dataframe:
        return dataset, df
    else:
        return dataset

def create_pdb_from_xyz(input_filename: str, output_filename: str) -> str:
    """
    Convert the first frame of an XYZ file into a PDB file using ASE.
    This pdb file can then serve as the topology for MDTraj.

    Parameters:
        input_filename: Path to the input .xyz file.
        output_filename: Path to the output .pdb file.

    Returns:
        The path to the generated PDB file.
    """
    atoms: Atoms = read(input_filename, index=0)

    if (atoms.cell == 0).all():
        warn("A topology file was generated from the xyz trajectory file but no cell information were provided!")
    if not atoms.pbc.any():
        warn("A topology file was generated from the xyz trajectory file but no PBC information were provided!")
    elif not atoms.pbc.all():
        warn( f"Partial PBC are not supported! The provided input has pbc {atoms.pbc}")

    write(output_filename, atoms, format='proteindatabank')
    return output_filename



def create_dataset_from_trajectories(
    trajectories: Union[List[str], str],
    top: Union[List[str], str, None],
    cutoff: float,
    buffer: float = 0.0,
    z_table: AtomicNumberTable = None,
    load_args: list = None,
    folder: str = None,
    labels: list = None,
    system_selection: str = None,
    environment_selection: str = None,
    return_trajectories: bool = False,
    remove_isolated_nodes: bool = True,
    show_progress: bool = True,
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
    top: Union[List[str], str, None]
        Path to topology files. Only for .xyz files it can be set to None or empty to generate automatically a topology file.
    cutoff: float (units: Ang)
        The graph cutoff radius.
    buffer: float
        Buffer size used in finding active environment atoms.
    z_table: mlcolvar.graph.data.atomic.AtomicNumberTable
        The atomic number table used to build the node attributes. If not
        given, it will be created from the given trajectories.
    load_args: list[dict], optional
        List of dictionaries for loading options for each file (keys: start,stop,stride), by default None
    folder: str
        Common path for the files to be imported. If set, filenames become
        `folder/file_name`.
    labels: list
        List of labels to be assigned to the given files. by default None.
        If None, it simply enumerates the files.
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

    # check if using truncated graph
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

    # initialize simple labels if not provided
    if labels is None:
        labels = [i for i in range(len(trajectories))]
    else:
        assert len(labels) == len(trajectories), (
            "Number of labels and trajectories must be the same!"
            )

    # check topologies if given, with xyz it can be None
    if top is not None:
        assert len(trajectories) == len(top) or len(top)==1 or isinstance(top, str), (
            'Either a single topology file or as many as the trajectory files must be provided!'
        )

    # ensure trajectories is a list
    if isinstance(trajectories, str):
        trajectories = [trajectories]
    
    # --- Handle topologies input ---
    # Allow top to be None or empty. In that case, create a list of empty strings.
    shared_top = True
    if isinstance(top, str):
        top = [top for _ in trajectories]
    elif top is None or (isinstance(top, list) and len(top) == 0):
        top = ["" for _ in trajectories]
    elif len(top) == 1 and len(trajectories) > 1:
        top = [top for _ in trajectories]
    else: 
        shared_top = False


    # load topologies and trajectories
    topologies = []
    trajectories_in_memory = []
    for i in range(len(trajectories)):
        # =============== PREPARATION ===============
        assert isinstance(trajectories[i], str)

        # check if folder is given
        if folder is not None:
            trajectories[i] = os.path.join(folder, trajectories[i])
            if top[i]:
                top[i] = os.path.join(folder, top[i])
        
        # check if trajectories[i] is an url
        download_traj = False
        if "http" in trajectories[i]:
            download_traj = True
            url = trajectories[i]
            if delete_download:
                temp_traj = tempfile.NamedTemporaryFile(suffix=os.path.splitext(trajectories[i])[1].lower() )
                trajectories[i] = temp_traj.name   
            else:
                trajectories[i] = "tmp_" + trajectories[i].split("/")[-1]
            urllib.request.urlretrieve(url, trajectories[i])

        # check if top[i] is an url
        download_top = False
        if "http" in top[i]:
            download_top = True
            # check if it is really needed to download or top is shared
            if shared_top and i > 0: 
                top[i] = top[0]
            else:
                url = top[i]
                if delete_download:
                    temp_top = tempfile.NamedTemporaryFile(suffix=os.path.splitext(top[i])[1].lower() )
                    top[i] = temp_top.name   
                else:
                    top[i] = "tmp_" + top[i].split("/")[-1]
                urllib.request.urlretrieve(url, top[i])

        # check extension of file, if .xyz create topology file
        _, ext = os.path.splitext(trajectories[i])
        if (ext.lower() == ".xyz") and (not top[i]):
            pdb_file = trajectories[i].replace('.xyz', '_top.pdb')
            top[i] = create_pdb_from_xyz(trajectories[i], pdb_file)


        # =============== LOADING ===============
        # load trajectory
        traj = mdtraj.load(trajectories[i], top=top[i])
        traj.top = mdtraj.core.trajectory.load_topology(top[i])
        
        # mdtraj does not load cell info from xyz, so we use ASE and add it
        _, ext = os.path.splitext(trajectories[i])
        if (ext.lower() == ".xyz"):
            ase_atoms = read(trajectories[i], index=':')
            ase_cells = np.array([a.get_cell().array for a in ase_atoms], dtype=float)
            # the pdb for the topology are in nm, ase work in A so we need to scale it
            traj.unitcell_vectors = ase_cells/10

        if selection is not None:
            subset = traj.top.select(selection)
            assert len(subset) > 0, (
                'No atoms will be selected with selection string '
                + '"{:s}"!'.format(selection)
            )
            traj = traj.atom_slice(subset)
        trajectories_in_memory.append(traj)
        topologies.append(traj.top)

        # remove temporary files from dowload if needed
        if download_traj:
            if delete_download:
                temp_traj.close()
            else:
                print(f"downloaded file ({url}) saved as ({trajectories[i]}).")

        if download_top:
            if shared_top and i == len(trajectories):
                if delete_download:
                    temp_top.close()
                else:
                    print(f"downloaded file ({url}) saved as ({top[i]}).")

    if z_table is None:
        z_table = _z_table_from_top(topologies)

    if save_names:
        atom_names = _names_from_top(topologies)
    else:
        atom_names = None

    # create configurations objects from trajectories
    configurations = []
    for i in range(len(trajectories_in_memory)):
            configuration = _configures_from_trajectory(
                trajectory=trajectories_in_memory[i],
                label=labels[i],
                system_selection=system_selection,
                environment_selection=environment_selection,
                start=load_args[i]['start'] if load_args is not None else 0,
                stop=load_args[i]['stop']  if load_args is not None else None,
                stride=load_args[i]['stride']  if load_args is not None else 1,
                lengths_conversion=lengths_conversion,
            )
            configurations.extend(configuration)

    # convert configurations into DictDataset
    dataset = create_dataset_from_configurations(
        config=configurations,
        z_table=z_table,
        cutoff=cutoff,
        buffer=buffer,
        atom_names=atom_names,
        remove_isolated_nodes=remove_isolated_nodes,
        show_progress=show_progress
    )

    if return_trajectories:
        return dataset, trajectories_in_memory
    else:
        return dataset

def _names_from_top(top: List[mdtraj.Topology] ):
    it = iter(top)
    atom_names = list(next(it).atoms)
    if not all([atom_names == list(n.atoms) for n in it]):
        raise ValueError(
            "The atoms names or their order are different in the topology files. Check or deactivate save_names"
        )
    
    return atom_names


def _z_table_from_top(
    top: List[mdtraj.Topology]
) -> AtomicNumberTable:
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
    z_table = AtomicNumberTable.from_zs(atomic_numbers)
    return z_table


def _configures_from_trajectory(
    trajectory: mdtraj.Trajectory,
    label: int = None,
    system_selection: str = None,
    environment_selection: str = None,
    start: int = 0,
    stop: int = None,
    stride: int = 1,
    lengths_conversion : float = 10.0) -> Configurations:
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
    lengths_conversion: float,
        Conversion factor for length units, by default 10.
        MDTraj uses nanometers, the default sends to Angstroms.
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

    if stop is None:
        stop = len(trajectory)

    configurations = []

    for i in range(start,stop,stride):
        configuration = Configuration(
            atomic_numbers=atomic_numbers,
            positions=trajectory.xyz[i] * lengths_conversion,
            cell=cell[i] * lengths_conversion,
            pbc=pbc,
            graph_labels=label,
            node_labels=None,  # TODO: Add supports for per-node labels.
            system=system_atoms,
            environment=environment_atoms
        )
        configurations.append(configuration)

    return configurations


# =================================================================================================
# ============================================= TESTS =============================================
# =================================================================================================

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

def test_load_dataframe():
    # Test naive single file
    pd_dataframe = load_dataframe(file_names="state_A.dat",
                                  folder="mlcolvar/tests/data",
                                  start=0, 
                                  stop=5,
                                  stride=1,
                                )
    assert(len(pd_dataframe) == 5)

    # Test with global loading parameters
    pd_dataframe = load_dataframe(file_names=["state_A.dat", "state_B.dat", "state_C.dat"],
                                  folder="mlcolvar/tests/data",
                                  start=0, 
                                  stop=5,
                                  stride=1,
                                )
    assert(len(pd_dataframe) == 15)

    # Test with per-file loading parameters
    load_args = [{"start": 0, "stop": 5, "stride": 1},
                 {"start": 0, "stop": 5, "stride": 1},
                 {"start": 0, "stop": 5, "stride": 1}]
    pd_dataframe = load_dataframe(file_names=["state_A.dat", "state_B.dat", "state_C.dat"],
                                  folder="mlcolvar/tests/data",
                                  load_args=load_args,
                                )
    assert(len(pd_dataframe) == 15)

    # Test with per-file loading parameters with default fallback
    load_args = [{"start": 0, "stop": 6, "stride": 2},
                 {"start": 0, "stop": 6, "stride": 2},
                 {"start": 0, "stop": 6}] # this should fall back to default
    pd_dataframe = load_dataframe(file_names=["state_A.dat", "state_B.dat", "state_C.dat"],
                                  folder="mlcolvar/tests/data",
                                  load_args=load_args,
                                )
    assert(len(pd_dataframe) == 12)

    # test wrong length error
    try:
        load_args = [{"start": 0, "stop": 6, "stride": 2},
                 {"start": 0, "stop": 6}] # this should fall back to default
        pd_dataframe = load_dataframe(file_names=["state_A.dat", "state_B.dat", "state_C.dat"],
                                  folder="mlcolvar/tests/data",
                                  load_args=load_args,
                                )
    except TypeError as e:
        print("[TEST LOG] Checked this error: ", e)

    # test load_args and global key conflict error
    try:
        load_args = [{"start": 0, "stop": 6, "stride": 2},
                 {"start": 0, "stop": 6}] # this should fall back to default
        pd_dataframe = load_dataframe(file_names=["state_A.dat", "state_B.dat", "state_C.dat"],
                                  folder="mlcolvar/tests/data",
                                  load_args=load_args,
                                  start=10,
                                )
    except ValueError as e:
        print("[TEST LOG] Checked this error: ", e)

def test_datasesetFromTrajectories():
    create_dataset_from_trajectories(
        trajectories=['r.dcd',
                      'p.dcd'],
        top=['r.pdb', 
             'p.pdb'],
        folder="mlcolvar/tests/data",
        cutoff=8.0,  # Ang
        labels=None,
        system_selection='all and not type H',
        show_progress=False,
    )

    dataset = create_dataset_from_trajectories(
                trajectories=['r.dcd',
                            'p.dcd'],
                top=['r.pdb', 
                    'p.pdb'],
                folder="mlcolvar/tests/data",
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
                top=['r.pdb', 'r.pdb', 
                     'p.pdb', 'p.pdb'],
                folder="mlcolvar/tests/data",
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
            top=[test_dataset_path, test_dataset_path, test_dataset_path],
            cutoff=1.0,
            system_selection=system_selection,
            return_trajectories=True,
            show_progress=False
        )

        assert len(dataset) == 6
        assert dataset.metadata["cutoff"] == 1.0
        assert dataset.metadata["z_table"] == [1, 8]
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
            top=test_dataset_path,
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
                top=[test_dataset_path, test_dataset_path, test_dataset_path],
                cutoff=1.0,
                system_selection='type O and {:s}'.format(system_selection),
                environment_selection='type H and {:s}'.format(system_selection),
                show_progress=False
            )

            for i in range(6):
                check_data_1(dataset[i])

            dataset = create_dataset_from_trajectories(
                trajectories=[test_dataset_path, test_dataset_path, test_dataset_path],
                top=[test_dataset_path, test_dataset_path, test_dataset_path],
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
                assert(torch.allclose(data["data_list"]['node_attrs'], torch.tensor([[1.0], 
                                                                                    [1.0]])
                                    )
                        )

            for i in range(6):
                check_data_2(dataset[i])


def test_dataset_from_xyz():
    # load single file
    load_args = [{'start' : 0, 'stop' : 2, 'stride' : 1}]
    dataset = create_dataset_from_trajectories(trajectories="Cu.xyz",
                                               folder="mlcolvar/tests/data",
                                               top=None,
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
                                               folder="mlcolvar/tests/data",
                                               top=None,
                                               cutoff=3.5,  # Ang
                                               labels=None,
                                               system_selection="index 0 or index 1",
                                               environment_selection="not index 0 and not index 1",
                                               show_progress=False,
                                               load_args=load_args,
                                               buffer=1,
                                              )
    print(dataset)