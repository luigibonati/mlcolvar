import torch
import numpy as np
import mdtraj as md
from typing import Union, List, Tuple

from mlcolvar.graph import data as gdata

"""
Some I/O things.
"""

__all__ = ['create_dataset_from_trajectories']


def create_dataset_from_trajectories(
    trajectories: Union[List[List[str]], List[str], str],
    top: Union[List[List[str]], List[str], str],
    cutoff: float,
    buffer: float = 0.0,
    z_table: gdata.atomic.AtomicNumberTable = None,
    folder: str = None,
    create_labels: bool = None,
    system_selection: str = None,
    environment_selection: str = None,
    return_trajectories: bool = False,
    remove_isolated_nodes: bool = True,
    show_progress: bool = True
) -> Union[
    gdata.GraphDataSet,
    Tuple[
        gdata.GraphDataSet,
        Union[List[List[md.Trajectory]], List[md.Trajectory]]
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
        Assign a label to each file, default True if more than one set of files
        is given, otherwise False.
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
                md.load(trajectories[i][j], top=top[i][j])
                for j in range(len(trajectories[i]))
            ]
            for t in traj:
                t.top = md.core.trajectory.load_topology(top[i][j])
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
            traj = md.load(trajectories[i], top=top[i])
            traj.top = md.core.trajectory.load_topology(top[i])
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
                    i,  # NOTE: all these configurations have a label `i`
                    system_selection,
                    environment_selection,
                )
                configurations.extend(configuration)
        else:
            configuration = _configures_from_trajectory(
                trajectories_in_memory[i],
                i,
                system_selection,
                environment_selection,
            )
            configurations.extend(configuration)

    dataset = gdata.create_dataset_from_configurations(
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
    top: List[md.Topology]
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
    atomic_numbers = np.array(atomic_numbers, dtype=int)
    z_table = gdata.atomic.AtomicNumberTable.from_zs(atomic_numbers)
    return z_table


def _configures_from_trajectory(
    trajectory: md.Trajectory,
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


def test_create_dataset_from_trajectories(
    text: str, system_selection: str
) -> None:
    with open('test_dataset.pdb', 'w') as fp:
        print(text, file=fp)

    dataset, trajectories = create_dataset_from_trajectories(
        ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
        ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
        1.0,
        system_selection=system_selection,
        return_trajectories=True,
        show_progress=False
    )

    assert len(dataset) == 6
    assert dataset.cutoff == 1.0
    assert dataset.atomic_numbers == [1, 8]
    assert len(trajectories) == 2
    assert len(trajectories[0]) == 2
    assert len(trajectories[1]) == 2
    assert len(trajectories[1][0]) == 2
    assert len(trajectories[1][1]) == 2

    assert dataset[0]['graph_labels'] == torch.tensor([[0.0]])
    assert dataset[1]['graph_labels'] == torch.tensor([[0.0]])
    assert dataset[2]['graph_labels'] == torch.tensor([[1.0]])
    assert dataset[3]['graph_labels'] == torch.tensor([[1.0]])
    assert dataset[4]['graph_labels'] == torch.tensor([[1.0]])
    assert dataset[5]['graph_labels'] == torch.tensor([[1.0]])

    def check_data_1(data) -> None:
        assert (
            data['edge_index'] == torch.tensor(
                [[0, 0, 1, 1, 2, 2], [2, 1, 0, 2, 1, 0]]
            )
        ).all()
        assert (
            data['shifts'] == torch.tensor([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, 0.0],
            ])
        ).all()
        assert (
            data['unit_shifts'] == torch.tensor([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0],
            ])
        ).all()
        assert (
            data['positions'] == torch.tensor([
                [0.0, 0.0, 0.0],
                [0.7, 0.7, 0.0],
                [0.7, -0.7, 0.0],
            ])
        ).all()
        assert (
            data['cell'] == torch.tensor([
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ])
        ).all()
        assert (
            data['node_attrs'] == torch.tensor([
                [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]
            ])
        ).all()

    for i in range(6):
        check_data_1(dataset[i])

    if system_selection is not None:

        dataset = create_dataset_from_trajectories(
            ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
            ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
            1.0,
            system_selection='type O and {:s}'.format(system_selection),
            environment_selection='type H and {:s}'.format(system_selection),
            show_progress=False
        )

        for i in range(6):
            check_data_1(dataset[i])

        dataset = create_dataset_from_trajectories(
            ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
            ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
            1.0,
            system_selection='name H1 and {:s}'.format(system_selection),
            environment_selection='name H2 and {:s}'.format(system_selection),
            show_progress=False
        )

        def check_data_2(data) -> None:
            assert (data['edge_index'] == torch.tensor([[0, 1], [1, 0]])).all()
            assert (
                data['shifts'] == torch.tensor([
                    [0.0, 2.0, 0.0], [0.0, -2.0, 0.0]
                ])
            ).all()
            assert (
                data['unit_shifts'] == torch.tensor([
                    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                ])
            ).all()
            assert (
                data['positions'] == torch.tensor([
                    [0.7, 0.7, 0.0], [0.7, -0.7, 0.0],
                ])
            ).all()
            assert (
                data['cell'] == torch.tensor([
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 2.0],
                ])
            ).all()
            assert (
                data['node_attrs'] == torch.tensor([
                    [1.0], [1.0]
                ])
            ).all()

        for i in range(6):
            check_data_2(dataset[i])

    __import__('os').remove('test_dataset.pdb')


if __name__ == '__main__':
    text = """
CRYST1    2.000    2.000    2.000  90.00  90.00  90.00 P 1           1
ATOM      1  OH2 TIP3W   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  H1  TIP3W   1       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      3  H2  TIP3W   1       0.700  -0.700   0.000  1.00  0.00      WT1  H
ENDMODEL
ATOM      1  OH2 TIP3W   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  H1  TIP3W   1       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      3  H2  TIP3W   1       0.700  -0.700   0.000  1.00  0.00      WT1  H
END
"""
    test_create_dataset_from_trajectories(text, None)

    text = """
CRYST1    2.000    2.000    2.000  90.00  90.00  90.00 P 1           1
ATOM      1  OH2 TIP3W   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  H1  TIP3W   1       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      3  H2  TIP3W   1       0.700  -0.700   0.000  1.00  0.00      WT1  H
ATOM      4  OH2 XXXXW   2       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      5  H1  XXXXW   2       0.300   0.300   0.000  1.00  0.00      WT1  H
ATOM      6  H2  XXXXW   2       0.300  -0.300   0.000  1.00  0.00      WT1  H
ENDMODEL
ATOM      1  OH2 TIP3W   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  H1  TIP3W   1       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      3  H2  TIP3W   1       0.700  -0.700   0.000  1.00  0.00      WT1  H
ATOM      4  OH2 XXXXW   2       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      5  H1  XXXXW   2       0.300   0.300   0.000  1.00  0.00      WT1  H
ATOM      6  H2  XXXXW   2       0.300  -0.300   0.000  1.00  0.00      WT1  H
END
"""
    test_create_dataset_from_trajectories(text, 'not resname XXXX')

    text = """
CRYST1    2.000    2.000    2.000  90.00  90.00  90.00 P 1           1
ATOM      1  OH2 XXXXW   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  OH2 TIP3W   2       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      3  H1  XXXXW   1       0.300   0.300   0.000  1.00  0.00      WT1  H
ATOM      4  H1  TIP3W   2       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      5  H2  XXXXW   1       0.300  -0.300   0.000  1.00  0.00      WT1  H
ATOM      6  H2  TIP3W   2       0.700  -0.700   0.000  1.00  0.00      WT1  H
ENDMODEL
ATOM      1  OH2 XXXXW   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  OH2 TIP3W   2       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      3  H1  XXXXW   1       0.300   0.300   0.000  1.00  0.00      WT1  H
ATOM      4  H1  TIP3W   2       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      5  H2  XXXXW   1       0.300  -0.300   0.000  1.00  0.00      WT1  H
ATOM      6  H2  TIP3W   2       0.700  -0.700   0.000  1.00  0.00      WT1  H
END
"""
    test_create_dataset_from_trajectories(text, 'not resname XXXX')
