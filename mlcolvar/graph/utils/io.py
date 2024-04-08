import torch
import numpy as np
import mdtraj as md

from typing import Union, List, Tuple

from mlcolvar.graph.data import (
    atomic, GraphDataSet, create_dataset_from_configurations
)

__all__ = ['create_dataset_from_trajectories']


def create_dataset_from_trajectories(
    trajectories: Union[List[List[str]], List[str], str],
    top: Union[List[List[str]], List[str], str],
    cutoff: float,
    z_table: atomic.AtomicNumberTable = None,
    folder: str = None,
    create_labels: bool = None,
    system_selection: str = None,
    edge_sender_selection: str = None,
    edge_receiver_selection: str = None,
    return_trajectories: bool = False,
) -> Union[
    GraphDataSet,
    Tuple[GraphDataSet, Union[List[List[md.Trajectory]], List[md.Trajectory]]]
]:
    """
    Create a dataset from a set of trajectory files.

    Parameters
    ----------
    trajectories: Union[List[List[str]], List[str], str]
        Names of trajectories files.
    top: Union[List[List[str]], List[str], str]
        Names of topology files.
    cutoff: float (units: nm)
        The graph cutoff radius.
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
        MDTraj style atom selections [1] of the edge senders. If given, only
        selected atoms will be loaded from the trajectories. This option may
        increase the speed of building graphs.
    edge_sender_selection: str
        MDTraj style atom selections [1] of the edge senders. If given, only
        edges sent by these atoms will be kept in the graph.
    edge_receiver_selection: str
        MDTraj style atom selections [1] of the edge receivers. If given, only
        edges received by these atoms will be kept in the graph.
    return_trajectories: bool
        If also return the loaded trajectory objects.

    Returns
    -------
    dataset: mlcolvar.graph.data.GraphDataSet
        The graph dataset.
    trajectories: Union[List[List[mdtraj.Trajectory]], List[mdtraj.Trajectory]]
        The loaded trajectory objects.

    Notes
    -----
    The selections `edge_sender_selection` and `edge_receiver_selection`
    will be made after the `system_selection` was made. Thus, absolute index
    based edge selections may lead to unwanted results when `system_selection`
    is given.

    References
    ----------
    .. [1] https://www.mdtraj.org/1.9.8.dev0/atom_selection.html
    """
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
            if system_selection is not None:
                for j in range(len(traj)):
                    subset = traj[j].top.select(system_selection)
                    assert len(subset) > 0, (
                        'No atoms will be selected with system selection '
                        + '"{:s}"!'.format(system_selection)
                    )
                    traj[j] = traj[j].atom_slice(subset)
            trajectories_in_memory.append(traj)
            topologies.extend([t.top for t in traj])
        else:
            traj = md.load(trajectories[i], top=top[i])
            traj.top = md.core.trajectory.load_topology(top[i])
            if system_selection is not None:
                subset = traj.top.select(system_selection)
                assert len(subset) > 0, (
                    'No atoms will be selected with system selection '
                    + '"{:s}"!'.format(system_selection)
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
                    edge_sender_selection,
                    edge_receiver_selection,
                )
                configurations.extend(configuration)
        else:
            configuration = _configures_from_trajectory(
                trajectories_in_memory[i],
                i,
                edge_sender_selection,
                edge_receiver_selection,
            )
            configurations.extend(configuration)

    dataset = create_dataset_from_configurations(
        configurations, z_table, cutoff
    )

    if return_trajectories:
        return dataset, trajectories_in_memory
    else:
        return dataset


def _z_table_from_top(top: List[md.Topology]) -> atomic.AtomicNumberTable:
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
    z_table = atomic.AtomicNumberTable.from_zs(atomic_numbers)
    return z_table


def _configures_from_trajectory(
    trajectory: md.Trajectory,
    label: int = None,
    edge_sender_selection: str = None,
    edge_receiver_selection: str = None,
) -> atomic.Configurations:
    """
    Create configurations from one trajectory.

    Parameters
    ----------
    trajectory: mdtraj.Trajectory
        The MDTraj Trajectory object.
    label: int
        The graph label.
    edge_sender_selection: str
        MDTraj style atom selections [1] of the edge senders. If given, only
        edges sent by these atoms will be kept in the graph.
    edge_receiver_selection: str
        MDTraj style atom selections [1] of the edge receivers. If given, only
        edges received by these atoms will be kept in the graph.
    """
    if label is not None:
        label = np.array([label])

    if edge_sender_selection is not None:
        edge_senders = trajectory.top.select(edge_sender_selection)
        assert len(edge_senders) > 0, (
            'No atoms will be selected with edge_sender_selection selection '
            + '"{:s}"!'.format(edge_sender_selection)
        )
    else:
        edge_senders = None
    if edge_receiver_selection is not None:
        edge_receivers = trajectory.top.select(edge_receiver_selection)
        assert len(edge_receivers) > 0, (
            'No atoms will be selected with edge_receiver_selection selection '
            + '"{:s}"!'.format(edge_receiver_selection)
        )
    else:
        edge_receivers = None

    atomic_numbers = [a.element.number for a in trajectory.top.atoms]
    if trajectory.unitcell_vectors is not None:
        pbc = [True] * 3
        cell = trajectory.unitcell_vectors
    else:
        pbc = [False] * 3
        cell = [None] * len(trajectory)

    configurations = []
    for i in range(len(trajectory)):
        configuration = atomic.Configuration(
            atomic_numbers=atomic_numbers,
            positions=trajectory.xyz[i],
            cell=cell[i],
            pbc=pbc,
            graph_labels=label,
            node_labels=None,  # TODO: Add supports for per-node labels.
            edge_senders=edge_senders,
            edge_receivers=edge_receivers,
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
        0.1,
        system_selection=system_selection,
        return_trajectories=True
    )

    assert len(dataset) == 6
    assert dataset.cutoff == 0.1
    assert dataset.atomic_numbers == [1, 8]
    assert len(trajectories) == 2
    assert len(trajectories[0]) == 2
    assert len(trajectories[1]) == 2
    assert len(trajectories[1][0]) == 2
    assert len(trajectories[1][1]) == 2

    assert dataset[0]['graph_labels'] == torch.tensor([0.0])
    assert dataset[1]['graph_labels'] == torch.tensor([0.0])
    assert dataset[2]['graph_labels'] == torch.tensor([1.0])
    assert dataset[3]['graph_labels'] == torch.tensor([1.0])
    assert dataset[4]['graph_labels'] == torch.tensor([1.0])
    assert dataset[5]['graph_labels'] == torch.tensor([1.0])

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
                [0.0, 0.2, 0.0],
                [0.0, -0.2, 0.0],
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
                [0.07, 0.07, 0.0],
                [0.07, -0.07, 0.0],
            ])
        ).all()
        assert (
            data['cell'] == torch.tensor([
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.0, 0.2],
            ])
        ).all()
        assert (
            data['node_attrs'] == torch.tensor([
                [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]
            ])
        ).all()
        assert (data['n_receivers'] == torch.tensor([[3]])).all()

    for i in range(6):
        check_data_1(dataset[i])

    dataset = create_dataset_from_trajectories(
        ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
        ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
        0.1,
        system_selection=system_selection,
        edge_sender_selection='element O',
    )

    def check_data_2(data) -> None:
        assert (data['edge_index'] == torch.tensor([[0, 0], [2, 1]])).all()
        assert (
            data['node_attrs'] == torch.tensor([
                [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]
            ])
        ).all()
        # NOTE: This is a very bad testing case: as you can see, the oxygen
        # is one of the receivers, however no edge is pointing to it.
        # This is the risk when you have too few edge senders.
        assert (data['n_receivers'] == torch.tensor([[3]])).all()

    for i in range(6):
        check_data_2(dataset[i])

    dataset = create_dataset_from_trajectories(
        ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
        ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
        0.1,
        system_selection=system_selection,
        edge_receiver_selection='element H',
    )

    def check_data_3(data) -> None:
        assert (
            data['edge_index'] == torch.tensor([[0, 0, 1, 2], [2, 1, 2, 1]])
        ).all()
        assert (
            data['node_attrs'] == torch.tensor([
                [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]
            ])
        ).all()
        assert (data['n_receivers'] == torch.tensor([[2]])).all()
        assert (data['receiver_masks'] == torch.tensor([[0], [1], [1]])).all()

    for i in range(6):
        check_data_3(dataset[i])

    dataset = create_dataset_from_trajectories(
        ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
        ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
        0.1,
        system_selection=system_selection,
        edge_sender_selection='element O',
        edge_receiver_selection='element H',
    )

    def check_data_4(data) -> None:
        assert (data['edge_index'] == torch.tensor([[0, 0], [2, 1]])).all()
        assert (
            data['node_attrs'] == torch.tensor([
                [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]
            ])
        ).all()
        assert (data['n_receivers'] == torch.tensor([[2]])).all()
        assert (data['receiver_masks'] == torch.tensor([[0], [1], [1]])).all()

    for i in range(6):
        check_data_4(dataset[i])

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
    test_create_dataset_from_trajectories(text, "not resname XXXX")

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
    test_create_dataset_from_trajectories(text, "not resname XXXX")
