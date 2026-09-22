import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from mlcolvar.data import DictDataset
from mlcolvar.tests import data_dir


PDB_TEXT = """
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


def _write_pdb(tmpdir):
    path = Path(tmpdir) / "test.pdb"
    path.write_text(PDB_TEXT)
    return str(path)


def test_graph_from_trajectories():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_pdb(tmpdir)

        dataset, trajectories = DictDataset.graph_from_trajectories(
            trajectories=[path, path, path],
            topologies=path,
            cutoff=1.0,
            return_trajectories=True,
            show_progress=False,
        )

    assert len(dataset) == 6
    assert len(trajectories) == 3
    assert dataset.metadata["cutoff"] == 1.0
    assert dataset.metadata["atomic_numbers"] == [1, 8]

    expected = [0, 0, 1, 1, 2, 2]
    for data, label in zip(dataset, expected):
        torch.testing.assert_close(
            data["data_list"]["graph_labels"],
            torch.tensor([[float(label)]]),
        )


def test_trajectory_and_graph_labels():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_pdb(tmpdir)

        dataset = DictDataset.graph_from_trajectories(
            trajectories=[path, path],
            topologies=path,
            cutoff=1.0,
            trajectory_labels=[10, 20],
            show_progress=False,
        )

        expected = [10, 10, 20, 20]
        for data, label in zip(dataset, expected):
            torch.testing.assert_close(
                data["data_list"]["graph_labels"],
                torch.tensor([[float(label)]]),
            )

        dataset = DictDataset.graph_from_trajectories(
            trajectories=[path],
            topologies=path,
            cutoff=1.0,
            graph_labels=np.array([1.0, 2.0]),
            show_progress=False,
        )

        torch.testing.assert_close(
            dataset[0]["data_list"]["graph_labels"],
            torch.tensor([[1.0]]),
        )
        torch.testing.assert_close(
            dataset[1]["data_list"]["graph_labels"],
            torch.tensor([[2.0]]),
        )


def test_node_labels_and_invalid_labels():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_pdb(tmpdir)

        dataset = DictDataset.graph_from_trajectories(
            trajectories=[path],
            topologies=path,
            cutoff=1.0,
            node_labels=torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                ]
            ),
            show_progress=False,
        )

        torch.testing.assert_close(
            dataset[0]["data_list"]["node_labels"],
            torch.tensor([[0.0], [1.0], [2.0]]),
        )

        with pytest.raises(ValueError):
            DictDataset.graph_from_trajectories(
                trajectories=[path],
                topologies=path,
                cutoff=1.0,
                graph_labels=[1.0, 2.0, 3.0],
                show_progress=False,
            )


@pytest.mark.parametrize("backend", ["mdtraj", "ase"])
def test_graph_from_xyz(backend):
    with data_dir() as folder:
        dataset = DictDataset.graph_from_trajectories(
            trajectories="Cu.xyz",
            folder=str(folder),
            cutoff=3.5,
            load_args=[
                {
                    "start": 0,
                    "stop": 2,
                    "stride": 1,
                }
            ],
            backend=backend,
            show_progress=False,
        )

    assert isinstance(dataset, DictDataset)
    assert len(dataset) == 2


def test_graph_from_dcd():
    with data_dir() as folder:
        dataset = DictDataset.graph_from_trajectories(
            trajectories=["r.dcd", "p.dcd"],
            topologies=["r.pdb", "p.pdb"],
            folder=str(folder),
            cutoff=8.0,
            trajectory_labels=[0, 1],
            system_selection="all and not type H",
            load_args=[
                {"start": 0, "stop": 10, "stride": 1},
                {"start": 6, "stop": 10, "stride": 2},
            ],
            show_progress=False,
        )

    assert len(dataset) == 12