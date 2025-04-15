import pytest
import urllib
from mlcolvar.utils.io import load_dataframe
from mlcolvar.utils.io import test_datasetFromFile
from mlcolvar.utils.io import test_datasesetFromTrajectories
from mlcolvar.utils.io import test_create_dataset_from_trajectories
from mlcolvar.utils.io import test_dataset_from_xyz

example_files = {
    "str": "mlcolvar/tests/data/state_A.dat",
    "list": ["mlcolvar/tests/data/state_A.dat", "mlcolvar/tests/data/state_B.dat"],
    "url": "https://raw.githubusercontent.com/luigibonati/mlcolvar/main/mlcolvar/tests/data/2d_model/COLVAR_stateA",
}


@pytest.mark.parametrize("file_type", ["str", "list", "url"])
def test_loadDataframe(file_type):
    filename = example_files[file_type]
    if file_type == "url":
        # disable test if connection is not available
        try:
            urllib.request.urlopen(filename)
        except urllib.error.URLError:
            pytest.skip("internet not available")

    df = load_dataframe(filename, start=0, stop=10, stride=1)


if __name__ == "__main__":
    test_dataset_from_xyz()
    test_datasetFromFile()
    test_datasesetFromTrajectories()
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