import pytest

from mlcolvar.io.colvar import load_dataframe
from mlcolvar.tests import data_dir


def test_load_dataframe():
    with data_dir() as data_folder:
        data_folder = str(data_folder)

        # Single file
        dataframe = load_dataframe(
            file_names="state_A.dat",
            folder=data_folder,
            start=0,
            stop=5,
            stride=1,
        )
        assert len(dataframe) == 5

        # Multiple files with global loading parameters
        dataframe = load_dataframe(
            file_names=[
                "state_A.dat",
                "state_B.dat",
                "state_C.dat",
            ],
            folder=data_folder,
            start=0,
            stop=5,
            stride=1,
        )
        assert len(dataframe) == 15

        # Per-file loading parameters
        load_args = [
            {"start": 0, "stop": 5, "stride": 1},
            {"start": 0, "stop": 5, "stride": 1},
            {"start": 0, "stop": 5, "stride": 1},
        ]

        dataframe = load_dataframe(
            file_names=[
                "state_A.dat",
                "state_B.dat",
                "state_C.dat",
            ],
            folder=data_folder,
            load_args=load_args,
        )
        assert len(dataframe) == 15

        # Missing stride falls back to the default
        load_args = [
            {"start": 0, "stop": 6, "stride": 2},
            {"start": 0, "stop": 6, "stride": 2},
            {"start": 0, "stop": 6},
        ]

        dataframe = load_dataframe(
            file_names=[
                "state_A.dat",
                "state_B.dat",
                "state_C.dat",
            ],
            folder=data_folder,
            load_args=load_args,
        )
        assert len(dataframe) == 12


def test_load_dataframe_invalid_load_args():
    with data_dir() as data_folder:
        data_folder = str(data_folder)

        with pytest.raises(TypeError):
            load_dataframe(
                file_names=[
                    "state_A.dat",
                    "state_B.dat",
                    "state_C.dat",
                ],
                folder=data_folder,
                load_args=[
                    {"start": 0, "stop": 6, "stride": 2},
                    {"start": 0, "stop": 6},
                ],
            )

        with pytest.raises(ValueError):
            load_dataframe(
                file_names=[
                    "state_A.dat",
                    "state_B.dat",
                    "state_C.dat",
                ],
                folder=data_folder,
                load_args=[
                    {"start": 0, "stop": 6, "stride": 2},
                    {"start": 0, "stop": 6},
                    {"start": 0, "stop": 6},
                ],
                start=10,
            )