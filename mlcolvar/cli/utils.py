"""Shared helpers for mlcolvar command-line interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from mlcolvar.utils.io.colvar import load_dataframe


def parse_min_max_bounds(values: Sequence[float], dimensions: int, name: str):
    # CLI bounds are passed as min/max pairs, one pair for each selected CV dimension.
    expected = 2 * dimensions
    if len(values) != expected:
        raise ValueError(f"Expected {expected} values for {name} with {dimensions}D data "
                         f"(min max for each dimension), got {len(values)}.")

    if dimensions == 1:
        return (values[0], values[1])

    return [(values[i], values[i + 1]) for i in range(0, expected, 2)]


def load_colvar_data(file_names: Sequence[str],
                     fields: Sequence[str],
                     bias_fields: Sequence[str] | None,
                     start: int,
                     stop: int | None,
                     stride: int,
                     time_field: str | None = None,
                     flatten_single_cv: bool = False):
    # load_dataframe understands PLUMED COLVAR headers and returns named columns.
    dataframe = load_dataframe(file_names=file_names[0] if len(file_names) == 1 else list(file_names),
                               start=start,
                               stop=stop,
                               stride=stride)

    # Fail early with the available field names when a requested field is missing.
    missing = []
    missing.extend(field for field in fields if field not in dataframe.columns)
    if bias_fields is not None:
        missing.extend(field for field in bias_fields if field not in dataframe.columns)
    if time_field is not None and time_field not in dataframe.columns:
        missing.append(time_field)
    if missing:
        available = ", ".join(dataframe.columns)
        raise ValueError(f"Field(s) not found in COLVAR data: {', '.join(missing)}. Available fields: {available}.")

    if bias_fields is None:
        bias_fields = [column for column in dataframe.columns if "bias" in column.lower()]

    # Numerical routines work on NumPy arrays, so the dataframe is only used for loading/selection.
    data = dataframe.loc[:, fields].to_numpy()
    if flatten_single_cv and len(fields) == 1:
        data = data.ravel()
    bias = dataframe.loc[:, bias_fields].to_numpy().sum(axis=1) if bias_fields else None
    time = dataframe.loc[:, time_field].to_numpy() if time_field is not None else None

    return data, bias, time, list(fields), list(bias_fields) if bias_fields else None


def get_colvar_output_path(output: Path, output_colvar: Path | None) -> Path:
    return output_colvar if output_colvar is not None else output.with_suffix(".dat")


def save_colvar_table(path: Path, columns: Sequence[np.ndarray], fields: Sequence[str]):
    # Write a PLUMED-like text table that can be inspected with standard tools.
    table = np.column_stack([np.asarray(column).ravel() for column in columns])
    header = f"#! FIELDS {' '.join(fields)}"
    np.savetxt(path, table, header=header, comments="", fmt=" %.10g")
