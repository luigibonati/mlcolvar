"""Input/output functions."""

__all__ = ["colvar_to_pandas"]

import pandas as pd


def colvar_to_pandas(filename="COLVAR", folder="./", sep=" "):
    """
    Load a PLUMED colvar file and save it to a dataframe.

    Parameters
    ----------
    filename : string, optional
        Collective variables file
    folder : string, optional
        Folder
    sep: string, optional
        Fields separator

    Returns
    -------
    df : DataFrame
        Collective variables dataframe
    """
    skip_rows = 1
    headers = pd.read_csv(
        folder + filename, sep=" ", skipinitialspace=True, nrows=0
    ).columns[2:]
    df = pd.read_csv(
        folder + filename,
        sep=sep,
        skipinitialspace=True,
        header=None,
        skiprows=range(skip_rows),
        names=headers,
        comment="#",
    )

    return df
