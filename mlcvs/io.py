"""Input/output functions."""

__all__ = ["colvar_to_pandas"]

import pandas as pd


def colvar_to_pandas(filename="COLVAR", folder="./", sep=" "):
    """
    Load a PLUMED file and save it to a dataframe.

    Parameters
    ----------
    filename : string, optional
        PLUMED output file
    folder : string, optional
        Folder (default = "./")
    sep: string, optional
        Fields separator (default = " ")

    Returns
    -------
    df : DataFrame
        Collective variables dataframe
    """
    skip_rows = 1
    # Read header 
    headers = pd.read_csv(
        folder + filename, sep=" ", skipinitialspace=True, nrows=0
    )
    # Discard #! FIELDS
    headers = headers.columns[2:]

    # Load dataframe and use headers for columns names
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
