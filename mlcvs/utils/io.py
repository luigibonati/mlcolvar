"""Input/output functions."""

import pandas as pd

__all__ = ["load_dataframe", "plumed_to_pandas"]

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


def load_dataframe(data, start = 0, stop = None, stride = 1, **kwargs):
    """Load dataframe(s) from object or from file. In case of PLUMED colvar files automatically handles the column names.

    Parameters
    ----------
    data : str, pandas.DataFrame, or list
        input data

    Returns
    -------
    pandas.DataFrame
        Dataframe

    Raises
    ------
    TypeError
        if data is not a valid type
    """
    # check if data is Dataframe
    if type(data) == pd.DataFrame:
        df = data
        df = df.iloc[start:stop:stride, :]
        df.reset_index(drop=True, inplace=True)
        
    # or is a string
    elif type(data) == str:
        filename = data
        # check if file is in PLUMED format
        if is_plumed_file(filename):
            df = plumed_to_pandas(filename)
        # else use read_csv with optional kwargs
        else:
            df = pd.read_csv(filename, **kwargs)
        
        df = df.iloc[start:stop:stride, :]
        df.reset_index(drop=True, inplace=True)

    # or a list 
    elif type(data) == list:
        # (a) list of filenames
        if type(data[0]) == str:
            df_list = []
            for i, filename in enumerate(data):
                # check if file is in PLUMED format
                if is_plumed_file(filename):
                    df_tmp = plumed_to_pandas(filename)
                    df_tmp['walker'] = [i for _ in range(len(df_tmp))]
                    df_tmp = df_tmp.iloc[start:stop:stride, :]
                    df_list.append( df_tmp )
                    
                # else use read_csv with optional kwargs
                else:
                    df_tmp = pd.read_csv(filename, **kwargs)
                    df_tmp['walker'] = [i for _ in range(len(df_tmp))]
                    df_tmp = df_tmp.iloc[start:stop:stride, :]
                    df_list.append( df_tmp )

        elif type(data[0]) == pd.DataFrame:
            df_list = []
            for df_tmp in data:
                df_tmp = df_tmp.iloc[start:stop:stride, :]
                df_list.append(df_tmp)

        df = pd.concat(df_list)
        df.reset_index(drop=True, inplace=True)

    else:
        raise TypeError(f"{data}: Accepted types are 'pandas.Dataframe', 'str', or list")

    return df
