"""Input/output functions."""
try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "pandas is required to use the i/o utils (mlcolvar.io)\n", e
    )

import numpy as np
import torch
import os
from typing import Union, List


from mlcolvar.io._utils import _download_temp_file


__all__ = [
    "load_dataframe",
    "plumed_to_pandas",
]


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
            temp, filename = _download_temp_file(file_url=filename, 
                                                 delete_download=delete_download,  
                                                 return_name=True)

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


def _prepare_dataset_from_colvars(
    file_names: Union[list, str],
    folder: str = None,
    create_labels: bool = None,
    load_args: List[dict] = None,
    filter_args: dict = None,
    modifier_function=None,
    verbose: bool = True,
    start: int = 0,
    stop: int = None,
    stride: int = 1,
    delete_download: bool = True,
    read_csv_kwargs: dict = None,
):
    """Load and preprocess COLVAR-like files for dataset construction.

    This internal helper performs all file loading and preprocessing steps
    required to construct a descriptor-based dataset, but does not instantiate
    a ``DictDataset``.

    Parameters
    ----------
    file_names : str or list[str]
        File name or list of file names to load.
    folder : str, optional
        Common directory containing the input files.
    create_labels : bool, optional
        Assign one integer label to each input file. If None, labels are
        created automatically when more than one file is provided.
    load_args : list[dict], optional
        Per-file loading arguments. Each dictionary can contain ``start``,
        ``stop``, ``stride``, and arguments forwarded to ``pandas.read_csv``.
    filter_args : dict, optional
        Arguments passed to ``DataFrame.filter`` to select descriptors.
    modifier_function : callable, optional
        Function applied to the descriptor dataframe.
    verbose : bool, optional
        Whether to print information about the loaded data.
    start, stop, stride : int, optional
        Global slicing parameters used when ``load_args`` is not provided.
    delete_download : bool, optional
        Whether to delete temporary files downloaded from URLs after loading.
    read_csv_kwargs : dict, optional
        Global keyword arguments forwarded to ``pandas.read_csv``.

    Returns
    -------
    dataset_kwargs : dict
        Keyword arguments required to instantiate a descriptor-based
        ``DictDataset`` or compatible subclass.
    dataframe : pandas.DataFrame
        Full loaded dataframe before descriptor-only filtering.
    """
    if isinstance(file_names, str):
        file_names = [file_names]
    elif not isinstance(file_names, list):
        raise TypeError(
            f"file_names must be a string or list of strings, not {type(file_names)}."
        )

    num_files = len(file_names)

    if folder is not None:
        file_names = [os.path.join(folder, fname) for fname in file_names]

    if read_csv_kwargs is None:
        read_csv_kwargs = {}

    if load_args is None:
        load_args = [
            {"start": start, "stop": stop, "stride": stride}
            for _ in file_names
        ]
    else:
        if start != 0 or stop is not None or stride != 1:
            raise ValueError(
                "Both global and per-file loading parameters have been "
                "specified. Use either `load_args` or `start`, `stop`, "
                "and `stride`."
            )

        if not isinstance(load_args, list) or len(load_args) != num_files:
            raise TypeError(
                "load_args must be a list of dictionaries with the "
                "same length as file_names."
            )

    if create_labels is None:
        create_labels = num_files > 1

    df = pd.DataFrame()

    for i, filename in enumerate(file_names):
        file_load_args = {**read_csv_kwargs, **load_args[i]}
        df_tmp = load_dataframe(
            filename,
            delete_download=delete_download,
            **file_load_args,
        )

        if create_labels:
            df_tmp["labels"] = i

        if verbose:
            print(f"Class {i} dataframe shape: ", np.shape(df_tmp))

        df = pd.concat([df, df_tmp], ignore_index=True)

    df_data = (
        df.filter(**filter_args)
        if filter_args is not None
        else df.copy()
    )

    df_data = df_data.filter(
        regex="^(?!.*labels)^(?!.*time)^(?!.*bias)^(?!.*walker)"
    )

    if verbose:
        print(f"\n - Loaded dataframe {df.shape}:", list(df.columns))
        print(f" - Descriptors {df_data.shape}:", list(df_data.columns))

    if modifier_function is not None:
        df_data = df_data.apply(modifier_function)

    dictionary = {"data": torch.Tensor(df_data.values)}

    if create_labels:
        dictionary["labels"] = torch.Tensor(
            df["labels"].to_numpy(copy=True)
        )

    dataset_kwargs = {
        "dictionary": dictionary,
        "feature_names": df_data.columns.values,
        "data_type": "descriptors",
    }

    return dataset_kwargs, df