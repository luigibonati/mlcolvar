"""Input/output functions."""

import pandas as pd
import numpy as np
import string
import torch
from torch.utils.data import TensorDataset, random_split
from mlcvs.utils.data import FastTensorDataLoader


__all__ = ["load_dataframe", "plumed_to_pandas", "dataset_from_file"]

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

def dataset_from_file(file_names : list, 
                    max_rows : int or list,
                    create_labels : bool = True, 
                    skip_rows : int or list = 0,
                    regex_string : str = '',
                    files_folder : str = None,
                    dtype : torch.dtype = None,
                    return_dataframe : bool = False, 
                    verbose : bool =  True):
    """
    Initialize a torch labeled dataset from a list of files.

    Parameters
    ----------
    file_names : list
        Names of files from which import the data, one per class
    max_rows : int or list
        Maximum number of rows to be imported for each file. 
        If int, it is set for all file
    skip_rows : int or list, optional
        Number of rows to be skipped for each file, by default 0
        If int, it is set for all file
    regex_string : str, optional
        String for regex filtering of the inputs from pandas dataframe to torch dataset, by default ''
        The default already filters out the 'time' column
    files_folder : str, optional
        Common path for the files to be imported, by default None
        If set, filenames become 'files_folder/file_name'
    dtype : torch.dtype, optional
        Torch dtype for the tensor in the dataset, by default None
    return_dataframe : bool, optional
        Return also the imported Pandas dataframe for convenience, by default False
    verbose : bool, optional
        Print info on the imported data, by default True

    Returns
    -------
    torch.Dataset
        Torch labeled dataset of the given data
    optional, pandas.Dataframe
        Pandas dataframe of the given data #TODO improve
    """
    num_files = len(file_names)

    # parse rows to be imported
    if type(max_rows) == int:
        max_rows = np.ones(num_files, dtype=int) * max_rows
    if type(skip_rows) == int:
        skip_rows = np.ones(num_files, dtype=int) * skip_rows

    # set file paths
    if files_folder is not None:
            for i in range(len(file_names)):
                file_names[i] = files_folder + '/' + file_names[i]

    # initialize pandas dataframe
    dataframe = pd.DataFrame()

    # load data
    for i in range(num_files):
        dataframe_tmp = load_dataframe(file_names[i], 
                                        start = skip_rows[i], 
                                        stop = max_rows[i])
        # add label in the dataframe
        if create_labels: dataframe_tmp['Label'] = i

        if verbose: print(f' - Class {list(string.ascii_uppercase)[i]} dataframe shape: ', np.shape(dataframe_tmp))

        # update collective dataframe
        dataframe = pd.concat([dataframe, dataframe_tmp], ignore_index=True)
    
    if verbose: 
        print(f' - Imported pandas dataframe shape: ', np.shape(dataframe))
        print(f' - Filtered pandas dataframe shape: ', np.shape(dataframe.filter(regex=f'^(?!.*Label)^(?!.*time)({regex_string})')))

    # create torch dataset
    if create_labels: 
        dataset = TensorDataset(torch.tensor(dataframe.filter(regex=f'^(?!.*Label)^(?!.*time)({regex_string})').values, dtype=dtype), torch.tensor(dataframe.filter(like='Label').values, dtype=dtype))
    else:
        dataset = TensorDataset(torch.tensor(dataframe.filter(regex=f'^(?!.*Label)^(?!.*time)({regex_string})').values, dtype=dtype))
    
    if verbose:
        print(f' - Points in torch dataset: ', dataset.__len__())
        print(f' - Descriptors in dataset: ', list(dataset)[0][0].shape[0])
    
    if return_dataframe:
        return dataset, dataframe
    else:
        return dataset

def test_datasetFromFile():
    print('Test no regex on two states..')
    torch_dataset, pd_dataframe = dataset_from_file(file_names = ['state_A.dat', 'state_B.dat'], 
                                                    max_rows = 10, 
                                                    skip_rows =  0,
                                                    create_labels=True,
                                                    files_folder = 'mlcvs/tests/data',
                                                    return_dataframe = True,
                                                    regex_string='',
                                                    verbose = True)
    
    print()
    print('Test limited regex on two states..')
    torch_dataset, pd_dataframe = dataset_from_file(file_names = ['state_A.dat', 'state_B.dat'], 
                                                    max_rows = 10, 
                                                    skip_rows =  0,
                                                    create_labels=True,
                                                    files_folder = 'mlcvs/tests/data',
                                                    return_dataframe = True,
                                                    regex_string='n|o',
                                                    verbose = True)

    print()
    print('Test unlabeled on three states..')
    torch_dataset, pd_dataframe = dataset_from_file(file_names = ['state_A.dat', 'state_B.dat', 'state_C.dat'], 
                                                    max_rows = 10, 
                                                    skip_rows =  0,
                                                    create_labels=False,
                                                    files_folder = 'mlcvs/tests/data',
                                                    return_dataframe = True,
                                                    regex_string='',
                                                    verbose = True)

if __name__ == "__main__":
    test_datasetFromFile()
