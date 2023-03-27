"""Input/output functions."""
# pandas
try:
    import pandas as pd
except ImportError as e:
    raise ImportError('pandas is required to use the i/o utils (mlcvs.utils.io)\n',e)

import pandas as pd
import numpy as np
import torch
import os
from mlcvs.data import DictionaryDataset

__all__ = ["load_dataframe", "plumed_to_pandas", "create_dataset_from_files"]

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

def create_dataset_from_files(
                    file_names : list,
                    folder : str = None,
                    create_labels : bool = None, 
                    load_args : list = None,
                    filter_args : dict = None,
                    modifier_function = None, 
                    return_dataframe : bool = False,
                    verbose : bool =  True, 
                    **kwargs):
    """
    Initialize a dataset from (a list of) files. Suitable for supervised/unsupervised tasks.

    Parameters
    ----------
    file_names : list
        Names of files from which import the data
    folder : str, optional
        Common path for the files to be imported, by default None. If set, filenames become 'folder/file_name'.
    create_labels: bool, optional
        Assign a label to each file, default True if more than a file is given, otherwise False
    load_args: list[dict], optional
        List of dictionaries with the arguments passed to load_dataframe function for each file (keys: start,stop,stride and pandas.read_csv options), by default None 
    filter_args: dict, optional
        Dictionary of arguments which are passed to df.filter() to select descriptors (keys: items, like, regex), by default None 
        Note that 'time' and '*.bias' columns are always discarded.
    return_dataframe : bool, optional
        Return also the imported Pandas dataframe for convenience, by default False
    modifier_function : function, optional
        Function to be applied to the input data, by default None.
    verbose : bool, optional
        Print info on the datasets, by default False
    kwargs : optional
        args passed to mlcvs.utils.io.load_dataframe

    Returns
    -------
    torch.Dataset
        Torch labeled dataset of the given data
    optional, pandas.Dataframe
        Pandas dataframe of the given data #TODO improve

    See also
    --------
    mlcvs.utils.io.load_dataframe
        Function that is used to load the files

    """
    num_files = len(file_names)
    
    # set file paths
    if folder is not None:
        file_names = [ os.path.join(folder, fname ) for fname in file_names]

    # check if per file args are given, otherwise set to {}
    if load_args is None:
        load_args = [ {} for _ in file_names ]
    else:
        if (not isinstance(load_args,list)) or (len(file_names) != len(load_args)):
            raise TypeError('load_args should be a list of dictionaries of arguments of same length as file_names. If you want to use the same args for all file pass them directly as **kwargs.')

    # check if create_labels if given, otherwise set it to True if more than one file is given
    if create_labels is None:
        create_labels = False if len(file_names) == 1 else True

    # initialize pandas dataframe
    df = pd.DataFrame()

    # load data
    for i in range(num_files):
        df_tmp = load_dataframe(file_names[i], **load_args[i], **kwargs)
        
        # add label in the dataframe
        if create_labels: df_tmp['labels'] = i
        if verbose: print(f'Class {i} dataframe shape: ', np.shape(df_tmp))

        # update collective dataframe
        df = pd.concat([df, df_tmp], ignore_index=True)
    
    # filter inputs
    df_data = df.filter(**filter_args) if filter_args is not None else df.copy()
    df_data = df_data.filter(regex='^(?!.*labels)^(?!.*time)(?!.*bias)' ) 

    if verbose: 
        print(f'\n - Loaded dataframe {df.shape}:', list(df.columns)  )
        print(f' - Descriptors {df_data.shape}:', list(df_data.columns)  ) 

    # apply transformation 
    if modifier_function is not None:
        df_data = df_data.apply(modifier_function)

    # create DictionaryDataset
    dictionary = {'data' : torch.Tensor(df_data.values) } 
    if create_labels: 
        dictionary['labels'] = torch.Tensor(df['labels'].values)
    dataset = DictionaryDataset(dictionary)
    
    if return_dataframe:
        return dataset, df
    else:
        return dataset

def test_datasetFromFile():
    # Test with unlabeled dataset
    torch_dataset, pd_dataframe = create_dataset_from_files(file_names = ['state_A.dat','state_B.dat','state_C.dat'],
                                                            folder = 'mlcvs/tests/data',
                                                            create_labels=False,
                                                            load_args=None,
                                                            filter_args=None,
                                                            return_dataframe=True,
                                                            start=0, #kwargs to load_dataframe
                                                            stop=5,
                                                            stride=1,                     
    )

    # Test no regex on two states
    torch_dataset, pd_dataframe = create_dataset_from_files(file_names = ['state_A.dat', 'state_B.dat'],
                                                            folder = 'mlcvs/tests/data',
                                                            create_labels = True,
                                                            load_args=None,
                                                            filter_args=None,
                                                            return_dataframe=True,
                                                            start=0, #kwargs to load_dataframe
                                                            stop=5,
                                                            stride=1,                     
    )
                                                            
    # Test with filter regex on two states
    torch_dataset = create_dataset_from_files(file_names = ['state_A.dat', 'state_B.dat'],
                                                            folder = 'mlcvs/tests/data',
                                                            create_labels = True,
                                                            load_args=None,
                                                            filter_args={'regex':'n|o'},
                                                            return_dataframe=False,
                                                            start=0, #kwargs to load_dataframe
                                                            stop=5,
                                                            stride=1,                     
    )

    def test_modifier(x):
        return x**2

    # Test with filter regex on two states with modifier
    torch_dataset, pd_dataframe = create_dataset_from_files(file_names = ['state_A.dat', 'state_B.dat'],
                                                            folder = 'mlcvs/tests/data',
                                                            create_labels = True,
                                                            load_args=None,
                                                            filter_args={'regex':'n|o'},
                                                            modifier_function=test_modifier,
                                                            return_dataframe=True,
                                                            start=0, #kwargs to load_dataframe
                                                            stop=5,
                                                            stride=1,                     
    )

if __name__ == "__main__":
    test_datasetFromFile()
