"""Input/output functions."""

import pandas as pd

__all__ = ["load_dataframe", "plumed_to_pandas", "dataloader_from_file"]

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

def dataloader_from_file(states_num, files_folder, file_names, n_input, max_rows, from_column, train_set=None,
                         valid_set=None, batch_size = -1, device=torch.device('cpu'), dtype=None, silent=False ):
    '''Load data from files and directly creates the training and validation loaders with descriptors and labels.

    Parameters
    ----------
    states_num : int,
        number of states to train on
    cvs_num : int,
        number of collective variables (CVs) to train
    files_folder : str,
        path to the folder with training data /my/path/to/data
    file_names : list of str,
        names of the files of training data
    n_input : int,
        number of descriptors to be used for the training
    max_rows : int or list of int,
        max number of rows to be loaded from each file. Use a list to specify for each state
    from_column : int,
        zero-based index of the first column to be loaded
    train_set : int,
        size of the training set (default = 90% dataset)
    valid_set : int,
        size of the validation set (default = 10% dataset)
    batch_size : int,
        batch_size for the training set (default = -1, single batch)
    device : torch.device(),
        device on which load the data (default = torch.device('cpu'))
    dtype :
        torch.dtype, (default = torch.float)
    silent : bool,
        it makes screen output silent (default = False)

    Returns
    -------
    pandas.DataFrame
        Dataframe

    '''
    loaded_var, loaded_labels = {}, {}

    if type(max_rows) == int:
        max_rows = np.ones(states_num, dtype=int) * max_rows

    if not silent:
        print()
        print('Loading classes data... ')
        print('Check data shape: (set_size, n_descriptors)')
    for i in range(states_num):
        file_path = f'{files_folder}/{file_names[i]}'
        loaded_var[i] = (np.loadtxt(file_path, max_rows=max_rows[i] + 1,
                                    usecols=range(from_column, from_column + n_input)))
        if not silent: print(f' - Data shape for class {list(string.ascii_uppercase)[i]}: ', np.shape(loaded_var[i]))
        loaded_labels[i] = np.zeros(loaded_var[i].shape[0]) + i

    # we need a single big training vector
    var = np.concatenate([loaded_var[i] for i in loaded_var], axis=0)
    labels = np.concatenate([loaded_labels[i] for i in loaded_labels], axis=0)

    # get total number of data points
    datasize = len(var)

    # we randomly shuffle the deck
    perm = np.random.permutation(len(var))
    var, labels = var[perm], labels[perm]

    # organize data into sets
    if train_set is None:
        if valid_set is None:
            train_set = int( 0.9 * datasize)
            valid_set = datasize - train_set
        else:
            if (datasize - valid_set) / datasize < 0.6:  print('WARNING: Valid set > 40% of dataset! Are you sure?')
            train_set = datasize - valid_set

    # initialize dataset and loaders
    dataset = TensorDataset(torch.tensor(var, dtype=dtype, device=device), torch.tensor(labels, dtype=dtype, device=device))
    train_data, valid_data = random_split(dataset,[train_set,valid_set])
    train_loader = FastTensorDataLoader(train_data, batch_size=batch_size)
    valid_loader = FastTensorDataLoader(valid_data)

    return train_loader, valid_loader

