"""
Unit and regression test for the tica module.
"""

# Import package, test suite, and other packages as needed
import pytest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,random_split,TensorDataset
from mlcvs.utils.data import create_time_lagged_dataset,FastTensorDataLoader
from mlcvs.utils.io import load_dataframe
from mlcvs.ae import AutoEncoderCV

# set global variables
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_printoptions(precision=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="module")
def load_dataset_2d_md():
    """Load 2d-basins dataset"""

    # Load colvar files as pandas dataframes
    data = load_dataframe("mlcvs/tests/data/2d_model/COLVAR_md", stride=1)
    
    # Create input datasets
    X = data.filter(regex='p.').values
    t = data['time'].values
    names = data.filter(regex="p.*").columns.values

    # Convert np to torch
    X = torch.Tensor(X)
    return X, t, names

@pytest.mark.slow
def test_autoencoder_2d_dataloader(load_dataset_2d_md):
    """Unsupervised learning on 2d_model data folder."""

    # Load dataset
    X,_,names = load_dataset_2d_md

    X = torch.Tensor(X)
    dataset = TensorDataset(X)

    # split train - valid 
    n_train  = int( 0.8 * len(dataset) )
    n_valid  = len(dataset) - n_train
    train_data, valid_data = random_split(dataset,[n_train,n_valid]) 
    train_loader = FastTensorDataLoader(train_data)
    valid_loader = FastTensorDataLoader(valid_data)

    # Define model
    n_features = X.shape[1]
    layers            = [n_features,10,10,1]

    model = AutoEncoderCV(layers,activation='relu',device='cpu')
    model.fit(train_loader, valid_loader,
                standardize_inputs = True, standardize_outputs=True,
                nepochs=10, log_every=10)
