"""
Unit and regression test for the tica module.
"""

# Import package, test suite, and other packages as needed
import pytest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,random_split
from mlcvs.utils.data import create_time_lagged_dataset,FastTensorDataLoader
from mlcvs.utils.io import load_dataframe
from mlcvs.tica import DeepTICA_CV

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
def test_deeptica_train_2d_model(load_dataset_2d_md):
    """Perform TICA on 2d_model data folder."""

    # Load dataset
    X,t,names = load_dataset_2d_md

    # Define model
    n_features = X.shape[1]
    n_eig = 2
    model = DeepTICA_CV(layers=[n_features,10,10,n_eig],device=device)
    model.set_device(device)

    # Fit TICA
    model.fit(X=X, y=t, nepochs=10, options = { 'lag_time': 10 })

    # Project
    y_test = model(X[0].to(device))
    print(y_test)

@pytest.mark.slow
def test_deeptica_train_2d_dataloader(load_dataset_2d_md):
    """Perform TICA on 2d_model data folder."""

    # Load dataset
    X,t,names = load_dataset_2d_md

    # Create dataset
    lag_time = 10
    dataset = create_time_lagged_dataset(X,t=t,lag_time=lag_time,logweights=None)

    # split train - valid
    n_val = int(len(dataset)*.2)
    train_data, val_data = random_split(dataset, [len(dataset) - n_val, n_val])

    # create dataloaders 
    train_loader = FastTensorDataLoader(train_data, batch_size=len(train_data), shuffle=True)
    valid_loader = FastTensorDataLoader(val_data, batch_size=len(val_data),  shuffle=False)

    print(len(dataset))

    # Define model
    n_features = X.shape[1]
    n_eig = 2
    model = DeepTICA_CV(layers=[n_features,10,10,n_eig],device=device)
    model.to(device)

    # Fit TICA
    model.set_regularization(cholesky_reg=1e-6)
    model.set_earlystopping(patience=5)
    model.fit(train_loader,valid_loader,nepochs=10)

    # Project
    y_test = model(X[0].to(device))
    print(y_test)
