"""
Unit and regression test for the tica module.
"""

# Import package, test suite, and other packages as needed
import pytest
import torch
import numpy as np
import pandas as pd
from mlcvs.utils.io import colvar_to_pandas
from mlcvs.tica import TICA_CV

# set global variables
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_tica_train_2d_model():
    """Perform TICA on 2d_model data folder."""

    # Load dataset
    data = colvar_to_pandas(folder="mlcvs/tests/data/2d_model/", filename="COLVAR_md")
    data = data[::50]

    X = data.filter(regex='p.')

    # Define model
    n_features = X.shape[1]
    tica = TICA_CV(n_features, device=device)

    # Fit TICA
    tica.train(data, lag=10)

    # Project
    y_test = tica(X.iloc[0])
    
    # assert
    print(X.iloc[0])
    print(y_test)
    y_test_expected = torch.tensor(
                        [-0.83703607,  0.90135466]
                      ).to(device)

    assert torch.abs(y_test_expected - y_test).sum() < 1e-6