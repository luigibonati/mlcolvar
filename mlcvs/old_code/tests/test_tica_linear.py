"""
Unit and regression test for the tica module.
"""

# Import package, test suite, and other packages as needed
import pytest
import torch
import numpy as np
import pandas as pd
from mlcvs.utils.io import load_dataframe
from mlcvs.tica import TICA_CV

# set global variables
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=8)

def test_tica_train_2d_model():
    """Perform TICA on 2d_model data folder."""

    # Load dataset
    data = load_dataframe("mlcvs/tests/data/2d_model/COLVAR_md", stride=1)

    X = data.filter(regex='p.')

    # Define model
    n_features = X.shape[1]
    tica = TICA_CV(n_features)

    # Fit TICA
    tica.fit(data, lag=10)

    # Project
    y_test = tica(X.iloc[0])
    
    # assert
    print(X.iloc[0])
    print(y_test)
    y_test_expected = torch.tensor(
                        [-0.66858778,  0.90833012]
                      )

    assert torch.abs(y_test_expected - y_test).sum() < 1e-6