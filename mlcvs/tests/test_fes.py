"""
Unit and regression test for the fes module.
"""

# Import package, test suite, and other packages as needed
import pytest
import torch
import numpy as np
from mlcvs.utils.fes import compute_fes

# set global variables
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=8)

def test_compute_fes():
    """Test compute FES."""
    # Generate data
    np.random.seed(1)
    X = np.random.multivariate_normal([ 0 for i in range(2)], np.eye(2), 100)
    w = np.random.rand(100)
    # 1D
    fes,grid,bounds,error = compute_fes(X[:,0], blocks = 2, bandwidth=0.02, weights = w, scale_by='std')
    assert (fes[50] - 2.605479) < 1e-4
    # 2D
    fes,grid,bounds,_ = compute_fes(X, blocks = 1, bandwidth=0.02, scale_by='range', plot=True)
    assert (fes[50,50] - 0.762659) <1e-4

