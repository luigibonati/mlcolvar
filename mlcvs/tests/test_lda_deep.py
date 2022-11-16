"""
Unit and regression test for the lda module.
"""

# Import package, test suite, and other packages as needed
import pytest
import torch
import numpy as np
import pandas as pd
from mlcvs.utils.io import load_dataframe
from mlcvs.lda import  DeepLDA_CV
from mlcvs.utils.data import FastTensorDataLoader
from torch.utils.data import random_split,TensorDataset

# set global variables
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO avoid duplication with lda_linear
@pytest.fixture(scope="module")
def load_dataset_2d_classes():
    """Load 2d-basins dataset"""

    # Load colvar files as pandas dataframes
    dataA = load_dataframe("mlcvs/tests/data/2d_model/COLVAR_stateA")
    dataB = load_dataframe("mlcvs/tests/data/2d_model/COLVAR_stateB")

    # Create input datasets
    xA = dataA.filter(regex="p.*").values
    xB = dataB.filter(regex="p.*").values
    names = dataA.filter(regex="p.*").columns.values

    # Create labels
    yA = np.zeros(len(dataA))
    yB = np.ones(len(dataB))

    # Concatenate
    X = np.concatenate([xA, xB], axis=0)
    y = np.concatenate([yA, yB], axis=0)

    # Shuffle
    np.random.seed(1)
    p = np.random.permutation(len(X))
    X, y = X[p], y[p]

    # Convert np to torch
    X = torch.Tensor(X).to(device)
    y = torch.Tensor(y).to(device)
    return X, y, names


@pytest.mark.parametrize("n_classes", [2, 3])
def test_deeplda_nclasses(n_classes):
    """Define a DeepLDA object with different number of classes."""

    # define dataset
    n_data = 100
    n_features = 10
    X = torch.rand((n_data, n_features)).to(device)
    y = torch.randint(low=0, high=n_classes, size=(n_data,))

    # split train/test
    ntrain = int(n_data * 0.8)
    nvalid = int(n_data * 0.2)
    train_data = [X[:ntrain], y[:ntrain]]
    valid_data = [X[ntrain : ntrain + nvalid], y[ntrain : ntrain + nvalid]]

    # Architecture
    hidden_nodes = "20,20,5"
    nodes = [int(x) for x in hidden_nodes.split(",")]
    nodes.insert(0, X.shape[1])
    n_hidden = nodes[-1]

    # Model
    model = DeepLDA_CV(nodes)
    model.set_device(device)

    # Define input
    xtest = torch.ones(n_features).to(device)

    # Forward
    ytest = model(xtest)

    # ASSERT if shape == n_hidden
    expected_y_shape = torch.rand(n_hidden).shape
    assert ytest.shape == expected_y_shape

    # Compute lda and set params; new forward
    with torch.no_grad():
        loss = model.evaluate_dataset(train_data, save_params=True)
    y2test = model(xtest)

    # ASSERT if shape == n_classes-1
    expected_y2_shape = torch.rand(n_classes - 1).shape
    assert y2test.shape == expected_y2_shape

    # Check PLUMED INPUT
    model.set_params({'feature_names' : [f'd{i}' for i in range(n_features)] })
    input = model.plumed_input()
    expected_input = (
        "deeplda_cv: PYTORCH_MODEL FILE=model.ptc ARG=d0,d1,d2,d3,d4,d5,d6,d7,d8,d9"
    )
    assert expected_input == input

@pytest.mark.slow
#@pytest.mark.skip
def test_deeplda_train_2d_model(load_dataset_2d_classes):
    """Perform DeepLDA on 2d-basins data folder."""

    # load dataset
    X, y, feature_names = load_dataset_2d_classes

    # split train/test
    ntrain = 800
    nvalid = 200

    standardize_inputs = True  # @param {type:"boolean"}

    train_data = [X[:ntrain], y[:ntrain]]
    valid_data = [X[ntrain : ntrain + nvalid], y[ntrain : ntrain + nvalid]]

    hidden_nodes = "30,30,5"
    nodes = [int(x) for x in hidden_nodes.split(",")]
    nodes.insert(0, X.shape[1])
    n_hidden = nodes[-1]

    # -- Parameters --
    lrate = 0.001
    sw_reg = 0.05
    l2_reg = 1e-5

    # MODEL
    model = DeepLDA_CV(nodes)
    model.set_device(device)

    # OPTIMIZER
    opt = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=l2_reg)

    # REGULARIZATION
    model.set_optimizer(opt)
    model.set_earlystopping(
        patience=10, min_delta=0.0, consecutive=False, save_best_model=True
    )
    model.set_regularization(sw_reg=sw_reg)

    # TRAIN (with X,y)
    model.fit(X=X, y=y, info=True, log_every=100)

    # FORWARD
    xtest = torch.rand(X.size(1)).to(device)
    with torch.no_grad():
        ytest = model(xtest)

    # ASSERT SHAPE OUTPUT
    expected_y_shape = torch.rand(1).shape
    assert ytest.shape == expected_y_shape

    # COMPARE SINGLE TENSOR AND BATCH OF SIZE 1
    xtest2 = xtest.unsqueeze(0)

    with torch.no_grad():
        ytest2 = model(xtest2)

    expected_y2_shape = torch.rand(1).shape
    assert torch.equal(ytest, ytest2[0])

    # EXPORT CHECKPOINT AND LOAD (see function below for comments)
    model.export(folder='mlcvs/tests/__pycache__/')
    model_loaded = DeepLDA_CV(nodes)
    model_loaded.to(device)

    with torch.no_grad():
        loss = model_loaded.evaluate_dataset(train_data, save_params=True) 
        model_loaded.standardize_inputs(train_data[0])
        model_loaded.standardize_outputs(train_data[0])

    model_loaded.load_checkpoint('mlcvs/tests/__pycache__/model_checkpoint.pt') 

    # New forward
    ytest_loaded = model_loaded(xtest)

    print(model_loaded.opt_)
    # Assert results
    assert torch.equal(ytest, ytest_loaded)

def test_deeplda_export_load():
    """Test export / loading functions."""

    # define dataset
    n_data = 100
    n_features = 10
    n_classes = 2 
    X = torch.rand((n_data, n_features)).to(device)
    y = torch.randint(low=0, high=n_classes, size=(n_data,))

    # split train/test
    ntrain = int(n_data * 0.8)
    nvalid = int(n_data * 0.2)
    train_data = [X[:ntrain], y[:ntrain]]
    valid_data = [X[ntrain : ntrain + nvalid], y[ntrain : ntrain + nvalid]]

    # Architecture
    hidden_nodes = "20,20,5"
    nodes = [int(x) for x in hidden_nodes.split(",")]
    nodes.insert(0, X.shape[1])
    n_hidden = nodes[-1]

    # Model
    model = DeepLDA_CV(nodes)
    model.to(device)

    # Compute lda and set params
    with torch.no_grad():
        loss = model.evaluate_dataset(train_data, save_params=True)
    # Fake assignment
    model.set_params({'lorentzian_reg' : 40.0 })

    # Define input and forward
    xtest = torch.ones(n_features).to(device)
    ytest = model(xtest)

    # Export model
    model.export(folder='mlcvs/tests/__pycache__/')

    # (1) --- Load checkpoint into new model
    model_loaded = DeepLDA_CV(nodes)
    model_loaded.to(device)

    # Note: it requires the correct shape for w and b of linear projection TODO
    # Workaround: make a fake LDA assignment
    with torch.no_grad():
        loss = model_loaded.evaluate_dataset(train_data, save_params=True) 

    model_loaded.load_checkpoint('mlcvs/tests/__pycache__/model_checkpoint.pt') 

    # New forward
    ytest_loaded = model_loaded(xtest)

    # Assert results
    assert torch.equal(ytest, ytest_loaded)
    # Assert parameters loading
    assert model.get_params()['lorentzian_reg'] == model_loaded.get_params()['lorentzian_reg'] 

    # (2) --- Load TorchScript model
    model_traced = torch.jit.load('mlcvs/tests/__pycache__/model.ptc')
    ytest_traced = model_traced(xtest)

    # Assert results
    assert torch.equal(ytest, ytest_traced)
