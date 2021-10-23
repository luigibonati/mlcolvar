"""
Unit and regression test for the lda module.
"""

# Import package, test suite, and other packages as needed
import pytest
import torch
import numpy as np
from mlcvs.io import colvar_to_pandas
from mlcvs.lda import LDA_CV, DeepLDA_CV

# set global variables
#torch.set_default_tensor_type(torch.FloatTensor)
device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def load_dataset_2d_model():
    """Load 2d-basins dataset"""

    # Load colvar files as pandas dataframes
    dataA = colvar_to_pandas(folder="mlcvs/data/2d_model/", filename="COLVAR_stateA")
    dataB = colvar_to_pandas(folder="mlcvs/data/2d_model/", filename="COLVAR_stateB")

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
    X = torch.Tensor(X, device=device)
    y = torch.Tensor(y, device=device)
    return X, y, names


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("is_harmonic_lda", [False, True])
def test_lda_harmonic_nclasses(n_classes,is_harmonic_lda):
    """Perform LDA on toy dataset."""

    n_data = 100
    n_features = 3

    # Generate classes
    np.random.seed(1)

    x_list = []
    y_list = []

    for i in range(n_classes):
        mean = [1 if j == i else 0 for j in range(n_features)]
        cov = 0.2 * np.eye(n_features)

        x_i = np.random.multivariate_normal(mean, cov, n_data)
        y_i = i * np.ones(len(x_i))

        x_list.append(x_i)
        y_list.append(y_i)

    # Concatenate
    X = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Transform to tensor 
    X = torch.Tensor(X, device=device)
    y = torch.Tensor(y, device=device)

    # Define model
    n_features = X.shape[1]
    lda = LDA_CV(n_features,harmonic_lda=is_harmonic_lda,device=device)

    # Fit and transform LDA
    result = lda.train_forward(X, y)

    # Project
    x_test = torch.tensor(n_features).to(device)
    y_test = lda(x_test)
    if is_harmonic_lda:
        y_test_expected = torch.tensor(
            [0.0709] if n_classes == 2 else [0.1525, -0.2786]
        ).to(device)
    else:
        y_test_expected = torch.tensor(
            [0.2407] if n_classes == 2 else [0.2316, -0.1087]
        ).to(device)
    print(y_test)
    assert (y_test_expected - y_test).abs().sum() < 1e-4

@pytest.mark.parametrize("is_harmonic_lda", [False, True])
def test_lda_train_2d_model_harmonic(load_dataset_2d_model,is_harmonic_lda):
    """Perform LDA on 2d_model data folder."""

    # Load dataset
    X, y, feature_names = load_dataset_2d_model

    # Define model
    n_features = X.shape[1]
    lda = LDA_CV(n_features,harmonic_lda = is_harmonic_lda,device=device)
    # Set features names (for PLUMED input)
    lda.set_params({"feature_names": feature_names})
    
    print(X.dtype)

    # Fit LDA
    lda.train(X, y)

    # Project
    x_test = np.ones(2)
    y_test = lda(x_test)
    
    print(y_test.cpu().numpy())
    y_test_expected = torch.tensor(
                        [-0.0356541] if is_harmonic_lda else [-0.0960027] 
                      ).to(device)

    assert torch.abs(y_test_expected - y_test) < 1e-5

    # Check PLUMED INPUT
    input = lda.plumed_input()
    expected_input = (
        "hlda_cv: COMBINE ARG=p.x,p.y COEFFICIENTS=0.689055,-0.72470 PERIODIC=NO" if is_harmonic_lda
        else "lda_cv: COMBINE ARG=p.x,p.y COEFFICIENTS=0.657474,-0.75347 PERIODIC=NO"
    )
    assert expected_input == input


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
    model = DeepLDA_CV(nodes, device=device)
    model.to(device)

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
# @pytest.mark.skip
def test_deeplda_train_2d_model(load_dataset_2d_model):
    """Perform DeepLDA on 2d-basins data folder."""

    # load dataset
    X, y, feature_names = load_dataset_2d_model

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
    model = DeepLDA_CV(nodes, device=device)
    model.to(device)

    # OPTIMIZER
    opt = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=l2_reg)

    # REGULARIZATION
    model.set_optimizer(opt)
    model.set_earlystopping(
        patience=10, min_delta=0.0, consecutive=False, save_best_model=True
    )
    model.set_regularization(sw_reg=sw_reg)

    # TRAIN
    model.train(train_data, valid_data, info=True, log_every=100)

    # standardize outputs
    model.standardize_outputs(train_data[0])

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
    model_loaded = DeepLDA_CV(nodes, device=device)
    model_loaded.to(device)

    with torch.no_grad():
        loss = model_loaded.evaluate_dataset(train_data, save_params=True) 
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
    model = DeepLDA_CV(nodes, device=device)
    model.to(device)

    # Compute lda and set params
    with torch.no_grad():
        loss = model.evaluate_dataset(train_data, save_params=True)
    # Fake assignment
    model.set_params({'epochs' : 1 })

    # Define input and forward
    xtest = torch.ones(n_features).to(device)
    ytest = model(xtest)

    # Export model
    model.export(folder='mlcvs/tests/__pycache__/')

    # (1) --- Load checkpoint into new model
    model_loaded = DeepLDA_CV(nodes, device=device)
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
    assert model.get_params()['epochs'] == model_loaded.get_params()['epochs'] 

    # (2) --- Load TorchScript model
    model_traced = torch.jit.load('mlcvs/tests/__pycache__/model.ptc')
    ytest_traced = model_traced(xtest)

    # Assert results
    assert torch.equal(ytest, ytest_traced)

if __name__ == "__main__":
    test_lda_harmonic_nclasses(2,False)
