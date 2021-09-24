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
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture(scope="module")
def load_dataset():
    """Load 2d-basins dataset"""

    # Load colvar files as pandas dataframes
    dataA = colvar_to_pandas(folder='mlcvs/data/2d-basins/',filename='COLVAR_stateA')
    dataB = colvar_to_pandas(folder='mlcvs/data/2d-basins/',filename='COLVAR_stateB')

    # Create input datasets
    xA = dataA.filter(regex='p.*').values
    xB = dataB.filter(regex='p.*').values
    names = dataA.filter(regex='p.*').columns.values

    # Create labels
    yA = np.zeros(len(dataA))
    yB = np.ones(len(dataB))

    # Concatenate
    X = np.concatenate([xA,xB],axis=0)
    y = np.concatenate([yA,yB],axis=0)

    # Shuffle (not used in test)
    #p = np.random.permutation(len(X))
    #X, y = X[p], y[p]

    # Convert np to torch 
    X = torch.tensor(X,dtype=dtype,device=device)
    y = torch.tensor(y,dtype=dtype,device=device)

    return X,y,names

def test_lda_basins(load_dataset):
    """Perform LDA on 2d-basins data folder."""

    # Load dataset
    X,y,features_names = load_dataset
    
    # Define model 
    n_features = X.shape[1]
    lda = LDA_CV(n_features)
    # Set features names (for PLUMED input)
    lda.set_params({'features_names': features_names})

    # Fit LDA
    lda.fit(X,y)

    # Project
    x_test = np.ones(2)
    y_test = lda(x_test)

    y_test = y_test.cpu()[0]
    y_test_expected = torch.tensor(-0.0960027)
    assert y_test_expected == y_test

    # Check PLUMED INPUT
    input = lda.plumed_input()
    expected_input = "lda_cv: COMBINE ARG=p.x,p.y COEFFICIENTS=0.657474,-0.75347 PERIODIC=NO"
    assert expected_input == input

def test_deeplda():
    """Define a DeepLDA object."""
    
    # define dataset
    n_data = 100 
    n_features = 10
    n_classes = 2 
    X = torch.rand((n_data,n_features)).to(device)
    y = torch.randint(low=0,high=n_classes,size=(n_data,))
    print(y)

    # split train/test
    ntrain = int(n_data*0.8)
    nvalid = int(n_data*0.2)
    train_data = [X[:ntrain],y[:ntrain]]
    valid_data = [X[ntrain:ntrain+nvalid],y[ntrain:ntrain+nvalid]]

    # Architecture
    hidden_nodes = "20,20,5"
    nodes = [int(x) for x in hidden_nodes.split(',')]
    nodes.insert(0, X.shape[1])
    n_hidden=nodes[-1]

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
        loss = model.evaluate_dataset(train_data,save_params=True)
    y2test = model(xtest)
    print(model.w.shape)
    print(y2test)

    # ASSERT if shape == n_classes-1
    expected_y2_shape = torch.rand(n_classes-1).shape
    assert y2test.shape == expected_y2_shape

@pytest.mark.slow
@pytest.mark.skip
def test_deeplda_train(load_dataset):
    """Perform DeepLDA on 2d-basins data folder."""
    
    # load dataset
    X,y,features_names = load_dataset

    # split train/test
    ntrain = 800
    nvalid = 200

    standardize_inputs = True #@param {type:"boolean"}

    train_data = [X[:ntrain],y[:ntrain]]
    valid_data = [X[ntrain:ntrain+nvalid],y[ntrain:ntrain+nvalid]]

    hidden_nodes = "30,30,5"
    nodes = [int(x) for x in hidden_nodes.split(',')]
    nodes.insert(0, X.shape[1])
    n_hidden=nodes[-1]

    # -- Parameters --
    lrate = 0.001 
    sw_reg = 0.05 
    l2_reg = 1e-5

    # MODEL
    model = DeepLDA_CV(nodes, device=device)
    model.to(device)

    #OPTIMIZER
    opt = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=l2_reg)

    # REGULARIZATION
    model.set_optimizer(opt)
    model.set_earlystopping(patience=40,min_delta=0.,consecutive=False,save_best_model=True)
    model.set_regularization(sw_reg=sw_reg)

    # TRAIN
    model.train(train_data,valid_data,info=True,log_every=10)


#if __name__ == "__main__":
    # Do something if this file is invoked on its own
    #test_lda_basins()
    #test_deeplda_train()