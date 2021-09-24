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

@pytest.mark.slow
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

    hidden_nodes = "30,30,5" #@param {type:"raw"}
    nodes = [int(x) for x in hidden_nodes.split(',')]
    nodes.insert(0, X.shape[1])
    n_hidden=nodes[-1]

    # -- Parameters --
    lrate = 0.001 #@param {type:"slider", min:0.0001, max:0.005, step:0.0001}
    sw_reg = 0.05 #@param {type:"number"}
    l2_reg = 1e-5 #@param {type:"number"}

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