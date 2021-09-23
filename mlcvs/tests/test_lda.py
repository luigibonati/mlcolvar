"""
Unit and regression test for the lda module.
"""

# Import package, test suite, and other packages as needed
import torch
import numpy as np
from mlcvs.io import colvar_to_pandas
from mlcvs.lda import LDA, LDA_CV

def test_lda_basins():
    """Perform LDA on 2d-basins data folder."""
    #dtype = torch.float32
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    dataA = colvar_to_pandas(folder='mlcvs/data/2d-basins/',filename='COLVAR_stateA')
    dataB = colvar_to_pandas(folder='mlcvs/data/2d-basins/',filename='COLVAR_stateB')

    # Create input dataset
    xA = dataA.filter(regex='p.*').values
    xB = dataB.filter(regex='p.*').values
    names = dataA.filter(regex='p.*').columns.values
    n_features = xA.shape[1]

    # Create labels
    yA = np.zeros(len(dataA))
    yB = np.ones(len(dataB))

    X = np.concatenate([xA,xB],axis=0)
    y = np.concatenate([yA,yB],axis=0)

    # Transform to Pytorch Tensors
    #X = torch.tensor(X,dtype=dtype,device=device)
    #y = torch.tensor(y,dtype=dtype,device=device)
    
    # Define model 
    lda = LDA_CV(n_features)
    # Set features name (for PLUMED input)
    lda.set_params({'features_names': names})

    # Fit LDA
    lda.fit(X,y)

    # Project
    x_test = np.ones(2)
    y_test = lda(x_test)

    y_test = y_test.cpu()[0]
    y_test_expected = torch.tensor(-0.0960027)
    
    #ASSERT 
    print(type(y_test),type(y_test_expected))
    assert y_test_expected == y_test

    # Check PLUMED INPUT

    input = lda.plumed_input()
    expected_input = "lda_cv: COMBINE ARG=p.x,p.y COEFFICIENTS=0.657474,-0.75347 PERIODIC=NO"
    
    # ASSERT
    assert expected_input == input

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    test_lda_basins()