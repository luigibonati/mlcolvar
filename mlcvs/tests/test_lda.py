"""
Unit and regression test for the lda module.
"""

# Import package, test suite, and other packages as needed
import torch
import numpy as np
from mlcvs.io import colvar_to_pandas
from mlcvs.lda import LinearDiscriminant

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

    # Create labels
    yA = np.zeros(len(dataA))
    yB = np.ones(len(dataB))

    X = np.concatenate([xA,xB],axis=0)
    y = np.concatenate([yA,yB],axis=0)

    # Transform to Pytorch Tensors
    #X = torch.tensor(X,dtype=dtype,device=device)
    #y = torch.tensor(y,dtype=dtype,device=device)
    
    # Perform LDA
    lda = LinearDiscriminant()
    lda.set_features_names(names)
    lda.fit(X,y)

    input = lda.plumed_input()
    
    expected_input = "lda: COMBINE ARG=p.x,p.y COEFFICIENTS=0.657474,-0.753477 PERIODIC=NO"
    
    assert expected_input == input
