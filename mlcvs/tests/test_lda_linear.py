"""
Unit and regression test for the lda module.
"""

# Import package, test suite, and other packages as needed
import pytest
import torch
import numpy as np
import pandas as pd
from mlcvs.utils.io import load_dataframe
from mlcvs.lda import LDA_CV

# set global variables
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=8)

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
    X = torch.Tensor(X) 
    y = torch.Tensor(y) 
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
    X = torch.Tensor(X)
    y = torch.Tensor(y)

    # Define model
    n_features = X.shape[1]
    lda = LDA_CV(n_features,harmonic_lda=is_harmonic_lda)

    # Fit and transform LDA
    result = lda.fit_predict(X, y)

    # Project
    x_test = torch.tensor(n_features)
    y_test = lda(x_test)
    if is_harmonic_lda:
        y_test_expected = torch.tensor(
            [0.07089982] if n_classes == 2 else [ 0.15250435, -0.27857320]
        )
    else:
        y_test_expected = torch.tensor(
            [0.24074882] if n_classes == 2 else [ 0.23162389, -0.10873218]
        )
    print(f'hlda:{is_harmonic_lda} - nclasses:{n_classes} - result:{y_test}')
    assert (y_test_expected - y_test).abs().sum() < 1e-6

def test_lda_from_dataframe():
    # params
    n_feat = 2
    n_class = 2
    n_points = 100
    np.random.seed(1)

    # fake dataset
    df = pd.DataFrame(np.random.rand(n_points,n_feat),columns=['X1','X2'] )
    y = np.zeros(len(df))
    y[:int(len(y)/2)] = int(1)
    df ['y'] = y
    
    # filter columns
    X = df.filter(regex='X')
    y = df['y']
    
    # train lda cv
    lda = LDA_CV(n_features=X.shape[1]) 
    lda.fit(X,y)

    assert (lda.feature_names == X.columns.values).all()

    s = lda.predict(X)[0]
    s_expected = torch.Tensor([-0.28010044]) 
    print(s)
    assert  torch.abs(s - s_expected) < 1e-6

@pytest.mark.parametrize("is_harmonic_lda", [False, True])
def test_lda_train_2d_model_harmonic(load_dataset_2d_classes,is_harmonic_lda):
    """Perform LDA on 2d_model data folder."""

    # Load dataset
    X, y, feature_names = load_dataset_2d_classes

    # Define model
    n_features = X.shape[1]
    lda = LDA_CV(n_features,harmonic_lda = is_harmonic_lda)
    # Set features names (for PLUMED input)
    lda.set_params({"feature_names": feature_names})
    
    print(X.dtype)

    # Fit LDA
    lda.fit(X, y)

    # Project
    x_test = np.ones(2)
    y_test = lda(x_test)
    
    print(f'hlda:{is_harmonic_lda} - result: {y_test}')
    y_test_expected = torch.tensor(
                        [-0.03565392] if is_harmonic_lda else [-0.09600264]
                      )

    assert torch.abs(y_test_expected - y_test) < 1e-6

    # Check PLUMED INPUT
    input = lda.plumed_input()
    expected_input = (
        "hlda_cv: CUSTOM ARG=p.x,p.y VAR=x0,x1 FUNC=+0.689055*x0-0.724709*x1 PERIODIC=NO\n" if is_harmonic_lda
        else "lda_cv: CUSTOM ARG=p.x,p.y VAR=x0,x1 FUNC=+0.657474*x0-0.753477*x1 PERIODIC=NO\n"
    )
    assert expected_input == input


if __name__ == "__main__":
    test_lda_harmonic_nclasses(2,False)
