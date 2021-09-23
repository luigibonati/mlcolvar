"""
Unit and regression test for the linearCV class.
"""

# Import package, test suite, and other packages as needed
import torch
from mlcvs.models import LinearCV

def test_linearCV():
    """Define linear combination."""

    # Define model
    n_input = 2
    cv = LinearCV(n_features=n_input)

    # Define inputs
    x = torch.ones(n_input)*1

    # Propagate with default values
    y = cv.transform(x)
    expected_y = torch.tensor(2.)
    # ASSERT
    assert y == expected_y 

    # Change parameters
    cv.set_weights(0.5*torch.ones(n_input))
    cv.set_offset(0.5*torch.ones(n_input))

    # Propagate (call method)
    y2 = cv(x)
    expected_y2 = torch.tensor(0.5)
    # ASSERT
    assert y2 == expected_y2

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    test_linearCV()