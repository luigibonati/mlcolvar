"""
Unit and regression test for the linearCV class.
"""

# Import package, test suite, and other packages as needed
import pytest
import torch
from mlcvs.models import LinearCV, NeuralNetworkCV

# set global variables
dtype = torch.float32
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.mark.parametrize("n_input", [1, 2])
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_LinearCV(n_input, dev):
    """Test Linear CV class."""

    # Set device
    device = torch.device(dev)
    if (dev == "cuda") and (not torch.cuda.is_available()):
        pytest.skip("cuda not available")

    # Define model
    # n_input = 2
    cv = LinearCV(n_features=n_input, device=device)

    # Define inputs
    x = torch.ones(n_input).to(device)

    # Propagate with default values
    y = cv.transform(x)
    expected_y = torch.tensor([1.0 for _ in range(n_input)]).to(device)
    # ASSERT arrays
    assert torch.equal(y, expected_y)

    # Change parameters
    cv.set_weights(0.5 * torch.ones(n_input))
    cv.set_offset(0.5 * torch.ones(n_input))

    # Propagate (call method)
    y2 = cv(x)
    expected_y2 = torch.tensor(0.25 * n_input).to(device)
    # ASSERT
    assert torch.equal(y2, expected_y2)


@pytest.mark.parametrize("n_input", [1, 2])
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_NeuralNetworkCV(n_input, dev):
    """Test NN CV class."""

    # Set device
    device = torch.device(dev)
    if (dev == "cuda") and (not torch.cuda.is_available()):
        pytest.skip("cuda not available")

    # Parameters
    n_input = 1
    n_output = 2

    # Define model
    net = NeuralNetworkCV(layers=[n_input, 5, n_output], activation="relu")
    net.to(device)

    # Define inputs
    x = torch.ones(n_input).to(device)

    # Forward
    y = net(x)

    # Assert
    expected_y_shape = torch.rand(n_output).shape
    assert y.shape == expected_y_shape

    # Project output along single direction
    w = torch.ones(n_output).to(device)
    b = torch.ones(n_output).to(device)
    net.set_params({"w": w})
    net.set_params({"b": b})

    # Forward 2
    y2 = net(x)

    # Assert
    expected_y2_shape = torch.tensor(1).shape
    assert y2.shape == expected_y2_shape
