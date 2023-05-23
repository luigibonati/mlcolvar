#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in mlcolvar.core.nn.feedforward.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

from mlcolvar.core.nn.feedforward import FeedForward


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('activation', [
    'relu', 'elu', 'softplus', 'shifted_softplus',
    ['relu', 'tanh', None],
    ['relu', 'tanh', 'relu'],
])
@pytest.mark.parametrize('dropout', [
    None,
    0.5,
    [0.2, 0.3, None],
    [0.2, 0.3, 0.4],
])
@pytest.mark.parametrize('batchnorm', [
    False,
    True,
    [True, True, False],
    [True, False, True],
])
@pytest.mark.parametrize('last_layer_activation', [False, True])
def test_feedforward(activation, dropout, batchnorm, last_layer_activation):
    """The model is constructed with the correct layers."""
    in_features = 2
    layers = [in_features, 10, 10, 1]
    max_n_activ_layers = len(layers) - 1

    # Create model.
    model = FeedForward(
        layers=layers,
        activation=activation,
        dropout=dropout,
        batchnorm=batchnorm,
        last_layer_activation=last_layer_activation
    )

    # Check batchnorm.
    if isinstance(batchnorm, list):
        n_batchnorm = sum(batchnorm)
    elif last_layer_activation:
        n_batchnorm = int(batchnorm) * max_n_activ_layers
    else:
        n_batchnorm = int(batchnorm) * (max_n_activ_layers-1)
    assert sum(isinstance(l, torch.nn.BatchNorm1d) for l in model.nn) == n_batchnorm

    # Check dropout.
    if isinstance(dropout, list):
        n_dropout = sum(d is not None for d in dropout)
    elif last_layer_activation:
        n_dropout = int(dropout is not None) * max_n_activ_layers
    else:
        n_dropout = int(dropout is not None) * (max_n_activ_layers-1)
    assert sum(isinstance(l, torch.nn.Dropout) for l in model.nn) == n_dropout

    # Check activation.
    if isinstance(activation, list):
        n_activations = sum(a is not None for a in activation)
    elif last_layer_activation:
        n_activations = int(activation is not None) * max_n_activ_layers
    else:
        n_activations = int(activation is not None) * (max_n_activ_layers-1)
    n_linear = len(layers) - 1
    assert len(model.nn) == n_linear + n_activations + n_dropout + n_batchnorm

    # Test that the forward doesn't explode.
    batch_size = 2
    x = torch.zeros((batch_size, in_features))
    model(x)
