#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Variational Autoencoder collective variable.
"""

__all__ = ["FeedForward"]


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Optional, Union

import torch
import lightning
from mlcolvar.core.nn.utils import get_activation, parse_nn_options


# =============================================================================
# STANDARD FEED FORWARD
# =============================================================================

class FeedForward(lightning.LightningModule):
    """Define a feedforward neural network given the list of layers.

    Optionally dropout and batchnorm can be applied (the order is activation -> dropout -> batchnorm).
    """

    def __init__(
            self,
            layers : list,
            activation: Union[str, list] = "relu",
            dropout: Optional[Union[float, list]] = None,
            batchnorm: Union[bool, list] = False,
            last_layer_activation: bool = False,
            **kwargs
    ):
        """Constructor.

        Parameters
        ----------
        layers : list
            Number of neurons per layer.
        activation : string or list[str], optional
            Add activation function (options: relu, tanh, elu, linear). If a
            ``list``, this must have length ``len(layers)-1``, and ``activation[i]``
            controls whether to add the activation to the ``i``-layer.
        dropout : float or list[float], optional
            Add dropout with this probability after each layer. If a ``list``,
            this must have length ``len(layers)-1``, and ``dropout[i]`` specifies
            the the dropout probability for the ``i``-th layer.
        batchnorm : bool or list[bool], optional
            Add batchnorm after each layer. If a ``list``, this must have
            length ``len(layers)-1``, and ``batchnorm[i]`` controls whether to
            add the batchnorm to the ``i``-th layer.
        last_layer_activation : bool, optional
            If ``True`` and activation, dropout, and batchnorm are added also to
            the output layer when ``activation``, ``dropout``, or ``batchnorm``
            (i.e., they are not lists). Otherwise, the output layer will be linear.
            This option is ignored for the arguments among ``activation``, ``dropout``,
            and ``batchnorm`` that are passed as lists.
        **kwargs:
            Optional arguments passed to torch.nn.Module
        """
        super().__init__(**kwargs)

        # Parse layers
        if not isinstance(layers[0], int):
            raise TypeError('layers should be a list-type of integers.')
        
        # Parse options per each hidden layer
        n_layers = len(layers) - 1
        # -- activation
        activation_list = parse_nn_options(activation, n_layers, last_layer_activation)
        # -- dropout
        dropout_list = parse_nn_options(dropout, n_layers, last_layer_activation)
        # -- batchnorm
        batchnorm_list = parse_nn_options(batchnorm, n_layers, last_layer_activation)
        
        # Create network
        modules = []
        for i in range(len(layers) - 1):
            modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
            activ, drop, norm = activation_list[i], dropout_list[i], batchnorm_list[i]

            if activ is not None:
                modules.append(get_activation(activ))

            if drop is not None:
                modules.append(torch.nn.Dropout(p=drop,inplace=True))
            
            if norm:
                modules.append(torch.nn.BatchNorm1d(layers[i+1]))

        # store model and attributes
        self.nn = torch.nn.Sequential(*modules)
        self.in_features = layers[0]
        self.out_features = layers[-1]

    #def extra_repr(self) -> str:
    #    repr = f"in_features={self.in_features}, out_features={self.out_features}"
    #    return repr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)
