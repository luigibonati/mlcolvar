import torch
import torch.nn.functional as F
import math


class Shifted_Softplus(torch.nn.Softplus):
    """Element-wise softplus function shifted as to pass from the origin."""
    
    def __init__(self, beta=1, threshold=20):
        super(Shifted_Softplus, self).__init__(beta, threshold)

    def forward(self, input):
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0


def get_activation(activation : str):
    """ Return activation module given string. """
    activ = None
    if activation == "relu":
        activ = torch.nn.ReLU(True)
    elif activation == "elu":
        activ = torch.nn.ELU(True)
    elif activation == "tanh":
        activ = torch.nn.Tanh()
    elif activation == "softplus":
        activ = torch.nn.Softplus()
    elif activation == "shifted_softplus":
        activ = Shifted_Softplus()
    elif activation == "linear":
        print("WARNING: no activation selected")
    elif activation == None:
        pass
    else:
        raise ValueError(
            f"Unknown activation: {activation}. options: 'relu','elu','tanh','softplus','shifted_softplus','linear'. "
        )
    return activ


def parse_nn_options(options: str, n_layers: int, last_layer_activation: bool):
    """Parse args per layer of the NN.

    If a single value is given, repeat options to all layers but for the output one,
    unless ``last_layer_activation is True``, in which case the option is repeated
    also for the output layer.
    """
    # If an iterable is given cheeck that its length matches the number of NN layers
    if hasattr(options, '__iter__') and not isinstance(options, str):
        if len(options) != n_layers:
            raise ValueError(f'Length of options: {options} ({len(options)} should be equal to number of layers ({n_layers})).')
        options_list = options
    # if a single value is given, repeat options to all layers but for the output one
    else:
        if last_layer_activation:
            options_list = [options for _ in range(n_layers)]
        else:
            options_list = [options for _ in range(n_layers-1)]
            options_list.append(None)
    
    return options_list
