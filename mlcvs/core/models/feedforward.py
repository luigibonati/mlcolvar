import torch 
import pytorch_lightning as pl

__all__ = ["FeedForward"]

def get_activation_from_string(activation : str):
    activ = None
    if activation == "relu":
        activ = torch.nn.ReLU(True)
    elif activation == "elu":
        activ = torch.nn.ELU(True)
    elif activation == "tanh":
        activ = torch.nn.Tanh()
    elif activation == "linear":
        print("WARNING: no activation selected")
    elif activation == None:
        pass
    else:
        raise ValueError(
            f"Unknown activation: {activation}. options: 'relu','elu','tanh','linear'. "
        )
    return activ

def parse_nn_options( options : str, n_layers : int ):
    # If an iterable is given cheeck that its length matches the number of NN layers
    if hasattr(options, '__iter__') and not isinstance(options, str):
        if len(options) != n_layers:
            raise ValueError(f'Length of options: {options} ({len(options)} should be equal to number of layers ({n_layers})).')
        options_list = options
    # if a single value is given, repeat options to all layers but for the output one
    else: 
        options_list = [ options for _ in range(n_layers-1) ]
        options_list.append(None)
    
    return options_list

class FeedForward(pl.LightningModule):

    def __init__(self, 
                layers : list, 
                activation: str or list = "relu", 
                dropout: int or list = None, 
                batchnorm: str or list = None, 
                **kwargs 
    ):
        """
        Define a feedforward neural network given the list of layers.

        Optionally dropout and batchnorm can be applied (the order is activation -> dropout -> batchnorm).

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        activation : string or list of strings (activation per each layer)
            Activation function (options: relu, tanh, elu, linear)
        dropout : float or list of floats (dropout per each layer)
            Dropout after each layer
        batchnorm : bool or list of booleans (batchnorm per each layer)
            Batchnorm after each layer
        **kwargs:
            Optional arguments passed to torch.nn.module
        """
        super().__init__(**kwargs)

        # Parse layers
        if not isinstance(layers[0],int):
            raise TypeError('layers should be a list-type of integers.')
        
        # Parse options per each hidden layer
        # -- activation
        activation_list = parse_nn_options(activation,  len(layers)-1)
        # -- dropout
        dropout_list    = parse_nn_options(dropout,     len(layers)-1)
        # -- batchnorm
        batchnorm_list  = parse_nn_options(batchnorm,   len(layers)-1)
        
        # Create network
        modules = []
        for i in range(len(layers) - 1):
            modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
            activ, drop, norm = activation_list[i], dropout_list[i], batchnorm_list[i]

            if activ is not None:
                modules.append(get_activation_from_string(activ))

            if drop is not None:
                modules.append(torch.nn.Dropout(p=drop,inplace=True))
            
            if norm:
                modules.append(torch.nn.BatchNorm1d(layers[i+1]))

        # store model and attributes
        self.nn     = torch.nn.Sequential(*modules)
        self.in_features   = layers[0]
        self.out_features  = layers[-1]

    def extra_repr(self) -> str:
        repr = f"in_features={self.in_features}, out_features={self.out_features}"
        return repr

    def forward(self, x: torch.tensor) -> (torch.tensor):
        return self.nn(x)

def test_feedforward():
    torch.manual_seed(42)
    
    in_features = 2
    layers = [2,10,10,1]
    model = FeedForward(layers,activation='relu')
    print(model)

    X = torch.zeros(in_features)
    print(model(X))

    # custom options per layers
    activ   = ['relu','tanh',None]
    dropout = 0.5
    batchnorm = True

    model = FeedForward(layers, activation=activ, dropout=dropout, batchnorm=batchnorm)
    print(model)


if __name__ == "__main__":
    test_feedforward()