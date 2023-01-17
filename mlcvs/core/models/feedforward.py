import torch 
import pytorch_lightning as pl

__all__ = ["FeedForward"]

class FeedForward(pl.LightningModule):

    def __init__(
        self, layers, activation="relu", **kwargs 
    ):
        """
        Define a simple neural network given the list of layers.

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        activation : string
            Activation function (relu, tanh, elu, linear)
        **kwargs:
            Optional arguments passed to torch base class
        """
        super().__init__(**kwargs)

        # get activation function
        activ = None
        if activation == "relu":
            activ = torch.nn.ReLU(True)
        elif activation == "elu":
            activ = torch.nn.ELU(True)
        elif activation == "tanh":
            activ = torch.nn.Tanh()
        elif activation == "linear":
            print("WARNING: no activation selected")
        else:
            raise ValueError(
                "Unknown activation. options: 'relu','elu','tanh','linear'. "
            )

        # Create architecture
        if not isinstance(layers[0],int):
            raise TypeError('layers should be a list-type of integers.')
        
        modules = []
        for i in range(len(layers) - 1):
            if i < len(layers) - 2:
                modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
                if activ is not None:
                    modules.append(activ)
            else:
                modules.append(torch.nn.Linear(layers[i], layers[i + 1]))

        # store model and attributes
        self.nn     = torch.nn.Sequential(*modules)
        self.n_in   = layers[0]
        self.n_out  = layers[-1]

    def forward(self, x: torch.tensor) -> (torch.tensor):
        return self.nn(x)

def test_feedforward():
    torch.manual_seed(42)
    
    n_in = 2
    layers = [2,10,1]
    model = FeedForward(layers,activation='relu')
    
    print(model)

    X = torch.zeros(n_in)
    print(model(X))

if __name__ == "__main__":
    test_feedforward()