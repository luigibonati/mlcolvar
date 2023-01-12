"""Autoencoding CVs."""

__all__ = ["AutoEncoderCV"]

import torch
from mlcvs.models import NeuralNetworkCV
from mlcvs.models.utils import normalize, unnormalize

class AutoEncoderCV (NeuralNetworkCV):
    """
    Autoencoding collective variable

    Attributes
    ----------
    encoder : nn.Module
        Neural network encoder
    decoder : nn.Module
        Neural network decoder
    n_features : int
        Number of input features
    device_: torch.device
        Device (cpu or cuda)
    opt_: torch.optimizer
        Optimizer
    earlystopping_: EarlyStopping
        EarlyStopping scheduler
    normIn: bool
        Normalize inputs
    normOut: bool
        Normalize outputs
    feature_names : list
        List of input features names
    """

    def __init__(
        self, encoder_layers, decoder_layers=None, device=None, activation='relu', gaussian_random_initialization=False, **kwargs
    ):
        """
        Define an autoencoder given given the list of layers.

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        activation : string
            Activation function (relu, tanh, elu, linear)
        random_initialization: bool
            if initialize the weights of the network with random values gaussian distributed N(0,1)

        """
        super().__init__(layers=None, **kwargs)

        # get activation function
        activ = None
        if activation == "relu":
            activ = torch.nn.ReLU(True)
        elif activation == "elu":
            activ = torch.nn.ELU(True)
        elif activation == "tanh":
            activ = torch.nn.Tanh()
        elif activation == "linear":
            print("WARNING: linear activation selected")
        else:
            raise ValueError(
                "Unknown activation. options: 'relu','elu','tanh','linear'. "
            )

        # Initialize encoder
        modules = []
        for i in range(len(encoder_layers) - 1):
            if i < len(encoder_layers) - 2:
                modules.append(torch.nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
                if activ is not None:
                    modules.append(activ)
            else:
                modules.append(torch.nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
        
        self.encoder = torch.nn.Sequential(*modules)

        # Initialize decoder
        if decoder_layers is None:
            decoder_layers = encoder_layers[::-1]

        modules = []
        for i in range(len(decoder_layers) - 1):
            if i < len(decoder_layers) - 2:
                modules.append(torch.nn.Linear(decoder_layers[i], decoder_layers[i + 1]))
                if activ is not None:
                    modules.append(activ)
            else:
                modules.append(torch.nn.Linear(decoder_layers[i], decoder_layers[i + 1]))
        
        self.decoder = torch.nn.Sequential(*modules)

        # Initialize parameters
        self.n_features = encoder_layers[0]
        self.n_hidden   = encoder_layers[-1]

        # Initialization of the weights and offsets
        if gaussian_random_initialization:
            self.apply(self._init_weights)

        # Input and output normalization
        self.normIn = False
        self.register_buffer("MeanIn", torch.zeros(self.n_features))
        self.register_buffer("RangeIn", torch.ones(self.n_features))
        self.normOut = False
        self.register_buffer("MeanOut", torch.zeros(self.n_hidden))
        self.register_buffer("RangeOut", torch.ones(self.n_hidden))

        # Generic attributes
        self.name_ = "AE_CV"
        self.feature_names = ["x" + str(i) for i in range(self.n_features)]

        # send model to device
        self.set_device(device) 

    # Forward pass
    def forward(self, x: torch.tensor) -> (torch.tensor):
        """
        Compute latent space (encoder). 

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        z : torch.Tensor
            CVs

        See Also
        --------
        encode_decode : compute encoder and decoder pass

        """
        if self.normIn:
            x = normalize(x, self.MeanIn, self.RangeIn)

        x = self.encoder(x)

        if self.normOut:
            x = normalize(x, self.MeanOut, self.RangeOut)
            
        return x

    def encode_decode(self, x: torch.tensor) -> (torch.tensor):
        """
        Compute forward pass (encoder+decoder). 

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        x : torch.Tensor
            reconstructed data

        See Also
        --------
        forward : compute only encoder (CVs)

        """
        x = self.forward(x) #input norm + encoder
        x = self.decoder(x) 
        if self.normIn: # add back normalization
            x = unnormalize(x, self.MeanIn, self.RangeIn)

        return x

    def forward_nn(self, x: torch.tensor) -> (torch.tensor):
        """
        Compute NN output (alias for encode_decode).

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        x : torch.Tensor
            reconstructed data

        See Also
        --------
        forward : compute only encoder (CVs)

        """
        x = self.encode_decode(x)
        return x


    # Loss function
    def loss_function(self, X, X_hat):
        """
        Reconstruction loss.

        Parameters
        ----------
        X : torch.tensor
            NN output
        X_hat : torch.tensor
            Input

        Returns
        -------
        loss : torch.tensor
            loss function
        """
        #if self.custom_loss is None: # TODO REMOVE

        func = torch.nn.MSELoss()
        loss = func(X,X_hat)

        return loss

    # Define function for training an epoch
    def train_epoch(self,loader):
        """
        Auxiliary function for training an epoch. 

        Parameters
        ----------
        loader: DataLoader
            training set
        """
        for data in loader:
            # =================get data===================
            X = data[0].to(self.device_)
            # =================forward====================
            X2 = self.forward_nn(X)
            # ===================loss======================
            loss = self.loss_function(X,X2)
            # =================backprop===================
            self.opt_.zero_grad()
            loss.backward()
            self.opt_.step()

    # default function for evaluating loss on dataset
    def evaluate_dataset(self, dataset, save_params=False, unravel_dataset=False):
        """
        Evaluate loss function on dataset.

        Parameters
        ----------
        dataset : dataloader or list of batches
            dataset
        save_params: bool
            save the parameters of the estimators if present (keep it for compatibility)
        unravel_dataset: bool, optional
            unravel dataset to calculate loss on all dataset instead of averaging over batches

        Returns
        -------
        loss : torch.tensor
            loss value
        """
        with torch.no_grad():
            loss = 0
            n_batches = 0

            if unravel_dataset:
                batches = [batch for batch in dataset]  # to deal with shuffling
                batches = [torch.cat([batch[i] for batch in batches]) for i in range(2)]
            elif type(dataset) == list:
                batches = [dataset]
            else:
                batches = dataset

            for batch in batches:
                # =================get data===================
                X = batch[0].to(self.device_)
                # =================forward====================
                X2 = self.forward_nn(X)
                # ===================loss=====================
                loss += self.loss_function(X,X2)
                n_batches += 1

        return loss / n_batches


