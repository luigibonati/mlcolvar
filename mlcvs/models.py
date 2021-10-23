"""Collective variables base models."""

__all__ = ["LinearCV", "NeuralNetworkCV"]

import torch
import numpy as np
from warnings import warn
from . import optim


class LinearCV(torch.nn.Module):
    """
    Linear CV base class.

    Attributes
    ----------
    w : torch.Tensor
        Weights array of linear model
    b : torch.Tensor
        Offset array 
    n_features : int
        Number of input features
    feature_names : list
        List of input features names
    device_ : torch.Device
        Device used for the model

    Methods
    -------
    __init__(n_features)
        Create a linear model.
    forward(X)
        Project data along linear model
    get_params()
        Return saved parameters
    set_params(dict)
        Set parameters via dictionaries
    set_weights(w)
        Set coefficients
    set_offset(b)
        Set linear model
    plumed_input()
        Generate PLUMED input file
    """

    def __init__(self, n_features, device="auto", **kwargs):
        super().__init__(**kwargs)

        # Device

        if device == "auto":
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_ = device

        # Initialize parameters
        self.n_features = n_features
        weight = torch.eye(n_features,  device=self.device_)
        offset = torch.zeros(n_features,  device=self.device_)
        self.register_buffer("w", weight)
        self.register_buffer("b", offset)

        # Generic attributes
        self.name_ = "LinearCV"
        self.feature_names = ["x" + str(i) for i in range(n_features)]

        # Flags

    def forward(self, X):
        """
        Project data along linear components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features)
            Inference data.

        Returns
        -------
        s : array-like of shape (n_samples, n_classes-1)
            Linear projection of inputs.
        """
        if type(X) != torch.Tensor:
            X = torch.Tensor(X, device=self.device_)

        s = torch.matmul(X - self.b, self.w)

        return s

    def set_weights(self, w):
        """
        Set weights of linear combination.

        Parameters
        ----------
        w : torch.tensor
            weights

        """
        self.w = w.to(self.device_)

    def set_offset(self, b):
        """
        Set linear offset

        Parameters
        ----------
        b : torch.tensor
            offset

        """
        self.b = b.to(self.device_)

    def get_params(self):
        """
        Return saved parameters.

        Returns
        -------
        out : namedtuple
            Parameters
        """
        return vars(self)

    def set_params(self, dict_params):
        """
        Set parameters.

        Parameters
        ----------
        dict_params : dictionary
            Parameters

        """

        for key in dict_params.keys():
            if hasattr(self, key):
                setattr(self, key, dict_params[key])
            else:
                raise AttributeError(
                    f"{self.__class__.__name__} has no attribute {key}."
                )

    def plumed_input(self):
        """
        Generate PLUMED input file

        Returns
        -------
        out : string
            PLUMED input file
        """
        # count number of output components
        weights = self.w.cpu().numpy()
        offset = self.b.cpu().numpy()
        n_cv = 1 if weights.ndim == 1 else weights.shape[1]

        out = ""
        for i in range(n_cv):
            if n_cv == 1:
                out += f"{self.name_}: COMBINE ARG="
            else:
                out += f"{self.name_}{i+1}: COMBINE ARG="
            for j in range(self.n_features):
                out += f"{self.feature_names[j]},"
            out = out[:-1]

            out += " COEFFICIENTS="
            for j in range(self.n_features):
                out += str(np.round(weights[j, i], 6)) + ","
            out = out[:-1]
            if not np.all(offset == 0):
                out += " PARAMETERS="
                for j in range(self.n_features):
                    out += str(np.round(offset[j, i], 6)) + ","
            out = out[:-1]

            out += " PERIODIC=NO"
        return out


class NeuralNetworkCV(torch.nn.Module):
    """
    Neural Network CV base class

    Attributes
    ----------
    nn : nn.Module
        Neural network object
    n_features : int
        Number of input features
    device_: torch.device
        Device (cpu or cuda)
    opt_: torch.optimizer
        Optimizer
    earlystopping_: optim.EarlyStopping
        EarlyStopping scheduler
    normIn: bool
        Normalize inputs
    normOut: bool
        Normalize outputs
    outputHidden: bool
        Output NN last layer rather than CVs
    feature_names : list
        List of input features names

    Methods
    -------
    __init__(layers,activation,device)
        Create a neural network object.
    forward(x)
        Compute model output.
    forward_nn(x)
        Compute NN output.
    linear_projection(H)
        Apply linear projection to NN output.
    set_optimizer(opt)
        Save optimizer
    set_earlystopping(patience,min_delta, ...)
        Enable EarlyStopping
    standardize_inputs(x)
        Standardize inputs over dataset
    standardize_outputs(x)
        Standardize outputs over dataset
    get_params()
        Return saved parameters
    set_params(dict)
        Set parameters via dictionaries
    print_info()
        Display information about model
    export(folder,checkpoint_name,traced_name)
        Save checkpoint for Pytorch and Torchscript model
    load_checkpoint(checkpoint_path)
        Load checkpoint
    plumed_input()
        Generate PLUMED input file

    Examples
    --------
    Create a neural-network with 20 inputs, one hidden layer with 10 nodes and 5 outputs
    
    >>> net = NeuralNetworkCV(layers=[20,10,5], activation = 'relu')
    >>> net
    NeuralNetworkCV(
      (nn): Sequential(
        (0): Linear(in_features=20, out_features=10, bias=True)
        (1): ReLU(inplace=True)
        (2): Linear(in_features=10, out_features=5, bias=True)
      )
    )

    """

    def __init__(
        self, layers, activation="relu", device="auto",  **kwargs
    ):
        """
        Define a neural network module given the list of layers.

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        activation : string
            Activation function (relu, tanh, elu, linear)
        devi

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
            print("WARNING: linear activation selected")
        else:
            raise ValueError(
                "Unknown activation. options: 'relu','elu','tanh','linear'. "
            )

        # Create architecture
        modules = []
        for i in range(len(layers) - 1):
            if i < len(layers) - 2:
                modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
                if activ is not None:
                    modules.append(activ)
            else:
                modules.append(torch.nn.Linear(layers[i], layers[i + 1]))

        # Device
        if device == "auto":
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_ = device

        # Initialize parameters
        self.nn = torch.nn.Sequential(*modules)
        self.n_features = layers[0]
        self.n_hidden = layers[-1]

        # Linear projection output
        weight = torch.eye(self.n_hidden, device=self.device_)
        offset = torch.zeros(self.n_hidden, device=self.device_)
        self.register_buffer("w", weight)
        self.register_buffer("b", offset)

        # Input and output normalization
        self.normIn = False
        self.normOut = False

        MeanIn = torch.zeros(self.n_features, device=self.device_)
        RangeIn = torch.ones(self.n_features, device=self.device_)
        self.register_buffer("MeanIn", MeanIn)
        self.register_buffer("RangeIn", RangeIn)

        # Flags
        self.output_hidden = False

        # Optimizer
        self.opt_ = None
        self.earlystopping_ = None

        # Generic attributes
        self.name_ = "NN_CV"
        self.feature_names = ["x" + str(i) for i in range(self.n_features)]

    # Forward pass
    def forward_nn(self, x: torch.tensor) -> (torch.tensor):
        """
        Compute NN output.

        Parameters
        ----------
        x : torch.tensor
            input data

        Returns
        -------
        z : torch.tensor
            NN output

        See Also
        --------
        forward : compute forward pass of the mdoel
        """
        if self.normIn:
            x = self._normalize(x, self.MeanIn, self.RangeIn)
        z = self.nn(x)
        return z

    def forward(self, x: torch.tensor) -> (torch.tensor):
        """
        Compute model output.

        Parameters
        ----------
        x : torch.tensor
            input data

        Returns
        -------
        z : torch.tensor
            CVs

        See Also
        --------
        forward_nn : compute forward pass of NN module

        """
        z = self.forward_nn(x)
        if not self.output_hidden:
            z = self.linear_projection(z)
        if self.normOut:
            z = self._normalize(z, self.MeanOut, self.RangeOut)
        return z

    def linear_projection(self, H):
        """
        Apply linear projection to NN output.

        Parameters
        ----------
        H : array-like of shape (n_samples, n_features) or (n_features)
            Inference data.

        Returns
        -------
        s : array-like of shape (n_samples, n_classes-1)
            Linear projection of inputs.
        """

        s = torch.matmul(H - self.b, self.w)

        return s

    # Optimizer

    def set_optimizer(self, opt):
        """
        Set optimizer object. If not set uses ADAM with default parameters.

        Parameters
        ----------
        opt : torch.optimizer
            Optimizer
        """
        self.opt_ = opt

    def _default_optimizer(self):
        """
        Initialize default optimizer (ADAM).
        """
        self.opt_ = torch.optim.Adam(self.parameters())

    def set_earlystopping(
        self, patience=5, min_delta=0, consecutive=True, log=False, save_best_model=True
    ):
        """
        Enable earlystopping.

        Parameters
        ----------
        patience : int
            how many epochs to wait before stopping when loss is not improving (default = 5)
        min_delta : float, optional
            minimum difference between new loss and old loss to be considered as an improvement (default = 0)
        consecutive: bool, optional
            whether to consider cumulative or consecutive patience
        log: bool, optional
            print info when counter increases
        save_best: bool, optional
            store the best model
        """
        self.earlystopping_ = optim.EarlyStopping(
            patience, min_delta, consecutive, log, save_best_model
        )
        self.best_valid = None
        self.best_model = None

    # Input / output standardization

    def _compute_mean_range(self, x: torch.Tensor, print_values=False):
        """
        Compute mean and range of values over dataset (for inputs / outputs standardization)
        """
        Max, _ = torch.max(x, dim=0)
        Min, _ = torch.min(x, dim=0)

        Mean = (Max + Min) / 2.0
        Range = (Max - Min) / 2.0

        if print_values:
            print("Standardization enabled.")
            print("Mean:", Mean.shape, "-->", Mean)
            print("Range:", Range.shape, "-->", Range)
        if (Range < 1e-6).nonzero().sum() > 0:
            print(
                "[Warninthe follig] Normalization: owing features have a range of values < 1e-6:",
                (Range < 1e-6).nonzero(),
            )
            Range[Range < 1e-6] = 1.0

        return Mean, Range

    def standardize_inputs(self, x: torch.Tensor, print_values=False):
        """
        Standardize inputs over dataset (based on max and min).

        Parameters
        ----------
        x : torch.tensor
            reference set over which compute the standardization
        """

        Mean, Range = self._compute_mean_range(x, print_values)

        self.normIn = True
        self.MeanIn = Mean 
        self.RangeIn = Range

    def standardize_outputs(self, input: torch.Tensor, print_values=False):
        """
        Standardize outputs over dataset (based on max and min).

        Parameters
        ----------
        x : torch.tensor
            reference set over which compute the standardization
        """
        # disable normOut for unbiased cv evaluation
        self.normOut = False
        with torch.no_grad():
            x = self.forward(input)

        Mean, Range = self._compute_mean_range(x, print_values)

        if hasattr(self,"MeanOut"):
            self.MeanOut = Mean
            self.RangeOut = Range
        else:
            self.register_buffer("MeanOut", Mean)
            self.register_buffer("RangeOut", Range)

        self.normOut = True

    def _normalize(
        self, x: torch.Tensor, Mean: torch.Tensor, Range: torch.Tensor
    ) -> (torch.Tensor):
        """
        Compute standardized inputs/outputs (internal).

        Parameters
        ----------
        x: torch.Tensor
            input/output
        Mean: torch.Tensor
            mean values to be subtracted.
        Range: torch.Tensor
            interval range to be divided by.

        Returns
        -------
        out : torch.Tensor
            standardized inputs/outputs
        """

        # if shape ==

        if x.ndim == 2:
            batch_size = x.size(0)
            x_size = x.size(1)

            Mean_ = Mean.unsqueeze(0).expand(batch_size, x_size)
            Range_ = Range.unsqueeze(0).expand(batch_size, x_size)
        elif x.ndim == 1:
            Mean_ = Mean
            Range_ = Range
        else:
            raise ValueError(
                "Input tensor must of shape (n_features) or (n_batch,n_features)."
            )

        return x.sub(Mean_).div(Range_)

        # Parameters

    def get_params(self):
        """
        Return saved parameters.

        Returns
        -------
        out : namedtuple
            Parameters
        """
        return vars(self)

    def set_params(self, dict_params):
        """
        Set parameters.

        Parameters
        ----------
        dict_params : dictionary
            Parameters
        """

        for key in dict_params.keys():
            if hasattr(self, key):
                setattr(self, key, dict_params[key])
            else:
                raise AttributeError(
                    f"{self.__class__.__name__} has no attribute {key}."
                )

    # Info

    def print_info(self):
        """
        Display information about model.
        """

        print("================INFO================")
        print("[MODEL]")
        print(self)
        print("\n[OPTIMIZER]")
        print(self.opt_)
        print("\n[PARAMETERS]")
        print(self.get_params())
        print("====================================")

    # Log

    def print_log(self, log_dict, spacing=None, decimals=3):
        """
        Utility function for training log.

        Parameters
        ----------
        log_dict : dict
            training log values
        spacing: int
            columns width
        decimals: int
            number of decimals in log

        """
        if spacing is None:
            spacing = [16 for i in range(len(log_dict))]
        if self.log_header:
            for i, key in enumerate(log_dict.keys()):
                print("{0:<{width}s}".format(key, width=spacing[i]), end="")
            print("")
            self.log_header = False

        for i, value in enumerate(log_dict.values()):
            if type(value) == int:
                print("{0:<{width}d}".format(value, width=spacing[i]), end="")

            if (type(value) == torch.Tensor) or (
                type(value) == torch.nn.parameter.Parameter
            ):
                value = value.cpu().numpy()
                if value.shape == ():
                    print(
                        "{0:<{width}.{dec}f}".format(
                            value, width=spacing[i], dec=decimals
                        ),
                        end="",
                    )
                else:
                    for v in value:
                        print("{0:<6.3f}".format(v), end=" ")
        print("")

    def export(self, folder, checkpoint_name = 'model_checkpoint.pt', 
                             traced_name = 'model.ptc'):
        """
        Save checkpoint for Pytorch and Torchscript model.

        Parameters
        ----------
        folder : str
            export folder
        checkpoint_name : str
            name of Pytorch checkpoint
        libtorch_name : str
            name of traced model (for Libtorch)

        Notes
        -----
        As there is not yet a convention on naming extensions of models, we follow here the following:
            - .pt for PyTorch models
            - .ptc for PyTorch Compiled (i.e. Torchscript) models
        
        """

        # == Export checkpoint ==
        torch.save({
                    # FLAGS
                    'normIn': self.normIn,
                    'normOut': self.normOut,
                    'feature_names': self.feature_names,
                    # STATE DICT
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.opt_.state_dict() if self.opt_ is not None else None,
                    # TRAINING LCURVES
                    'epochs': self.epochs if hasattr(self, 'epochs') else None,
                    'loss_train': self.loss_train if hasattr(self, 'loss_train') else None,
                    'loss_valid': self.loss_valid if hasattr(self, 'loss_valid') else None,   
                    }, folder+checkpoint_name)

        # == Export jit model ==
        fake_input = torch.zeros(self.n_features, device=self.device_) #.reshape(1,self.n_features) #TODO check with plumed interface
        mod = torch.jit.trace(self, fake_input)
        mod.save(folder+traced_name)

    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            location of checkpoint (.pt)

        """
        checkpoint = torch.load(checkpoint_path)

        # STATE DICT
        self.load_state_dict(checkpoint['model_state_dict'])

        if checkpoint['optimizer_state_dict'] is not None:
            try:
                self.opt_.load_state_dict(checkpoint['optimizer_state_dict'])
            except AttributeError as e:
                warn('Optimizer not set. Initializing default one.')
                self._default_optimizer()
                self.opt_.load_state_dict(checkpoint['optimizer_state_dict'])
            
        for key,val in checkpoint.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def plumed_input(self):
        """
        Generate PLUMED input file

        Returns
        -------
        out : string
            PLUMED input file
        """

        out = ""
        out += f"{self.name_}: PYTORCH_MODEL FILE=model.ptc ARG="
        for feat in self.feature_names:
            out += f"{feat},"
        out = out[:-1]

        return out
