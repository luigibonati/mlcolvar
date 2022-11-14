"""Neural network base class."""

__all__ = ["NeuralNetworkCV"]

import torch
from warnings import warn
from pathlib import Path

from torch.utils.data import TensorDataset, random_split
from ..utils.data import FastTensorDataLoader
from ..utils.optim import EarlyStopping, LRScheduler
from .utils import normalize,compute_mean_range

class NeuralNetworkCV(torch.nn.Module):
    """
    Neural Network CV base class.

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
    earlystopping_: EarlyStopping
        EarlyStopping scheduler
    normIn: bool
        Normalize inputs
    normOut: bool
        Normalize outputs
    outputHidden: bool
        Output NN last layer rather than CVs
    feature_names : list
        List of input features names

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
        self, layers, activation="relu", gaussian_random_initialization=False, **kwargs 

    ):
        """
        Define a neural network module given the list of layers.

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        activation : string
            Activation function (relu, tanh, elu, linear)
        random_initialization: bool
            if initialize the weights of the network with random values gaussian distributed N(0,1)

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

        if layers is not None:
            # Create architecture
            modules = []
            for i in range(len(layers) - 1):
                if i < len(layers) - 2:
                    modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
                    if activ is not None:
                        modules.append(activ)
                else:
                    modules.append(torch.nn.Linear(layers[i], layers[i + 1]))

            # Initialize parameters
            self.nn = torch.nn.Sequential(*modules)
            self.n_features = layers[0]
            self.n_hidden = layers[-1]

            # Initialization of the weights and offsets
            if gaussian_random_initialization:
                self.apply(self._init_weights)
                weight = torch.normal(0,1,[self.n_hidden,self.n_hidden])
            else:
                # Linear projection output
                weight = torch.eye(self.n_hidden)

            offset = torch.zeros(self.n_hidden)
            self.register_buffer("w", weight)
            self.register_buffer("b", offset)

            # Input and output normalization
            self.normIn = False
            self.register_buffer("MeanIn", torch.zeros(self.n_features))
            self.register_buffer("RangeIn", torch.ones(self.n_features))
            self.normNN = False
            self.register_buffer("MeanNN", torch.zeros(self.n_hidden))
            self.register_buffer("RangeNN", torch.ones(self.n_hidden))
            self.normOut = False
            self.register_buffer("MeanOut", torch.zeros(self.n_hidden))
            self.register_buffer("RangeOut", torch.ones(self.n_hidden))

            # Flags
            self.output_hidden = False

            # Generic attributes
            self.feature_names = ["x" + str(i) for i in range(self.n_features)]
            self.custom_train = None

        # Device
        self.device_ = None

        # Optimizer
        self.opt_ = None
        self.earlystopping_ = None
        self.lrscheduler_ = None

        # Generic attributes
        self.name_ = "NN_CV"

        # training logs
        self.epochs = 0
        self.loss_train = []
        self.loss_valid = []
        self.log_header = True

    def _init_weights(self, module):
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=1.0)
                if module.bias is not None:
                    module.bias.data.zero_()

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
            x = normalize(x, self.MeanIn, self.RangeIn)
        z = self.nn(x)
        return z

    def forward(self, x: torch.tensor) -> (torch.tensor):
        """
        Compute model output. 

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
        forward_nn : compute forward pass of NN module

        """
        z = self.forward_nn(x)

        if self.normNN:
            z = normalize(z, self.MeanNN, self.RangeNN)

        if not self.output_hidden:
            z = self.linear_projection(z)

        if self.normOut:
            z = normalize(z, self.MeanOut, self.RangeOut)
        return z

    def predict(self, X):
        """
        Alias for self.forward.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features)
            Inference data.

        Returns
        -------
        s : array-like of shape (n_samples, n_classes-1)
            Model outputs.
        """

        return self.forward(X)

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

    # Device

    def set_device(self,device):
        """ Set the device to be used and move the model to it (call model.to(device) ) 

        Parameters
        ----------
        device : torch.device
        """
        if type(device) == str:
            device = torch.device(device)
            
        self.device_=device
        self.to(device)

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

    def _set_default_optimizer(self):
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
        self.earlystopping_ = EarlyStopping(
            patience, min_delta, consecutive, log, save_best_model
        )

    def set_LRScheduler(self ,optimizer, patience=5, min_lr=1e-6, factor=0.9, log=False):
        """
        Enable LRScheduler.

        Parameters
        ----------
        optimizer : torch.optimizer
            the optimizer we are using
        patience : int, optional
            how many epochs to wait before updating the lr (default = 5)
        min_lr: float, optional
            least lr value to reduce to while updating (defaul = 1e-6)
        factor: float, optional
            factor by which the lr should be updated (default = 0.9)
        log: bool, optional
            print verbose info
        """
        self.lrscheduler_ = LRScheduler(optimizer, patience=patience, min_lr=min_lr, factor=factor, log=log
        )

    # Default function for training an epoch
    def train_epoch(self,loader):
        """
        Auxiliary function for training an epoch. 
        By default assumes a supervised task. It can be overloaded in the child class.

        Parameters
        ----------
        loader: DataLoader
            training set
        """
        for data in loader:
            # =================get data===================
            X = data[0].to(self.device_)
            y = data[1].to(self.device_)
            # =================forward====================
            H = self.forward_nn(X)
            # ===================loss======================
            loss = self.loss_function(H, y)
            # =================backprop===================
            self.opt_.zero_grad()
            loss.backward()
            self.opt_.step()
        # ===================log======================
        self.epochs += 1

    # default function for evaluating loss on dataset
    def evaluate_dataset(self, dataset, save_params=False, unravel_dataset=False):
        """
        Evaluate loss function on dataset.

        Parameters
        ----------
        dataset : dataloader or list of batches
            dataset
        save_params: bool
            save the parameters of the estimators if present (keep it with compatibility)
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
                y = batch[1].to(self.device_)
                H = self.forward_nn(X)
                # ===================loss=====================
                loss += self.loss_function(H, y)
                n_batches += 1

        return loss / n_batches

    # set custom loss
    def set_custom_train(self, function):
        """"Specify custom training function to be used for training a single epoch. 
        Overloads train_epoch function. It expects a function which takes a single arguent (a dataloader) and train for an epoch

        Parameters
        ----------
        function : function
            function to be saved

        See Also
        --------
        train_epoch:
            single epoch training 
        """
        self.custom_train = function

    # fit
    def fit(
            self,
            train_loader=None,
            valid_loader=None,
            X=None,
            y=None,
            standardize_inputs=True,
            standardize_outputs=True,
            batch_size=0,
            nepochs=1000,
            log_every=1,
            info=False,
            earlystopping=None,
            options={}
    ):
        """
        Fit the CV. Takes as input a FastTensorDataLoader/standard Dataloader constructed from a TensorDataset, or even a tuple of (colvar,labels) data. 
        It relies on the function train_epoch which defines the training loop. 

        Parameters
        ----------
        train_data: FastTensorDataLoader/DataLoader, or tuple of torch.tensors (X:input, y:labels)
            training set
        valid_data: tuple of torch.tensors (X:input, y:labels) #TODO add dataloader option?
            validation set
        X: np.array or torch.Tensor, optional
            input data, alternative to train_loader (default = None)
        y: np.array or torch.Tensor, optional
            labels (default = None)
        standardize_inputs: bool
            whether to standardize input data
        standardize_outputs: bool
            whether to standardize CVs
        batch_size: bool, optional
            number of points per batch (default = -1, single batch)
        nepochs: int, optional
            number of epochs (default = 1000)
        log_every: int, optional
            frequency of log (default = 1)
        print_info: bool, optional
            print debug info (default = False)
        options: dict, optional
            custom options for creating a dataset 

        See Also
        --------
        loss_function
            Loss function for training 
        train_epoch
            Function to train a single epoch
        """

        # check optimizer
        if self.opt_ is None:
            self._set_default_optimizer()

        # check device
        if self.device_ is None:
            self.device_ = next(self.nn.parameters()).device

        # set earlystopping variable
        self.earlystopping_ = earlystopping

        # assert to avoid redundancy
        if (train_loader is not None) and (X is not None):
            raise KeyError('Only one between train_loader and X can be used.')

        # create dataloaders if not given
        if X is not None:
            train_loader, valid_loader = self.prepare_dataloader(X,y,batch_size,options)

        # standardize inputs (unravel dataset to compute average)
        x_train = torch.cat([batch[0] for batch in train_loader])
        if standardize_inputs:
            self.standardize_inputs(x_train)

        # print info
        if info:
            self.print_info()

        # train
        for ep in range(nepochs):
            # use custom train epoch function if present
            if self.custom_train is not None:
                self.custom_train(train_loader)
            else:
                self.train_epoch(train_loader)

            loss_train = self.evaluate_dataset(train_loader, save_params=True).cpu() 
            loss_valid = self.evaluate_dataset(valid_loader).cpu()
            
            self.loss_train.append(loss_train)
            self.loss_valid.append(loss_valid)

            # standardize output
            if standardize_outputs:
                self.standardize_outputs(x_train)

            # earlystopping
            if self.earlystopping_ is not None:
                if valid_loader is None:
                    raise ValueError('EarlyStopping requires validation data')
                self.earlystopping_(loss_valid, model=self.state_dict())
            else:
                self.set_earlystopping(patience=1e30)

            # log
            if ((ep + 1) % log_every == 0) or (self.earlystopping_.early_stop):
                self.print_log(
                    {
                        "Epoch": ep + 1,
                        "Train Loss": loss_train,
                        "Valid Loss": loss_valid,
                    },
                    spacing=[6, 12, 12],
                    decimals=4,
                )

            # check whether to stop
            if (self.earlystopping_ is not None) and (self.earlystopping_.early_stop):
                self.load_state_dict(self.earlystopping_.best_model)
                break

    # Prepare dataloader
    def prepare_dataloader(self,X, y=None, batch_size=0, options={}):
        """Function for creating dataloaders if they are not given. 

        Parameters
        ----------
        X : array-like
            data
        y : array-like
            labels, default None
        batch_size : int
            default 0 (single batch)
        options: dict
            specific options, default empty

        Returns
        -------
        train_loader: FastTensorDataloader
            train loader
        valid_loader: FastTensorDataloader
            valid loader
        """

        # convert to Tensors
        if type(X) != torch.Tensor:
            X = torch.Tensor(X)

        if y is None:
            dataset = TensorDataset(X)
        else:
            if type(y) != torch.Tensor:
                y = torch.Tensor(y)
            dataset = TensorDataset(X, y)

        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size

        train_data, valid_data = random_split(dataset, [train_size, valid_size])
        train_loader = FastTensorDataLoader(train_data, batch_size)
        valid_loader = FastTensorDataLoader(valid_data)
        print('Created dataloaders')
        print('Training   set:', len(train_data))
        print('Validation set:', len(valid_data))

        return train_loader, valid_loader


    # Input / output standardization

    def standardize_inputs(self, x: torch.Tensor, print_values=False):
        """
        Standardize inputs over dataset (based on max and min).

        Parameters
        ----------
        x : torch.tensor
            reference set over which compute the standardization
        """

        Mean, Range = compute_mean_range(x, print_values)

        self.MeanIn = Mean.to(self.device_)
        self.RangeIn = Range.to(self.device_)

        #if hasattr(self,"MeanIn"):
        #    self.MeanIn = Mean
        #    self.RangeIn = Range
        #else:
        #    self.register_buffer("MeanIn", Mean)
        #    self.register_buffer("RangeIn", Range)

        self.normIn = True

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
            x = self.forward(input.to(self.device_)) # TODO CHECK FOR MEMORY ISSUE ON GPU

        Mean, Range = compute_mean_range(x, print_values)

        self.MeanOut = Mean
        self.RangeOut = Range

        #if hasattr(self,"MeanOut"):
        #    self.MeanOut = Mean.to(self.MeanOut.device)
        #    self.RangeOut = Range.to(self.RangeOut.device)
        #else:
        #    self.register_buffer("MeanOut", Mean)
        #    self.register_buffer("RangeOut", Range)

        self.normOut = True

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
                value = value.detach().cpu().numpy()
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

        # == Create folder
        Path(folder).mkdir(parents=True, exist_ok=True)

        # == Export checkpoint ==
        torch.save({
                    # FLAGS
                    'normIn': self.normIn,
                    'normOut': self.normOut,
                    'normNN': self.normNN,
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
        device = next(self.nn.parameters()).device
        fake_input = torch.zeros(self.n_features, device = device) #self.device_) #.reshape(1,self.n_features) #TODO check with plumed interface
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
                self._set_default_optimizer()
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
