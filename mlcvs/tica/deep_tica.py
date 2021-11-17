"""Time-lagged independent component analysis-based CV"""

__all__ = ["DeepTICA_CV"] 

import numpy as np
import torch
from torch.utils.data import DataLoader

from .tica import TICA
from ..models import NeuralNetworkCV
from ..utils.data import create_time_lagged_dataset, FastTensorDataLoader

class DeepTICA_CV(NeuralNetworkCV):
    """
    Neural network based estimator for TICA.
    Perform a non-linear featurization of the inputs with a neural-network and optimize it as to maximize autocorrelation (e.g. eigenvalues of the transfer operator approximation).

    Attributes
    ----------
    tica : mlcvs.tica.TICA 
        Number of classes
    reg_cholesky: float
        Magnitude of cholesky regularization
    epochs : int
        Number of performed epochs of training
    loss_train : list
        Loss function for train data over training. 
    loss_valid : list
        Loss function for validation data over training.
    evals_train: list
        Eigenvalues over training process.

    Methods
    -------
    __init__(layers,activation,**kwargs)
        Create a DeepTICA_CV object
    train(train_loader, valid_loader, ... , n_epochs, ... )
        Train DeepTICA CVs.
    loss_function(evals)
        Loss function for the DeepTICA CVs.
    evaluate_dataset(dataset)
        Evaluate loss function on dataset.
    set_regularization(cholesky_reg)
        Set magnitudes of regularizations.
    set_loss_function(func)
        Custom loss function.
    """

    def __init__(self, layers, activation="relu", device = None, **kwargs):
        """
        Create a DeepTICA_CV object.

        Parameters
        ----------
        layers : list
            Number neurons per layers.
        **kwargs : dict
            Additional parameters for NeuralNetworkCV class.
        """
        
        super().__init__(
            layers=layers, activation=activation, **kwargs 
        )
        self.name_ = "deeptica_cv"
        self.tica = TICA()

        # set device 
        self.device_ = device

        # custom loss function
        self.custom_loss = None

        # lorentzian regularization
        self.reg_cholesky = 0

        # training logs
        self.epochs = 0
        self.loss_train = []
        self.loss_valid = []
        self.evals_train = []
        self.log_header = True

    def set_regularization(self, cholesky_reg=1e-6):
        """
        Set magnitude of regularization on cholesky decomposition.
        - add identity matrix multiplied by `cholesky_reg` to correlation matrix.
        
        Parameters
        ----------
        cholesky_reg : float
            Regularization value for C_lag.
        """
        self.tica.reg_cholesky = cholesky_reg

    def loss_function(self,evals,objective='sum2',n_eig=0):
        """
        Loss function for the DeepTICA CV, given the eigenvalues of the generalized eigenvalue equation.

        Parameters
        ----------
        eval : torch.tensor
            Eigenvalues
        objective : str
            function of the eigenvalues to optimize (see notes)
        n_eig: int, optional
            number of eigenvalues to include in the loss (default: 0 --> all). in case of single and single2 is used to specify which eigenvalue to use.

        Notes
        -----
        A monotonic function of the eigenvalues is maximized. The following functions are implemented:
            - sum     : -sum_i (lambda_i)
            - sum2    : -sum_i (lambda_i)**2
            - gap     : -(lambda_1-lambda_2)
            - gapsum  : -sum_i (lambda_{i+1}-lambda_i)
            - its     : -sum_i (1/log(lambda_i))
            - single  : - (lambda_i)
            - single2 : - (lambda_i)**2

        Returns
        -------
        loss : torch.tensor (scalar)
            score
        """
        if self.custom_loss is None:

            #check if n_eig is given and
            if (n_eig>0) & (len(evals) < n_eig):
                raise ValueError("n_eig must be lower than the number of eigenvalues.")
            elif (n_eig==0):
                if ( (objective == 'single') | (objective == 'single2')):
                    raise ValueError("n_eig must be specified when using single or single2.")
                else:
                    n_eig = len(evals)
            elif (n_eig>0) & (objective == 'gapsum') :
                raise ValueError("n_eig parameter not valid for gapsum. only sum of all gaps is implemented.")

            loss = None
            
            if   objective == 'sum':
                loss = - torch.sum(evals[:n_eig])
            elif objective == 'sum2':
                g_lambda = - torch.pow(evals,2)
                loss = torch.sum(g_lambda[:n_eig])
            elif objective == 'gap':
                loss = - (evals[0] -evals[1])
            #elif objective == 'gapsum':
            #    loss = 0
            #    for i in range(evals.shape[0]-1):
            #        loss += - (evals[i+1] -evals[i])
            elif objective == 'its':
                g_lambda = 1 / torch.log(evals)
                loss = torch.sum(g_lambda[:n_eig])
            elif objective == 'single':
                loss = - evals[n_eig-1]
            elif objective == 'single2':
                loss = - torch.pow(evals[n_eig-1],2)
            else:
                raise ValueError("unknown objective. options: 'sum','sum2','gap','single','its'.")
        else: 
            loss = self.custom_loss(evals)

        return loss

    def set_loss_function(self, func):
        """Set custom loss function

        TODO document with an example

        Parameters
        ----------
        func : function
            custom loss function
        """
        self.custom_loss = func

    def set_average(self, Mean, Range=None):
        """Save averages for computing mean-free inputs 

        Parameters
        ----------
        Mean : torch.Tensor
            Input means
        Range : torch.Tensor, optional
            Range of inputs, by default None
        """

        if Range is None:
            Range = torch.ones_like(Mean)

        if hasattr(self,"MeanNN"):
            self.MeanNN = Mean
            self.RangeNN = Range
        else:
            self.register_buffer("MeanNN", Mean)
            self.register_buffer("RangeNN", Range)

        self.normNN = True

    def train_epoch(self, loader):
        """
        Auxiliary function for training an epoch.

        1) Calculate the NN output
        2) Remove average
        3) Compute TICA

        Parameters
        ----------
        loader: DataLoader
            training set
        """
        for data in loader:
            # =================get data===================
            x_t, x_lag, w_t, w_lag = data
            x_t, x_lag = x_t.to(self.device_), x_lag.to(self.device_)
            w_t, w_lag = w_t.to(self.device_), w_lag.to(self.device_)            
            # =================forward====================
            f_t = self.forward_nn(x_t)
            f_lag = self.forward_nn(x_lag)
            # =============compute average================
            ave_f = self.tica.compute_average(f_t,w_t)
            f_t.sub_(ave_f)
            f_lag.sub_(ave_f)
            # ===================tica=====================
            eigvals, eigvecs = self.tica.compute_TICA(data = [f_t,f_lag], 
                                                      weights = [w_t,w_lag],
                                                      save_params=True)
            self.set_average(ave_f)
            self.w = eigvecs
            # ===================loss=====================
            loss = self.loss_function(eigvals,objective=self.loss_type,n_eig=self.n_eig)
            # =================backprop===================
            self.opt_.zero_grad()
            loss.backward()
            self.opt_.step()
        # ===================log======================
        self.epochs += 1

    def train(
        self,
        train_loader=None,
        valid_loader=None,
        X=None,
        t=None,
        lag_time=None,
        standardize_inputs=True,
        standardize_outputs=True,
        loss_type='sum2', #TODO SAVE TO INTERNAL VARIABLES
        n_eig=0, 
        batch_size=-1,
        nepochs=1000,
        log_every=1,
        info=False,
    ):
        """
        Train Deep-TICA CVs. This can be performed in two ways:
        1. (preferred) taking a `train_loader` (Dataloader) built from a TimeLaggedDataset, and optionally a `valid_loader`.
        2. if input data `X` (and `time`) are given and a `time_lag` is specified a Dataloader is constructed.

        Parameters
        ----------
        train_loader: DataLoader, optional
            training set
        valid_loader: DataLoader, optional
            validation set
        X: np.array or torch.Tensor, optional
            input data, alternative to train_loader (default = None)
        t: np.array or torch.Tensor, optional
            time series (default = None)
        lag_time: float,optional
            lag_time used to construct the dataloader from `X` and `t`, (default = None)
        loss_type: str, optional
            type of objective function, see loss_function for options (default = 'sum2')
        n_eig: int
            number of eigenvalues to optimize
        standardize_inputs: bool, optional
            whether to standardize input data (default = True)
        standardize_outputs: bool, optional
            whether to standardize output data (default = True)
        batch_size: bool, optional
            number of points per batch (default = -1, single batch)
        nepochs: int, optional
            number of epochs, only if constructing dataloader internally (default = 1000)
        log_every: int, optional
            frequency of log (default = 1)
        print_info: bool, optional
            print debug info (default = False)

        See Also
        --------
        TimeLaggedDataset
            Create dataset finding time-lagged configurations
        loss_function
            Loss functions for training Deep-TICA CVs
        """
        # check device
        if self.device_ is None:
            self.device_ = next(self.nn.parameters()).device

        # assert to avoid redundancy
        if (train_loader is not None) and (X is not None):
            raise KeyError('Only one between train_loader and X can be used')
        
        # create dataloader if not given
        if X is not None:
            if lag_time is None:
                raise KeyError('lag_time must be specified when X is given')
            if t is None:
                print('WARNING: time is not given, assuming t = np.arange(len(X))')
                t = np.arange(len(X))

            dataset = create_time_lagged_dataset(X,t,lag_time)
            train_loader = FastTensorDataLoader(*dataset.tensors, batch_size=batch_size, shuffle=False) 

        # standardize inputs (unravel dataset and copy to device) #TODO check memory usage on GPU
        x_train = torch.cat([batch[0] for batch in train_loader]).to(self.device_)
        if standardize_inputs:
            self.standardize_inputs(x_train)

        # check optimizer
        if self.opt_ is None:
            self._default_optimizer()

        # save loss function options
        self.loss_type = loss_type
        self.n_eig = n_eig

        # print info
        if info:
            self.print_info()

        # train
        for ep in range(nepochs):
            #optimize
            self.train_epoch(train_loader)

            # compute scores after epoch
            loss_train = self.evaluate_dataset(train_loader,save_params=False)
            self.loss_train.append(loss_train.cpu())
            
            if valid_loader is not None:
                loss_valid = self.evaluate_dataset(valid_loader)
                self.loss_valid.append(loss_valid.cpu())
            with torch.no_grad():
                self.evals_train.append(torch.unsqueeze(self.tica.evals_,0))

            #standardize output
            if standardize_outputs:
                self.standardize_outputs(x_train)

            # earlystopping
            if self.earlystopping_ is not None:
                if valid_loader is None:
                    raise ValueError('EarlyStopping requires validation data')
                self.earlystopping_(loss_valid, model=self.state_dict(), epoch=ep)

            # log
            print_log = False
            if ((ep + 1) % log_every == 0):
                print_log = True
            elif self.earlystopping_ is not None:
                if (self.earlystopping_.early_stop):
                    print_log = True

            if print_log:
                self.print_log(
                    {
                        "Epoch": ep + 1,
                        "Train Loss": loss_train,
                        "Valid Loss": loss_valid,
                        "Eigenvalues": self.tica.evals_,
                    } if valid_loader is not None else
                    {
                        "Epoch": ep + 1,
                        "Train Loss": loss_train,
                        "Eigenvalues": self.tica.evals_,
                    },
                    spacing=[6, 12, 12, 24] if valid_loader is not None
                            else [6, 12, 24],
                    decimals=3,
                )

            # check whether to stop 
            if (self.earlystopping_ is not None) and (self.earlystopping_.early_stop):
                self.load_state_dict( self.earlystopping_.best_model )
                break

    def evaluate_dataset(self, dataset, save_params=False, unravel_dataset = False):
        """
        Evaluate loss function on dataset.

        Parameters
        ----------
        dataset : dataloader or list of batches
            dataset
        save_params: bool, optional
            save the eigenvalues/vectors of TICA into the model
        unravel_dataset: bool, optional
            unravel dataset to calculate TICA on all data instead of batches   

        Returns
        -------
        loss : torch.tensor
            score
        """
        with torch.no_grad():
            loss = 0
            n_batches = 0

            if unravel_dataset:
                batches = [batch for batch in dataset] # to deal with shuffling
                batches = [torch.cat([batch[i] for batch in batches]) for i in range(4)] 
            else: 
                batches = dataset

            for batch in batches:
                # =================get data===================
                x_t, x_lag, w_t, w_lag = batch
                x_t, x_lag = x_t.to(self.device_), x_lag.to(self.device_)
                w_t, w_lag = w_t.to(self.device_), w_lag.to(self.device_)    
                # =================forward====================
                f_t = self.forward_nn(x_t)
                f_lag = self.forward_nn(x_lag)
                # =============compute average================
                if save_params:
                    ave_f = self.tica.compute_average(f_t,w_t)
                else:
                    ave_f = self.MeanNN

                f_t.sub_(ave_f)
                f_lag.sub_(ave_f)
                # ===================tica=====================
                eigvals, eigvecs = self.tica.compute_TICA(data = [f_t,f_lag], 
                                                        weights = [w_t,w_lag],
                                                        save_params=save_params)
                if save_params:
                    self.set_average(ave_f)
                    self.w = eigvecs
                # ===================loss=====================
                loss += self.loss_function(eigvals,objective=self.loss_type,n_eig=self.n_eig)
                n_batches +=1

        return loss/n_batches


