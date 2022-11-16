"""Time-lagged independent component analysis-based CV"""

__all__ = ["DeepTICA_CV"] 

import numpy as np
import torch

from .tica import TICA
from ..models import NeuralNetworkCV
from ..utils.data import create_time_lagged_dataset, FastTensorDataLoader
from torch.utils.data import random_split

class DeepTICA_CV(NeuralNetworkCV):
    """
    Neural network-based TICA CV.
    Perform a non-linear featurization of the inputs with a neural-network and optimize it as to maximize autocorrelation (e.g. eigenvalues of the transfer operator approximation).

    Attributes
    ----------
    tica : mlcvs.tica.TICA 
        TICA-object
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

    """

    def __init__(self, layers, activation="relu", gaussian_random_initialization=False ,device = None, **kwargs):
        """
        Create a DeepTICA_CV object.

        Parameters
        ----------
        layers : list
            Number neurons per layers.
        random_initialization: bool
            if initialize the weights of the network with random values uniform distributed in (0,1]
        **kwargs : dict
            Additional parameters for NeuralNetworkCV class.
        """
        
        super().__init__(
            layers=layers, activation=activation, gaussian_random_initialization=gaussian_random_initialization, **kwargs 
        )
        self.name_ = "deeptica_cv"
        self.tica = TICA()

        # lorentzian regularization
        self.reg_cholesky = 0

        # default loss function options
        self.loss_type = 'sum2'
        self.n_eig = 0

        # additional training logs
        self.logs['tica_eigvals'] = []

        # send model to device
        self.set_device(device) 

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

    def set_loss_function(self,objective='sum2',n_eig=0):
        """Set loss function parameters.

        Parameters
        ----------
        objective : str, optional
            function of the eigenvalues to optimize (see 'loss_function' for the options), by default 'sum2'
        n_eig: int, optional
            number of eigenvalues to include in the loss (default: 0 --> all). in case of single and single2 is used to specify which eigenvalue to use.
        
        See Also
        --------
        loss_function: loss function 
        """
        self.loss_type = objective
        self.n_eig = n_eig

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

        return loss

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
                                                      save_params=False)
            self.set_average(ave_f)
            self.w = eigvecs
            # ===================loss=====================
            loss = self.loss_function(eigvals,objective=self.loss_type,n_eig=self.n_eig)
            # =================backprop===================
            self.opt_.zero_grad()
            loss.backward()
            self.opt_.step()


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

            if save_params:
                self.logs['tica_eigvals'].append(self.tica.evals_)

        return loss/n_batches

    # Prepare dataloader
    def prepare_dataloader(self, X, y=None, batch_size=0, options={ 'lag_time': None } ) :
        """Function for creating dataloaders if they are not given. 

        Parameters
        ----------
        X : array-like
            data
        t : array-like
            time, default None
        batch_size : int
            default 0 (signle batch)

        Returns
        -------
        train_loader: FastTensorDataloader
            train loader
        valid_loader: FastTensorDataloader
            valid loader
        """

        # create dataloader if not given
        if X is not None:
            lag_time = options['lag_time']
            if lag_time is None:
                raise KeyError('keyword lag_time missing from options dictionary')
            if y is None:
                print('WARNING: time (y) is not given, assuming t = np.arange(len(X))')
                y = np.arange(len(X))

        dataset = create_time_lagged_dataset(X,y,lag_time)
        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size
        train_data, valid_data = random_split(dataset, [train_size, valid_size])
        train_loader = FastTensorDataLoader(train_data, batch_size, shuffle=False)
        valid_loader = FastTensorDataLoader(valid_data)
        print('Created dataloaders')
        print('Training   set:', len(train_data))
        print('Validation set:', len(valid_data))

        return train_loader, valid_loader


