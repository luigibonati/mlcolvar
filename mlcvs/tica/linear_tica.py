"""Time-lagged independent component analysis-based CV"""

__all__ = ["TICA_CV"] 

import numpy as np
import pandas as pd
import torch

from .tica import TICA,look_for_configurations
from ..models import LinearCV

class TICA_CV(LinearCV):

    def __init__(self, n_features, **kwargs):
        """Create a Linear TICA CV

        Parameters
        ----------
        n_features : int
            Number of input features
        device : str, optional
            device, by default "auto"
        """
        super().__init__(n_features=n_features, **kwargs)

        self.name_ = "tica_cv"
        self.tica = TICA()

    def train(self, X, t = None, lag = 10):
        """Fit TICA given time-lagged data (and weights). 

        Parameters
        ----------
        X : numpy array, pandas dataframe or torch.Tensor
            Input data
        t : numpy array, pandas dataframe or torch.Tensor, optional
            Time array, by default None -> np.arange(0,len(X))
        lag : int, optional
            lag-time, by default 10

        See Also
        --------
        train_forward : train and project along TICA components
        """
                
        # if DataFrame save feature names
        if type(X) == pd.DataFrame:
            if 'time' in X.columns:
                t = X['time'].values
                X = X.drop(columns='time')
            self.feature_names = X.columns.values
            X = X.values #torch.Tensor(X.values).to(self.device_)     
        #elif type(X) != torch.Tensor:
        #    X = torch.Tensor(X).to(self.device_) 

        # time 
        if t is None:
            t = np.arange(0,len(X))

        # time
        if type(t) == pd.DataFrame:
            t = t.values

        if len(X) != len(t):
            raise ValueError(f'length of X is {len(X)} while length of t is {len(t)}')

        # find time-lagged configurations
        x_t, x_lag, w_t, w_lag = look_for_configurations(X,t,lag)


        # compute mean-free variables
        ave = self.tica.compute_average(x_t,w_t)
        x_t.sub_(ave)
        x_lag.sub_(ave)

        # perform TICA
        _, eigvecs = self.tica.compute_TICA(data = [x_t,x_lag], 
                                            weights = [w_t,w_lag])

        # save parameters for estimator
        self.set_average(ave)
        self.w = eigvecs

    def train_forward(self, X, t = None, lag = 10):
        """Train TICA CV and project data

        Parameters
        ----------
        X : numpy array, pandas dataframe or torch.Tensor
            Input data
        t : numpy array, pandas dataframe or torch.Tensor, optional
            Time array, by default None -> np.arange(0,len(X))
        lag : int, optional
            lag-time, by default 10

        Returns
        -------
        torch.Tensor
            projection of input data along TICA components

        See Also
        --------
        train : train TICA estimator
        """

        self.train(X, t, lag)
        return self.forward(X)

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

        if hasattr(self,"MeanIn"):
            self.MeanIn = Mean
            self.RangeIn = Range
        else:
            self.register_buffer("MeanIn", Mean)
            self.register_buffer("RangeIn", Range)

        self.normIn = True

    def set_regularization(self, cholesky_reg):
        """
        Set regularization for cholesky decomposition.

        Parameters
        ----------
        cholesky_reg : float
            Regularization value.

        """
        self.lda.reg_cholesky = cholesky_reg