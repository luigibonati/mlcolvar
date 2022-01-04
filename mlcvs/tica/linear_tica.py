"""Time-lagged independent component analysis-based CV"""

__all__ = ["TICA_CV"] 

import numpy as np
import pandas as pd
import torch

from .tica import TICA
from ..models import LinearCV
from ..utils.data import find_time_lagged_configurations

class TICA_CV(LinearCV):
    """ Linear TICA CV.

    Attributes
    ----------
    tica : mlcvs.tica.TICA 
        TICA-object.

    """
    def __init__(self, n_features, **kwargs):
        """Create a Linear TICA CV

        Parameters
        ----------
        n_features : int
            Number of input features
        """
        super().__init__(n_features=n_features, **kwargs)

        self.name_ = "tica_cv"
        self.tica = TICA()

    def fit(self, X, t = None, lag = 10, logweights = None):
        """Fit TICA given time-lagged data (and optional weights). 

        Parameters
        ----------
        X : numpy array, pandas dataframe or torch.Tensor
            Input data
        t : numpy array, pandas dataframe or torch.Tensor, optional
            Time array, by default None -> np.arange(0,len(X))
        lag : int, optional
            lag-time, by default 10
        logweights: array, optional
            logarithm of the weights of the configurations

        See Also
        --------
        fit_predict : train and project along TICA components
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

        # if weights are given, rescale the time before finding time lagged configurations
        if logweights is not None:
            # compute time increment in simulation time t
            dt = np.round(t[1]-t[0],3)
            # sanitize logweights
            logweights = torch.Tensor(logweights)
            logweights -= torch.max(logweights)
            lognorm = torch.logsumexp(logweights,0)
            logweights /= lognorm
            # compute instantaneus time increment in rescaled time t'
            d_tprime = torch.exp(logweights)*dt
            # calculate cumulative time t'
            tprime = torch.cumsum(d_tprime,0)
        else:
            tprime = t

        # find time-lagged configurations
        x_t, x_lag, w_t, w_lag = find_time_lagged_configurations(X,tprime,lag)

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

    def fit_predict(self, X, t = None, lag = 10):
        """Train TICA CV and project data

        Parameters
        ----------
        X : numpy array, pandas dataframe or torch.Tensor
            Input data
        t : numpy array, pandas dataframe or torch.Tensor, optional
            Time array, by default None -> np.arange(0,len(X))
        lag : int, optional
            lag-time, by default 10
        logweights: array, optional
            logarithm of the weights of the configurations

        Returns
        -------
        torch.Tensor
            projection of input data along TICA components

        See Also
        --------
        fit : train TICA estimator
        """

        self.fit(X, t, lag)
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
        self.tica.reg_cholesky = cholesky_reg