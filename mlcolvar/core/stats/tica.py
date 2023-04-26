"""Time-lagged independent component analysis"""

__all__ = ["TICA"]

import torch
from mlcolvar.core.stats import Stats
from mlcolvar.core.stats.utils import correlation_matrix, cholesky_eigh, compute_average, reduced_rank_eig
from mlcolvar.core.transform.utils import batch_reshape
import warnings

class TICA(Stats):
    """ 
    Time-lagged independent component analysis base class.
    """

    def __init__(self, in_features, out_features = None):
        """
        Initialize a TICA object.
        """
        super().__init__()

        # save attributes
        self.in_features  = in_features 
        self.out_features = out_features if out_features is not None else in_features

        # buffers
        # tica eigenvectors
        self.register_buffer("evecs" ,torch.eye(in_features,self.out_features))
        # mean to obtain mean free inputs
        self.register_buffer("mean", torch.zeros(in_features))

        # init other attributes
        self.evals = None
        self.C_0 = None
        self.C_lag = None

        # Regularization
        self.reg_C_0 = 1e-6

    def extra_repr(self) -> str:
        repr = f"in_features={self.in_features}, out_features={self.out_features}"
        return repr

    def compute(self, data, weights = None, remove_average=True, save_params=False, algorithm = 'least_squares'):
        """Perform TICA computation.

        Parameters
        ----------
        data : [list of torch.Tensors]
            Time-lagged configurations (x_t, x_{t+lag})
        weights : [list of torch.Tensors], optional
            Weights at time t and t+lag, by default None
        remove_average: bool, optional
            whether to make the inputs mean free, by default True
        save_params : bool, optional
            Save parameters of estimator, by default False
        algorithm : str, optional
            Algorithm to use, by default 'least_squares'. Options are 'least_squares' and 'reduced_rank'. Both algorithms are described in [1]_.
        Returns
        -------
        torch.Tensor
            Eigenvalues
        
        References
        ----------
        .. [1] V. Kostic, P. Novelli, A. Maurer, C. Ciliberto, L. Rosasco, and M. Pontil, "Learning Dynamical Systems via Koopman Operator Regression in Reproducing Kernel Hilbert Spaces" (2022).

        """
        # parse args

        x_t, x_lag = data 
        w_t, w_lag = None, None
        if weights is not None:
            w_t, w_lag = weights

        if remove_average:
            x_ave = compute_average(x_t,w_t)
            x_t = x_t.sub(x_ave)
            x_lag = x_lag.sub(x_ave)

        C_0 = correlation_matrix(x_t,x_t,w_t)
        C_lag = correlation_matrix(x_t,x_lag,w_lag)

        if (algorithm == 'reduced_rank') and (self.out_features >= self.in_features):
            warnings.warn('out_features is greater or equal than in_features. reduced_rank is equal to least_squares.')
            algorithm = 'least_squares'
        
        if algorithm == 'reduced_rank':
            evals, evecs = reduced_rank_eig(C_0, C_lag, self.reg_C_0, rank = self.out_features)
        elif algorithm != 'least_squares':
            raise ValueError(f'algorithm {algorithm} not recognized. Options are least_squares and reduced_rank.')
        else:
            evals, evecs = cholesky_eigh(C_lag,C_0,self.reg_C_0,n_eig=self.out_features) 
            
        if save_params:
            self.evals = evals
            self.evecs = evecs
            if remove_average:
                self.mean = x_ave

        return evals, evecs

    def timescales(self, lag):
        """Return implied timescales from eigenvalues and lag-time.

        Parameters
        ----------
        lag : float
            lag-time

        Returns
        -------
        its : tensor
            implied timescales

        Notes
        -----
        If `lambda_i` are the eigenvalues and `tau` the lag-time, the implied times are given by:
        
        .. math:: t_i = - tau / \log\lambda_i
        """

        its = - lag/torch.log(torch.abs(self.evals))

        return its 

    def forward(self, x: torch.Tensor) -> (torch.Tensor):
        """
        Compute linear combination with saved eigenvectors

        Parameters
        ----------
        x: torch.Tensor
            input

        Returns
        -------
        out : torch.Tensor
            output
        """
        mean = batch_reshape( self.mean, x.size() )
        
        return torch.matmul(x.sub(mean), self.evecs)
        
def test_tica():
    in_features = 2
    X = torch.rand(100,in_features)*100
    x_t = X[:-1]
    x_lag = X[1:]
    w_t = torch.rand(len(x_t))
    w_lag = w_t
    
    # direct way, compute tica function
    tica = TICA(in_features,out_features=2)
    print(tica)
    tica.compute([x_t,x_lag],[w_t,w_lag], save_params=True)
    s = tica(X)
    print(X.shape,'-->',s.shape)
    print('eigvals',tica.evals)
    print('timescales', tica.timescales(lag=10))

    # step by step
    tica = TICA(in_features)
    C_0 = correlation_matrix(x_t,x_t)
    C_lag = correlation_matrix(x_t,x_lag)
    print(C_0.shape,C_lag.shape)

    evals, evecs = cholesky_eigh(C_lag,C_0) 
    print(evals.shape,evecs.shape)

    print('>> batch') 
    s = tica(X)
    print(X.shape,'-->',s.shape)
    print('>> single')
    X2 = X[0]
    s2 = tica(X2)
    print(X2.shape,'-->',s2.shape)

def test_reduced_rank_tica():
    in_features = 10
    X = torch.rand(100,in_features)*100
    x_t = X[:-1]
    x_lag = X[1:]
    w_t = torch.rand(len(x_t))
    w_lag = w_t
    
    # direct way, compute tica function
    tica = TICA(in_features,out_features=5)
    print(tica)
    tica.compute([x_t,x_lag],[w_t,w_lag], save_params=True, algorithm='reduced_rank')
    s = tica(X)
    print(X.shape,'-->',s.shape)
    print('eigvals',tica.evals)
    print('timescales', tica.timescales(lag=10))

    # step by step
    tica = TICA(in_features)
    C_0 = correlation_matrix(x_t,x_t)
    C_lag = correlation_matrix(x_t,x_lag)
    print(C_0.shape,C_lag.shape)

    evals, evecs = evals, evecs = reduced_rank_eig(C_0, C_lag, 1e-6, rank = 5)
    print(evals.shape,evecs.shape)

    print('>> batch') 
    s = tica(X)
    print(X.shape,'-->',s.shape)
    print('>> single')
    X2 = X[0]
    s2 = tica(X2)
    print(X2.shape,'-->',s2.shape)

if __name__ == "__main__":
    test_tica()
    test_reduced_rank_tica()