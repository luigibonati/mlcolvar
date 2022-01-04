"""Time-lagged independent component analysis-based CV"""

__all__ = ["TICA"] # TODO add other functions?

import torch

class TICA:
    """ 
    Time-lagged independent component analysis base class.
    """

    def __init__(self):
        """
        Initialize TICA object.
        """

        # initialize attributes
        self.evals_ = None
        self.evecs_ = None
        self.n_features = None 

        # Regularization
        self.reg_cholesky = None 


    def compute_average(self, x, w = None):
        """Compute (weighted) average to obtain mean-free inputs

        Parameters
        ----------
        x : torch.Tensor
            Input data
        w : torch.Tensor, optional
            Weights, by default None

        Returns
        -------
        torch.Tensor
            (weighted) mean of inputs

        """
        if w is not None:
            ave = torch.einsum('ij,i ->j',x,w)/torch.sum(w)
        else:
            ave = torch.average(x)
        
        return ave

    def compute_TICA(self, data, weights = None, n_eig = 0, save_params=False):
        """Perform TICA computation

        Parameters
        ----------
        data : [list of torch.Tensors]
            Time-lagged configurations (x_t, x_{t+lag})
        weights : [list of torch.Tensors], optional
            Weights at time t and t+lag, by default None
        n_eig : int, optional
            number of eigenfunctions to compute, by default 0 = all
        save_params : bool, optional
            Save parameters of estimator, by default False

        Returns
        -------
        torch.Tensor
            Eigenvalues

        """
        x_t, x_lag = data 
        if weights is not None:
            w_t, w_lag = weights

        C_0 = self.compute_correlation_matrix(x_t,x_t,w_t)
        C_lag = self.compute_correlation_matrix(x_t,x_lag,w_lag)
        evals, evecs = self.solve_tica_eigenproblem(C_0,C_lag,n_eig=n_eig,save_params=save_params) 

        return evals, evecs

    def compute_correlation_matrix(self,x,y,w=None,symmetrize=True):
        """Compute the correlation matrix between x and y with weights w

        Parameters
        ----------
        x : torch.Tensor
            first array
        y : torch.Tensor
            second array
        w : torch.Tensor
            weights, by default None
        symmetrize : bool, optional
            Enforce symmetrization, by default True

        Returns
        -------
        torch.Tensor
            correlation matrix

        """
        # TODO Add assert on shapes

        d = x.shape[1]

        if w is None: #TODO simplify it in the unbiased case?
            w = torch.ones(x.shape[0])
        
        #compute correlation matrix
        corr = torch.einsum('ij, ik, i -> jk', x, y, w )
        corr /= torch.sum(w)
            
        if symmetrize:
            corr = 0.5*(corr + corr.T)

        return corr

    def solve_tica_eigenproblem(self,C_0,C_lag,n_eig=0,save_params=False):
        """Compute generalized eigenvalue problem : C_lag * wi = lambda_i * C_0 * w_i

        Parameters
        ----------
        C_0 : torch.Tensor
            correlation matrix at lag-time 0
        C_lag : torch.Tensor
            correlation matrix at lag-time lag
        n_eig : int, optional
            number of eigenvectors to compute, by default 0
        save_params : bool, optional
            save estimator parameters, by default False

        Returns
        -------
        torch.Tensor
            eigenvalues
        torch.Tensor
            eigenvectors

        Notes
        -----
        The eigenvecs object which is returned is a matrix whose column eigvecs[:,i] is the eigenvector associated to eigvals[i]
        """ 
        #cholesky decomposition
        if self.reg_cholesky is not None:
            reg = self.reg_cholesky*torch.eye(C_0.shape[0]).to(C_0.device)
            L = torch.cholesky(C_0+reg,upper=False)
        else:
            L = torch.cholesky(C_0,upper=False)
            
        L_t = torch.t(L)
        L_ti = torch.inverse(L_t)
        L_i = torch.inverse(L)
        C_new = torch.matmul(torch.matmul(L_i,C_lag),L_ti)

        #find eigenvalues and vectors of C_new
        eigvals, eigvecs = torch.symeig(C_new,eigenvectors=True)
        #sort
        eigvals, indices = torch.sort(eigvals, 0, descending=True)
        eigvecs = eigvecs[:,indices]
        
        #return to original eigenvectors
        eigvecs = torch.matmul(L_ti,eigvecs)

        #normalize them
        for i in range(eigvecs.shape[1]): # maybe change in sum along axis?
            norm=eigvecs[:,i].pow(2).sum().sqrt()
            eigvecs[:,i].div_(norm)
        #set the first component positive
        eigvecs.mul_( torch.sign(eigvecs[0,:]).unsqueeze(0).expand_as(eigvecs) )

        #keep only first n_eig eigvals and eigvecs
        if n_eig>0:
            eigvals = eigvals[:n_eig]
            eigvecs = eigvecs[:,:n_eig]
        
        if save_params:
            self.evals_ = eigvals 
            self.evecs_ = eigvecs
    
        return eigvals, eigvecs

    def timescales(self, lag):
        """Return implied timescales as given by:

        .. math:: t_i = \frac{\tau}{\log\ \lambda_i}

        where $lambda_i$ are the eigenvalues and $\tau$ the lag-time.

        Args:
            lag (float): lag-time

        Returns:
            its (array): implied timescales
        """

        its = - lag/torch.log(self.evals_)

        return its 
        