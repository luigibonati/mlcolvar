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

        #enforce symmetrization of correlation matrix
        self.symmetrize = True


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
            ave = torch.mean(x.T,1,keepdim=True).T
        
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

    def correlation_matrices(self,x,x_lag,w=None,w_lag=None):
        """Compute the correlation matrices between x and x_lag with weights w and w_lag

        Parameters
        ----------
        x : torch.Tensor
            first array, from x(t=0) to x(t=T-tau) 
        x_lag : torch.Tensor
            second array, from x(t=tau) to x(t=T)
        w : torch.Tensor
            weights for x, by default None
        w_lag : torch.Tensor
            weights for x_lag, by default None

        Returns
        -------
        torch.Tensor
            correlation matrices, C(0) and C(tau)

        """

        if w is None: 
            w = torch.ones(x.shape[0])
        if w_lag is None: 
            w_lag = torch.ones(x_lag.shape[0])
        
        #compute correlation matrix C0
        corr_x = torch.einsum('ij, ik, i -> jk', x, x, w )
        corr_x /= torch.sum(w)
        corr_xlag = torch.einsum('ij, ik, i -> jk', x_lag, x_lag, w_lag )
        corr_xlag /= torch.sum(w_lag)
        # enforce symmetrization
        C0 = 0.5*(corr_x + corr_xlag)

        #compute correlation matrix Clag
        corr_x_xlag = torch.einsum('ij, ik, i -> jk', x, x_lag, w_lag )
        corr_x_xlag /= torch.sum(w_lag)
        corr_xlag_x = torch.einsum('ij, ik, i -> jk', x_lag, x, w_lag )
        corr_xlag_x /= torch.sum(w_lag)
        # enforce symmetrization
        Clag = 0.5*(corr_x_xlag + corr_xlag_x)

        return C0,Clag
        
    def compute_correlation_matrix(self,x,y,w=None):
        """Compute the correlation matrix between x and y with weights w

        Parameters
        ----------
        x : torch.Tensor
            first array
        y : torch.Tensor
            second array
        w : torch.Tensor
            weights, by default None

        Returns
        -------
        torch.Tensor
            correlation matrix

        """
        # TODO Add assert on shapes

        if w is None: #TODO simplify it in the unbiased case?
            w = torch.ones(x.shape[0])
        
        #compute correlation matrix
        corr = torch.einsum('ij, ik, i -> jk', x, y, w )
        corr /= torch.sum(w)
            
        if self.symmetrize:
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

        # if correlation matrices are symmetric the Cholesky decomposition and symeig method are used
        if ( (C_lag.transpose(0, 1) == C_lag).all() and (C_0.transpose(0, 1) == C_0).all() ): 
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

        else:
            #Compute the pseudoinverse (Moore-Penrose inverse) of C_0. if det(C_0) != 0 then the usual inverse is computed
            C_new = torch.matmul(torch.pinverse(C_0),C_lag)
            #find eigenvalues and vectors of C_new
            """ TODO: torch.eig() is deprecated in favor of torch.linalg.eig()  
                      torch.linalg.eig() returns complex tensors of dtype cfloat or cdouble
                      rather than real tensors mimicking complex tensors. 
                      For future developments it would be necessary to take either only the real part
                      or only the complex part or only the magnitude of the complex eigenvectors and eigenvalues """
            eigvals, eigvecs = torch.eig(C_new,eigenvectors=True)

            #sort
            eigvals, indices = torch.sort(eigvals, 0, descending=True)
            eigvecs = eigvecs[:,indices]

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

#TODO: check if t_i = - tau / \log\lambda_i or = - tau / \log \abs(\lambda_i) 
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

        its = - lag/torch.log(self.evals_)

        return its 
        