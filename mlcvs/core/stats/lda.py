"""Linear discriminant analysis"""

import torch
from .utils import cholesky_eigh

from mlcvs.core.stats import Stats

__all__ = ['LDA']

class LDA(Stats):
    """
    Fisher's discriminant class.

    Attributes
    ----------
    n_features : int
        Number of features
    n_states : int
        Number of states
    evals_ : torch.Tensor
        LDA eigenvalues
    evecs_ : torch.Tensor
        LDA eigenvectors
    S_b_ : torch.Tensor
        Between scatter matrix
    S_w_ : torch.Tensor
        Within scatter matrix
    sw_reg : float
        Regularization to S_w matrix
    """

    def __init__(self, in_features : int, n_states : int, mode : str = 'standard'):
        """
        Initialize a LDA object.

        Parameters
        ----------
        in_features : int
            number of input features
        n_states : int
            number of states
        mode : string
            options: {'standard','harmonic'}, standard or harmonic variant
        """
        super().__init__()

        # Save attributes
        self.n_states = n_states
        self.in_features = in_features 
        self.out_features = n_states - 1
        if (mode == 'standard') or (mode=='harmonic'):
            self.mode = mode
        else:
            raise ValueError(f'LDA mode should be standard or harmonic, not {mode}.')

        # create eigenvector buffer
        self.register_buffer("evecs", torch.eye(in_features,self.out_features))

        # initialize other attributes
        self.evals = None
        self.S_b = None
        self.S_w = None

        # Regularization
        self.sw_reg = 1e-6

    def extra_repr(self) -> str:
        repr = f"in_features={self.in_features}, out_features={self.out_features}"
        if self.mode=='harmonic':
            repr += f" mode={self.mode}"
        return repr

    def compute(self, X, labels, save_params = True):
        """
        Compute LDA eigenvalues and eigenvectors. 
        First compute the scatter matrices S_between (S_b) and S_within (S_w) and then solve the generalized eigenvalue problem.  

        Parameters
        ----------
        X : array-like of shape (n_samples, in_features)
            Training data.
        labels : array-like of shape (n_samples,)
            states labels.
        save_params: bool, optional
            Whether to store parameters in model

        Returns
        -------
        eigvals : torch.Tensor
            LDA eigenvalues (n_states-1)
        eigvecs : torch.Tensor
            LDA eigenvectors (n_feature,n_states-1)

        Notes
        -----
        The eigenvecs object which is returned is a matrix whose column eigvecs[:,i] is the eigenvector associated to eigvals[i]
        """

        S_b, S_w = self.compute_scatter_matrices(X,labels,save_params)
        evals, evecs = cholesky_eigh(S_b,S_w,self.sw_reg,n_eig=self.n_states-1)
        if save_params:
            self.evals = evals
            self.evecs = evecs

        return evals,evecs

    def compute_scatter_matrices(self, X, labels, save_params=True):
        """
        Compute between scatter and within scatter matrices.

        Parameters
        ----------
        X : array-like of shape (n_samples, in_features)
            Training data.
        labels : array-like of shape (n_samples,)
            states labels.
        save_params: bool, optional
            Whether to store parameters in model

        Returns
        -------
        S_b,S_w : arrays of shape (in_features,in_features)
            Between and within scatter matrices
        """
        # device
        device = X.device

        # sizes
        N, d = X.shape
        self.in_features = d

        # states
        states = torch.unique(labels)
        n_states = len(states)

        # Mean centered observations for entire population
        X_bar = X - torch.mean(X, 0, True)
        # Total scatter matrix (cov matrix over all observations)
        S_t = X_bar.t().matmul(X_bar) / (N - 1)
        # Define within scatter matrix and compute it
        S_w = torch.Tensor().new_zeros((d, d)).to(device)
        if self.mode == 'harmonic':
            S_w_inv = torch.Tensor().new_zeros((d, d)).to(device)
        # Loop over states to compute means and covs
        for i in states:
            # check which elements belong to class i
            X_i = X[torch.nonzero(labels == i).view(-1)]
            # compute mean centered obs of class i
            X_i_bar = X_i - torch.mean(X_i, 0, True)
            # count number of elements
            N_i = X_i.shape[0]
            if N_i == 0:
                continue
            # LDA
            S_w += X_i_bar.t().matmul(X_i_bar) / ((N_i - 1) * n_states)

            # HLDA
            if self.mode == 'harmonic':
                inv_i = X_i_bar.t().matmul(X_i_bar) / ((N_i - 1) * n_states)
                S_w_inv += inv_i.inverse()

        if self.mode == 'harmonic':
            S_w = S_w_inv.inverse()

        # Compute S_b from total scatter matrix
        S_b = S_t - S_w

        if save_params:
            self.S_b = S_b
            self.S_w = S_w

        return S_b, S_w

    
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
        return torch.matmul(x, self.evecs)

def test_lda():
    in_features = 2
    n_states = 2
    
    torch.manual_seed(42)
    X = torch.rand(100,in_features)*100
    y = torch.randint(n_states,(100,1)).squeeze(1)

    # standard
    lda = LDA(in_features,n_states)
    print(lda)
    S_b, S_w = lda.compute_scatter_matrices(X,y)
    print(S_w,S_b)
    evals,evecs = lda.compute(X,y,True)
    print(lda.S_w.shape,lda.S_b.shape)
    print(evals.shape,evecs.shape)
    s = lda(X)
    print(s.shape)
    assert (s.ndim == 2) and (s.shape[1]==n_states-1)

    # harmonic variant
    hlda = LDA(in_features, n_states, mode='harmonic')
    print(hlda)
    hlda.compute(X,y)
    s = lda(X)
    
if __name__ == "__main__":
    test_lda()