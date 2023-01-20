import torch

__all__ = ['LDA']

class LDA(torch.nn.Module):
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

    def __init__(self, n_in, n_states, harmonic_lda = False):
        """
        Create a LDA object

        Parameters
        ----------
        n_in : int
            number of input features
        n_states : int
            number of states
        harmonic_lda : bool
            Harmonic variant of LDA
        """
        super().__init__()

        # Save attributes
        self.n_states = n_states
        self.n_in = n_in 
        self.n_out = n_states - 1
        self.harmonic_lda = harmonic_lda

        # create eigenvector buffer
        self.register_buffer("evecs" , torch.eye(n_in,self.n_out))

        # initialize other attributes
        self.evals = None
        self.S_b = None
        self.S_w = None

        # Regularization
        self.sw_reg = 1e-6

    def compute_scatter_matrices(self, X, labels, save_params=True):
        """
        Compute between scatter and within scatter matrices.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_in)
            Training data.
        labels : array-like of shape (n_samples,)
            states labels.
        save_params: bool, optional
            Whether to store parameters in model

        Returns
        -------
        S_b,S_w : arrays of shape (n_in,n_in)
            Between and within scatter matrices
        """
        # device
        device = X.device

        # sizes
        N, d = X.shape
        self.n_in = d

        # states
        states = torch.unique(labels)
        n_states = len(states)
        self.n_states = n_states

        # Mean centered observations for entire population
        X_bar = X - torch.mean(X, 0, True)
        # Total scatter matrix (cov matrix over all observations)
        S_t = X_bar.t().matmul(X_bar) / (N - 1)
        # Define within scatter matrix and compute it
        S_w = torch.Tensor().new_zeros((d, d)).to(device)
        if self.harmonic_lda:
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

            # XLDA
            if self.harmonic_lda:
                inv_i = X_i_bar.t().matmul(X_i_bar) / ((N_i - 1) * n_states)
                S_w_inv += inv_i.inverse()

        if self.harmonic_lda:
            S_w = S_w_inv.inverse()

        # Compute S_b from total scatter matrix
        S_b = S_t - S_w

        # Regularize S_w
        S_w = S_w + self.sw_reg * torch.diag(
            torch.Tensor().new_ones((d)).to(device)
        )

        if save_params:
            self.S_b = S_b
            self.S_w = S_w

        return S_b, S_w

    def compute_eigenvalues(self, S_b = None , S_w = None, save_params=True):
        """
        Compute LDA eigenvalues from S_w and S_b. 

        Parameters
        ----------
        S_w, S_b : arrays of shape (n_in,n_in)
            Between and within scatter matrices
        save_params: bool, optional
            Whether to store parameters in model

        Returns
        -------
        eigvals : torch.Tensor
            LDA eigenvalues (n_states-1)
        eigvecs : torch.Tensor
            LDA eigenvectors (n_feature,n_states-1)
        """
        if S_w is None: 
            S_w = self.S_w
        if S_b is None: 
            S_b = self.S_b

        # -- Generalized eigenvalue problem: S_b * v_i = lambda_i * Sw * v_i --

        # (1) use cholesky decomposition for S_w
        L = torch.linalg.cholesky(S_w, upper=False)

        # (2) define new matrix using cholesky decomposition
        L_t = torch.t(L)
        L_ti = torch.inverse(L_t)
        L_i = torch.inverse(L)
        S_new = torch.matmul(torch.matmul(L_i, S_b), L_ti)

        # (3) find eigenvalues and vectors of S_new
        eigvals, eigvecs = torch.linalg.eigh(S_new, UPLO='L')
        # sort
        eigvals, indices = torch.sort(eigvals, 0, descending=True)
        eigvecs = eigvecs[:, indices]

        # (4) return to original eigenvectors
        eigvecs = torch.matmul(L_ti, eigvecs)

        # normalize them
        for i in range(eigvecs.shape[1]):  # TODO maybe change in sum along axis?
            norm = eigvecs[:, i].pow(2).sum().sqrt()
            eigvecs[:, i].div_(norm)
        # set the first component positive
        eigvecs.mul_(torch.sign(eigvecs[0, :]).unsqueeze(0).expand_as(eigvecs))

        # keep only C-1 eigvals and eigvecs
        eigvals = eigvals[: self.n_states - 1]
        eigvecs = eigvecs[:, : self.n_states - 1]
        if save_params:
            self.evals = eigvals
            self.evecs = eigvecs

        return eigvals, eigvecs
    
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
    n_in = 5
    n_states = 3
    X = torch.rand(100,n_in)*100
    y = torch.randint(n_states,(100,1)).squeeze(1)

    lda = LDA(n_in,n_states)
    S_w, S_b = lda.compute_scatter_matrices(X,y)
    print(S_w.shape,S_b.shape)
    evals,evecs = lda.compute_eigenvalues()
    print(evals.shape,evecs.shape)
    s = lda(X)
    print(s.shape)
    assert (s.ndim == 2) and (s.shape[1]==n_states-1)

if __name__ == "__main__":
    test_lda()