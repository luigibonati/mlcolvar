import torch

class LDA:
    """
    Fisher's discriminant class.

    Attributes
    ----------
    n_features : int
        Number of features
    n_classes : int
        Number of classes
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

    Methods
    -------
    compute_LDA(H,y,save_params):
        Perform LDA
    """

    def __init__(self, harmonic_lda = False, device = 'auto'):
        """
        Create a LDA object

        Parameters
        ----------
        harmonic_lda : bool
            Harmonic variant of LDA
        """

        # initialize attributes
        self.harmonic_lda = harmonic_lda
        self.evals_ = None
        self.evecs_ = None
        self.S_b_ = None
        self.S_w_ = None
        self.n_features = None 
        self.n_classes = None

        # Regularization
        self.sw_reg = 1e-6

        # Initialize device
        if device == "auto":
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_ = device

    def compute_LDA(self, H, label, save_params=True):
        """
        Performs LDA and saves parameters.

        Parameters
        ----------
        H : array-like of shape (n_samples, n_features)
            Training data.
        label : array-like of shape (n_samples,)
            Classes labels.
        save_params: bool, optional
            Whether to store parameters in model

        Returns
        -------
        evals : array of shape (n_classes-1)
            LDA eigenvalues.
        """

        # sizes
        N, d = H.shape
        self.n_features = d

        # classes
        classes = torch.unique(label)
        n_classes = len(classes)
        self.n_classes = n_classes

        # Mean centered observations for entire population
        H_bar = H - torch.mean(H, 0, True)
        # Total scatter matrix (cov matrix over all observations)
        S_t = H_bar.t().matmul(H_bar) / (N - 1)
        # Define within scatter matrix and compute it
        S_w = torch.Tensor().new_zeros((d, d)).to(device=self.device_)
        if self.harmonic_lda:
            S_w_inv = torch.Tensor().new_zeros((d, d)).to(device=self.device_)
        # Loop over classes to compute means and covs
        for i in classes:
            # check which elements belong to class i
            H_i = H[torch.nonzero(label == i).view(-1)]
            # compute mean centered obs of class i
            H_i_bar = H_i - torch.mean(H_i, 0, True)
            # count number of elements
            N_i = H_i.shape[0]
            if N_i == 0:
                continue

            # LDA
            S_w += H_i_bar.t().matmul(H_i_bar) / ((N_i - 1) * n_classes)

            # HLDA
            if self.harmonic_lda:
                inv_i = H_i_bar.t().matmul(H_i_bar) / ((N_i - 1) * n_classes)
                S_w_inv += inv_i.inverse()

        if self.harmonic_lda:
            S_w = S_w_inv.inverse()

        # Compute S_b from total scatter matrix
        S_b = S_t - S_w

        # Regularize S_w
        S_w = S_w + self.sw_reg * torch.diag(
            torch.Tensor().new_ones((d)).to(device=self.device_)
        )

        # -- Generalized eigenvalue problem: S_b * v_i = lambda_i * Sw * v_i --

        # (1) use cholesky decomposition for S_w
        L = torch.cholesky(S_w, upper=False)

        # (2) define new matrix using cholesky decomposition
        L_t = torch.t(L)
        L_ti = torch.inverse(L_t)
        L_i = torch.inverse(L)
        S_new = torch.matmul(torch.matmul(L_i, S_b), L_ti)

        # (3) find eigenvalues and vectors of S_new
        eigvals, eigvecs = torch.symeig(S_new, eigenvectors=True)
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
        eigvals = eigvals[: n_classes - 1]
        eigvecs = eigvecs[:, : n_classes - 1]
        if save_params:
            self.evals_ = eigvals
            self.evecs_ = eigvecs
            self.S_b_ = S_b
            self.S_w_ = S_w

        return eigvals, eigvecs
