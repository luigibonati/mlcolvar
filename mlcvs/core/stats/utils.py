import torch
from torch import Tensor
from typing import Optional

def generalized_eigh(A: Tensor, B: Tensor) -> tuple:
    """A workaround to solve a real symmetric generalized eigenvalue problem :math:`Av = \lambda Bv` using the eigenvalue decomposition of :math:`B^{-1/2}AB^{-1/2}`. This method is not numerically efficient.

    Parameters
    ----------
    A : Tensor

    B : Tensor

    Returns
    -------
    tuple
        Eigenvalues and eigenvectors of the generalized eigenvalue problem.
    """
    Lambda, Q = torch.linalg.eigh(B)
    rsqrt_Lambda = torch.diag(Lambda.rsqrt())
    rsqrt_B = Q@rsqrt_Lambda
    _A = 0.5*(rsqrt_B.T@(A@rsqrt_B) + rsqrt_B.T@((A.T)@rsqrt_B)) #Force Symmetrization
    values, _tmp_vecs = torch.linalg.eigh(_A) 
    vectors = rsqrt_B@_tmp_vecs
    return values, vectors

def spd_norm(vecs: Tensor, spd_matrix: Tensor) -> Tensor:
    """Compute the norm of a set of vectors with respect to a symmetric positive definite matrix.

    Parameters
    ----------
    vecs : Tensor
        Two dimensional tensor whose columns are the vectors whose norm is to be computed.
    spd_matrix : Tensor
        Symmetric positive matrix. Warning: this matrix is not checked for symmetry or positive definiteness.

    Returns
    -------
    Tensor
        One dimensional tensor whose i-th element is the norm of the i-th column of vecs with respect to spd_matrix.
    """    
    _v = torch.mm(spd_matrix, vecs)
    _v_T = torch.mm(spd_matrix.T, vecs)
    return torch.sqrt(0.5*torch.linalg.vecdot(vecs, _v + _v_T, dim = 0).real)

def reduced_rank_eig(
    input_covariance: Tensor,
    lagged_covariance: Tensor, #C_{0t}
    tikhonov_reg: float,
    rank: Optional[int] = None,
    ) -> tuple:
    """Reduced rank regression algorithm, as described in [1]_.

    Parameters
    ----------
    input_covariance : Tensor
        
    lagged_covariance : Tensor
        
    rank : Optional[int], optional
        Rank of the final estimator, by default None

    Returns
    -------
    tuple
        A tuple containing the eigenvalues and eigenvectors of the Koopman operator.
    
    References
    ----------
    .. [1] V. Kostic, P. Novelli, A. Maurer, C. Ciliberto, L. Rosasco, and M. Pontil, "Learning Dynamical Systems via Koopman Operator Regression in Reproducing Kernel Hilbert Spaces" (2022).
    """    
    n = input_covariance.shape[0]
    reg_input_covariance = input_covariance + tikhonov_reg*torch.eye(n, dtype=input_covariance.dtype, device=input_covariance.device)

    _crcov = torch.mm(lagged_covariance, lagged_covariance.T)
    _, _vectors = generalized_eigh(_crcov, reg_input_covariance) 
    
    _norms = spd_norm(_vectors, reg_input_covariance)
    vectors = _vectors*(1/_norms)

    if rank is not None:
        _, idxs = torch.topk(vectors.values, rank)
        U = vectors[:, idxs]
    else:
        U = vectors
    
    #U@(U.T)@Tw = v w -> (U.T)@T@Uq = vq and w = Uq 
    values, Q = torch.linalg.eig((U.T)@(lagged_covariance@U))
    return values, U@Q


def cholesky_eigh(A, B, reg_B = 1e-6, n_eig = None ):
    """
    -- Generalized eigenvalue problem: A * v_i = lambda_i * B * v_i --

    First apply cholesky decomposition to B and then solve the generalized eigvalue problem.

    Notes
    -----
    The eigenvecs object which is returned is a matrix whose column eigvecs[:,i] is the eigenvector associated to eigvals[i]"""

    # check that both matrices are symmetric
    if not torch.allclose( A.transpose(0, 1), A) : 
        raise ValueError('The matrices need to be symmetric to solve the generalized eigenvalue problem via cholesky decomposition. A >> ', A )
    if not (torch.allclose( B.transpose(0, 1), B) ): 
        raise ValueError('The matrices need to be symmetric to solve the generalized eigenvalue problem via cholesky decomposition. A >> ', B )

    # (0) regularize B matrix before cholesky
    B = B + reg_B*torch.eye(B.shape[0]).to(B.device)

    # (1) use cholesky decomposition for B
    L = torch.linalg.cholesky(B, upper=False)

    # (2) define new matrix using cholesky decomposition
    L_t = torch.t(L)
    L_ti = torch.inverse(L_t)
    L_i = torch.inverse(L)
    A_new = torch.matmul(torch.matmul(L_i, A), L_ti)

    # (3) find eigenvalues and vectors of A_new
    eigvals, eigvecs = torch.linalg.eigh(A_new, UPLO='L')
    # sort
    eigvals, indices = torch.sort(eigvals, 0, descending=True)
    eigvecs = eigvecs[:, indices]

    # (4) return to original eigenvectors
    eigvecs = torch.matmul(L_ti, eigvecs)

    # (5) normalize them
    eigvecs = torch.nn.functional.normalize(eigvecs, dim=0)
    # set the first component positive
    eigvecs = eigvecs.mul(torch.sign(eigvecs[0, :]).unsqueeze(0).expand_as(eigvecs))

    # (6) keep only first n_eig eigvals and eigvecs
    if n_eig is not None:
        eigvals = eigvals[: n_eig]
        eigvecs = eigvecs[:, : n_eig]

    return eigvals, eigvecs

""" TODO implement also the non-symmetric version?
#Compute the pseudoinverse (Moore-Penrose inverse) of C_0. if det(C_0) != 0 then the usual inverse is computed
            C_new = torch.matmul(torch.pinverse(C_0),C_lag)
            #find eigenvalues and vectors of C_new
            #NOTE: torch.linalg.eig() returns complex tensors of dtype cfloat or cdouble
            #          rather than real tensors mimicking complex tensors. 
            #          For future developments it would be necessary to take either only the real part
            #          or only the complex part or only the magnitude of the complex eigenvectors and eigenvalues 
            eigvals, eigvecs = torch.linalg.eig(C_new, UPLO='L')
"""

def correlation_matrix(x,y,w=None,symmetrize=True):
    """Compute the correlation matrix between x and y with weights w

    Parameters
    ----------
    x : torch.Tensor
        first array
    y : torch.Tensor
        second array
    w : torch.Tensor, optional
        weights, by default None
    symmetrize: bool, optional
        whether to return 0.5 * (C + C^T), by default True

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
        
    if symmetrize:
        corr = 0.5*(corr + corr.T)

    return corr

def compute_average(x, w = None):
    """Compute (weighted) average on a batch

    Parameters
    ----------
    x : torch.Tensor
        Input data of shape
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
        ave = torch.mean(x.T,1,keepdim=False).T
    
    return ave
