import torch

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
    for i in range(eigvecs.shape[1]): 
        norm = eigvecs[:, i].pow(2).sum().sqrt()
        eigvecs[:, i].div_(norm)
    # set the first component positive
    eigvecs.mul_(torch.sign(eigvecs[0, :]).unsqueeze(0).expand_as(eigvecs))

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
