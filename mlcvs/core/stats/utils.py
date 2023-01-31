import torch

def cholesky_eigh(A, B, reg_B = 1e-6, n_eig = None ):
    """
    -- Generalized eigenvalue problem: A * v_i = lambda_i * B * v_i --

    First apply cholesky decomposition to B and then solve the generalized eigvalue problem.

    Notes
    -----
    The eigenvecs object which is returned is a matrix whose column eigvecs[:,i] is the eigenvector associated to eigvals[i]"""

    # assert 
    if ( (A.transpose(0, 1) != A).any() or (B.transpose(0, 1) != B).any() ): 
        raise ValueError('The matrices need to be symmetric to solve the generalized eigenvalue problem via cholesky decomposition.')

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
