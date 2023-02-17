import torch

__all__ = ['reduce_eigenvalues']

def reduce_eigenvalues(evals : torch.Tensor, options : dict = { 'mode':'sum', 'n_eig': 0 }):
        """
        Calculate a monotonic function of the eigenvalues, by default the sum.

        Parameters
        ----------
        eval : torch.tensor
            Eigenvalues
        mode : str
            function of the eigenvalues to optimize (see notes)
        n_eig: int, optional
            number of eigenvalues to include in the loss (default: 0 --> all). in case of single and single2 is used to specify which eigenvalue to use.

        Notes
        -----
        The following mode are implemented:
            - sum     : sum_i (lambda_i)
            - sum2    : sum_i (lambda_i)**2
            - gap     : (lambda_1-lambda_2)
            - its     : sum_i (1/log(lambda_i))
            - single  : (lambda_i)
            - single2 : (lambda_i)**2

        Returns
        -------
        loss : torch.tensor (scalar)
            score
        """

        # parse args
        mode = options['mode'] if 'mode' in options else 'sum'
        n_eig = options['n_eig'] if 'n_eig' in options else 0

        #check if n_eig is given and
        if (n_eig>0) & (len(evals) < n_eig):
            raise ValueError("n_eig must be lower than the number of eigenvalues.")
        elif (n_eig==0):
            if ( (mode == 'single') | (mode == 'single2')):
                raise ValueError("n_eig must be specified when using single or single2.")
            else:
                n_eig = len(evals)

        loss = None
        
        if   mode == 'sum':
            loss =  torch.sum(evals[:n_eig])
        elif mode == 'sum2':
            g_lambda =  torch.pow(evals,2)
            loss = torch.sum(g_lambda[:n_eig])
        elif mode == 'gap':
            loss =  (evals[0] -evals[1])
        elif mode == 'its':
            g_lambda = 1 / torch.log(evals)
            loss = torch.sum(g_lambda[:n_eig])
        elif mode == 'single':
            loss =  evals[n_eig-1]
        elif mode == 'single2':
            loss = torch.pow(evals[n_eig-1],2)
        else:
            raise ValueError(f"unknown mode : {mode}. options: 'sum','sum2','gap','single','its'.")

        return loss