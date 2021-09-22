"""Linear discriminant analysis based CVs."""

from mlcvs.io import colvar_to_pandas
import torch
import numpy as np

class LinearDiscriminantAnalysis:
    """
    Linear Discriminant Analysis

    Attributes
    ----------
    d_ : int
        Number of classes
    evals_ : torch.Tensor
        LDA eigenvalues
    evecs_ : torch.Tensor
        LDA eignvectors
    S_b_ : torch.Tensor
        Between scatter matrix
    S_w_ : torch.Tensor
        Within scatter matrix
        
    Methods
    -------
    fit(x,label)
        Fit LDA given data and classes
    transform(x)
        Project data to maximize class separation
    fit_transform(x,label)
        Fit LDA and project data 
    get_params()
        Return saved parameters
    plumed_input()
        Generate PLUMED input file
    """
    def __init__(self):
        # initialize attributes
        self.evals_ = None
        self.evecs_ = None
        self.S_b_ = None
        self.S_w_ = None
        self.d_ = None
        self.features_names_ = None
        self.lambda_ = 1e-6

    def LDA(self, H, label, save_params = True):
        """
        Internal method which performs LDA and saves parameters.
        
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
        
        #sizes
        N, d = H.shape
        
        #classes
        categ = torch.unique(label)
        n_categ = len(categ)
        self.d_ = d
        
        # Mean centered observations for entire population
        H_bar = H - torch.mean(H, 0, True)
        #Total scatter matrix (cov matrix over all observations)
        S_t = H_bar.t().matmul(H_bar) / (N - 1)
        #Define within scatter matrix and compute it
        S_w = torch.Tensor().new_zeros((d, d), device = device, dtype = dtype)    
        S_w_inv = torch.Tensor().new_zeros((d, d), device = device, dtype = dtype)
        #Loop over classes to compute means and covs
        for i in categ:
            #check which elements belong to class i
            H_i = H[torch.nonzero(label == i).view(-1)]
            # compute mean centered obs of class i
            H_i_bar = H_i - torch.mean(H_i, 0, True)
            # count number of elements
            N_i = H_i.shape[0]
            if N_i == 0:
                continue
            S_w += H_i_bar.t().matmul(H_i_bar) / ((N_i - 1) * n_categ)       

        S_b = S_t - S_w

        # Regularize S_w
        S_w = S_w + self.lambda_ * torch.diag(torch.Tensor().new_ones((d), device = device, dtype = dtype))

        # -- Generalized eigenvalue problem: S_b * v_i = lambda_i * Sw * v_i --

        # (1) use cholesky decomposition for S_w
        L = torch.cholesky(S_w,upper=False)

        # (2) define new matrix using cholesky decomposition and 
        L_t = torch.t(L)
        L_ti = torch.inverse(L_t)
        L_i = torch.inverse(L)
        S_new = torch.matmul(torch.matmul(L_i,S_b),L_ti)

        #(3) find eigenvalues and vectors of S_new
        eigvals, eigvecs = torch.symeig(S_new,eigenvectors=True)
        #sort
        eigvals, indices = torch.sort(eigvals, 0, descending=True)
        eigvecs = eigvecs[:,indices]
        
        #(4) return to original eigenvectors
        eigvecs = torch.matmul(L_ti,eigvecs)

        #normalize them
        for i in range(eigvecs.shape[1]): # maybe change in sum along axis?
            norm=eigvecs[:,i].pow(2).sum().sqrt()
            eigvecs[:,i].div_(norm)
        #set the first component positive
        eigvecs.mul_( torch.sign(eigvecs[0,:]).unsqueeze(0).expand_as(eigvecs) )
        
        #keep only C-1 eigvals and eigvecs
        eigvals = eigvals[:n_categ-1]
        eigvecs = eigvecs[:,:n_categ-1]#.reshape(eigvecs.shape[0],n_categ-1)

        if save_params:
            self.evals_ = eigvals
            self.evecs_ = eigvecs
            self.S_b_ = S_b
            self.S_w_ = S_w
        
        return eigvals

    #TODO REMOVE THIS UTILITY
    def np_to_torch(self,x,dtype=None):
        """
        Utility which converts numpy array to torch.Tensor
        
        Parameters
        ----------
        x : Numpy array
            Input
        Returns
        -------
        y : array-like of shape (n_samples, n_classes-1)
            LDA projections.
        """
        return torch.tensor(x,dtype=dtype)

    def fit(self,X,y):
        """
        Fit LDA given data and classes.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) 
            Classes labels. 
        """
        
        if type(X) != torch.Tensor:
            X = self.np_to_torch(X,dtype=torch.float32)
        if type(y) != torch.Tensor:
            y = self.np_to_torch(y)
        _ = self.LDA(X,y)
    
    def transform(self,X):
        """
        Project data to maximize class separation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features)
            Inference data.
               
        Returns
        -------
        s : array-like of shape (n_samples, n_classes-1)
            LDA projections.
        """
        if type(X) != torch.Tensor:
            X = self.np_to_torch(X,dtype=torch.float32)
            
        s = torch.matmul(X,self.evecs_)
    
        return s
        
    def fit_transform(self,X,y):
        """
        Fit LDA and project data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) 
            Classes labels. 
        
        Returns
        -------
        s : array-like of shape (n_samples, n_classes-1)
            LDA projections.
        """
        
        self.fit(X,y)
        return self.transform(X)

    def get_params(self):
        """
        Return saved parameters.
        
        Returns
        -------
        out : dictionary
            Parameters
        """
        out = dict()
        if self.feature_names_ is not None:
            out['features'] = self.features_names_
        out['eigenvalues']=self.evals_
        out['eigenvectors']=self.evecs_
        out['S_between']=self.S_b_
        out['S_within']=self.S_w_
        return out
    
    def set_sw_regularization(self,reg):
        """
        Set regularization for within-scatter matrix.

        .. math:: S_w = S_w + \mathtt{reg}\ \mathbf{1}. 

        Parameters
        ----------
        reg : float
            Regularization value.
        
        """
        self.lambda_ = reg

    def set_features_names(self,names):
        """
        Set names of the features (useful for creating PLUMED input file)

        Parameters
        ----------
        reg : array-like of shape (n_descriptors)
            Features names
        
        """
        self.features_names_ = names

    def plumed_input(self):
        """
        Generate PLUMED input file
        
        Returns
        -------
        out : string
            PLUMED input file
        """
        for i in range(len(self.evals_)):
            if len(self.evals_)==1:
                print('lda: COMBINE ARG=', end='')
            else:
                print(f'lda{i+1}: COMBINE ARG=', end='')
            for j in range(self.d_):
                if self.features_names_ is None:
                    print(f'x{j},',end='')
                else:
                    print(f'{self.features_names_[j]},',end='')
            print("\b COEFFICIENTS=",end='') 
            for j in range(self.d_):
                print(np.round(self.evecs_[j,i].cpu().numpy(),6),end=',')
            print('\b PERIODIC=NO')


if __name__ == "__main__":

    # Set device and dtype
    dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #torch.device('cpu') #

    # Load data
    dataA = colvar_to_pandas(folder='mlcvs/data/2d-basins/',filename='COLVAR_stateA')
    dataB = colvar_to_pandas(folder='mlcvs/data/2d-basins/',filename='COLVAR_stateB')

    # Create input dataset
    xA = dataA.filter(regex='p.*').values
    xB = dataB.filter(regex='p.*').values
    names = dataA.filter(regex='p.*').columns.values

    # Create labels
    yA = np.zeros(len(dataA))
    yB = np.ones(len(dataB))

    x = np.concatenate([xA,xB],axis=0)
    Y = np.concatenate([yA,yB],axis=0)

    # Transform to Pytorch Tensors
    x = torch.tensor(x,dtype=dtype,device=device)
    Y = torch.tensor(Y,dtype=dtype,device=device)
    
    # Perform LDA
    lda = LinearDiscriminantAnalysis()
    lda.set_features_names(names)
    lda.fit(x,Y)

    lda.plumed_input()


