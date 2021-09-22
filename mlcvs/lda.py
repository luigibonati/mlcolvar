"""Linear discriminant analysis-based CVs."""

from .io import colvar_to_pandas
import torch
import numpy as np

class LinearDiscriminant:
    """
    Linear Discriminant CV

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

        # Initialize device and dtype
        self.dtype_ = torch.float32
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        S_w = torch.Tensor().new_zeros((d, d), device = self.device_, dtype = self.dtype_)    
        S_w_inv = torch.Tensor().new_zeros((d, d), device = self.device_, dtype = self.dtype_)
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
        S_w = S_w + self.lambda_ * torch.diag(torch.Tensor().new_ones((d), device = self.device_, dtype = self.dtype_))

        # -- Generalized eigenvalue problem: S_b * v_i = lambda_i * Sw * v_i --

        # (1) use cholesky decomposition for S_w
        L = torch.cholesky(S_w,upper=False)

        # (2) define new matrix using cholesky decomposition
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
        for i in range(eigvecs.shape[1]): # TODO maybe change in sum along axis?
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
            X = torch.tensor(X,dtype=self.dtype_,device=self.device_)
        if type(y) != torch.Tensor:
            y = torch.tensor(y,device=self.device_) #class labels are integers
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
            X = torch.tensor(X,dtype=self.dtype_,device=self.device_)
            
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
        if self.features_names_ is not None:
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
        out = "" 
        for i in range(len(self.evals_)):
            if len(self.evals_)==1:
                #print('lda: COMBINE ARG=', end='')
                out += 'lda: COMBINE ARG='
            else:
                #print(f'lda{i+1}: COMBINE ARG=', end='')
                out += f'lda{i+1}: COMBINE ARG='
            for j in range(self.d_):
                if self.features_names_ is None:
                    #print(f'x{j},',end='')
                    out += f'x{j},'
                else:
                    #print(f'{self.features_names_[j]},',end='')
                    out += f'{self.features_names_[j]},'
            #print("\b COEFFICIENTS=",end='') 
            out = out [:-1]
            out += " COEFFICIENTS="
            for j in range(self.d_):
                #print(np.round(self.evecs_[j,i].cpu().numpy(),6),end=',')
                out += str(np.round(self.evecs_[j,i].cpu().numpy(),6))+','
            #print('\b PERIODIC=NO')
            out = out [:-1]
            out += ' PERIODIC=NO'
        return out 


#if __name__ == "__main__":


