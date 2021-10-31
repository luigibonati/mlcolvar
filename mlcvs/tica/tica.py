"""Time-lagged independent component analysis-based CV"""

__all__ = ["TICA"] # TODO update?

import torch
import numpy as np
from bisect import bisect_left

def closest_idx_torch(array, value):
        '''
        Find index of the element of 'array' which is closest to 'value'.
        The array is first converted to a np.array in case of a tensor.
        Note: it does always round to the lowest one.

        Parameters:
            array (tensor/np.array)
            value (float)

        Returns:
            pos (int): index of the closest value in array
        '''
        if type(array) is np.ndarray:
            pos = bisect_left(array, value)
        else:
            pos = bisect_left(array.numpy(), value)
        if pos == 0:
            return 0
        elif pos == len(array):
            return -1
        else:
            return pos-1

def look_for_configurations(x,t,lag):
    '''
    Searches for all the pairs which are distant 'lag' in time, and returns the weights associated to lag=lag as well as the weights for lag=0.

    Parameters:
        x (tensor): array whose columns are the descriptors and rows the time evolution
        time (tensor): array with the simulation time
        lag (float): lag-time

    Returns:
        x_t (tensor): array of descriptors at time t
        x_lag (tensor): array of descriptors at time t+lag
        w_t (tensor): weights at time t
        w_lag (tensor): weights at time t+lag
    '''
    #define lists
    x_t = []
    x_lag = []
    w_t = []
    w_lag = []
    #find maximum time idx
    idx_end = closest_idx_torch(t,t[-1]-lag)
    start_j = 0
    #loop over time array and find pairs which are far away by lag
    for i in range(idx_end):
        stop_condition = lag+t[i+1]
        n_j = 0
        for j in range(i,len(t)):
            if ( t[j] < stop_condition ) and (t[j+1]>t[i]+lag):
                x_t.append(x[i])
                x_lag.append(x[j])
                deltaTau=min(t[i+1]+lag,t[j+1]) - max(t[i]+lag,t[j])
                w_lag.append(deltaTau)
                if n_j == 0: #assign j as the starting point for the next loop
                    start_j = j
                n_j +=1
            elif t[j] > stop_condition:
                break
        for k in range(n_j):
            w_t.append((t[i+1]-t[i])/float(n_j))

    x_t = torch.stack(x_t) if type(x) == torch.Tensor else torch.Tensor(x_t)
    x_lag = torch.stack(x_lag) if type(x) == torch.Tensor else torch.Tensor(x_lag)
    
    w_t = torch.Tensor(w_t)
    w_lag = torch.Tensor(w_lag)

    return x_t,x_lag,w_t,w_lag

def divide_in_batches(list_of_tensors,batch_size):
    '''
    Takes as input a list of torch.tensors and returns batches of length batch_size
    
    Parameters:
        list_of_tensors (list of tensors)
        batch_size 
        
    Returns:
        list of batches
    '''
    batch=[]
    for x in list_of_tensors:
        batch.append(torch.split(x,batch_size,dim=0))
    
    batch=list(zip(*batch))
    
    return batch

def split_train_valid(x,t,n_train,n_valid,every,last_valid=False):

    print("[SPLIT DATASET]")
    x_train = x[:n_train*every:every]
    t_train = t[:n_train*every:every]
    print("- Training points =",x_train.shape[0])

    if last_valid:
        x_valid = x[-n_valid*every::every]
        t_valid = t[-n_valid*every::every]
    else:
        x_valid = x[n_train*every:(n_train+n_valid)*every:every]
        t_valid = t[n_train*every:(n_train+n_valid)*every:every]
    
    train = [x_train,t_train]
    valid = [x_valid,t_valid]
    
    return train,valid
    
def create_tica_dataset(x,t,lag_time,n_train,n_valid,every=1,batch_tr=-1,last_valid=False):
    '''Returns [x_t, x_lag, w_t, w_lag]''' 
    
    train,valid = split_train_valid(x,t,n_train,n_valid,every,last_valid=last_valid)
    x_train,t_train = train
    x_valid,t_valid = valid

    # TRAINING SET --> DATASET WITH PAIRS (x_t,x_t+lag)
    print("- Search (x_t,x_t+lag) with lag time =",lag_time)
    train_configs = look_for_configurations(x_train,t_train,lag=lag_time)
    print("- Found n_pairs =",train_configs[0].shape[0])

    # create batches
    if batch_tr == -1:
        batch_tr = len(train_configs[0])
    train_batches = divide_in_batches(train_configs,batch_tr)
    print("- Batch size =",batch_tr)
    print("- N batches =",len(train_batches))

    # VALID CONFIGS
    valid_configs = look_for_configurations(x_valid,t_valid,lag=lag_time)

    return train_batches,valid_configs

class TICA:
    """ Time-lagged independent component analysis base class.
    """

    def __init__(self, device = 'auto'):
        """Initialize TICA object.

        Parameters
        ----------
        device : str, optional
            device, by default 'auto'

        """

        # initialize attributes
        self.evals_ = None
        self.evecs_ = None
        self.n_features = None 

        # Regularization
        self.reg_cholesky = 0 

        # Initialize device
        if device == "auto":
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_ = device

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
            ave = torch.average(x)
        
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

    def compute_correlation_matrix(self,x,y,w=None,symmetrize=True):
        """Compute the correlation matrix between x and y with weights w

        Parameters
        ----------
        x : torch.Tensor
            first array
        y : torch.Tensor
            second array
        w : torch.Tensor
            weights, by default None
        symmetrize : bool, optional
            Enforce symmetrization, by default True

        Returns
        -------
        torch.Tensor
            correlation matrix

        """
        # TODO Add assert on shapes

        d = x.shape[1]

        if w is None: #TODO simplify it in the unbiased case?
            w = torch.ones(x.shape[0])
            
        #define arrays
        corr = torch.zeros((d,d))
        norm = torch.sum(w)

        #compute correlation matrix
        corr = torch.einsum('ij, ik, i -> jk', x, y, w )
        corr /= norm
            
        if symmetrize:
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
        #cholesky decomposition
        if self.reg_cholesky is not None:
            L = torch.cholesky(C_0+self.reg_cholesky*torch.eye(C_0.shape[0]),upper=False)
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
