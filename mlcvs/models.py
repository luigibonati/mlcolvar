"""Collective variables base models."""

import torch
import numpy as np
from . import optim

class LinearCV:
    """
    Linear CV base class.

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
    set_features_names(names)
        Set features names
    plumed_input()
        Generate PLUMED input file
    """

    def __init__(self, n_features, device = 'auto', dtype = torch.float32 ):
        super().__init__()

        # Device and dtype
        self.dtype_ = dtype
        if device == 'auto':
            self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device_ = device

        # Initialize parameters
        self.n_features = n_features
        self.w = torch.eye(n_features, dtype = self.dtype_, device = self.device_)
        self.b = torch.zeros(n_features, dtype = self.dtype_, device = self.device_)
        
        # Generic attributes
        self.name_ = 'LinearCV'
        self.features_names = ['x'+str(i) for i in range(n_features)]
                                                    
        # Flags 

    def __call__(self, X):
        """
        Alias for transform.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features)
            Inference data.
               
        Returns
        -------
        s : array-like of shape (n_samples, n_classes-1)
            Linear projection along `self.weights_`.

        See Also
        --------
        transform : project data along linear combination
        """
        return self.transform(X)

    def fit(self):
        """
        Fit estimator (abstract method). 
        """
        pass
    
    def transform(self,X):
        """
        Project data along linear components.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features)
            Inference data.
               
        Returns
        -------
        s : array-like of shape (n_samples, n_classes-1)
            Linear projection of inputs. 
        """
        if type(X) != torch.Tensor:
            X = torch.tensor(X,dtype=self.dtype_,device=self.device_)

        s = torch.matmul(X-self.b,self.w)
    
        return s
        
    def fit_transform(self,X):
        """
        Call fit and then transform (abstract method).
        
        Parameters
        ----------
        
        Returns
        -------

        """
        pass

    def set_weights(self, w):
        """
        Set weights of linear combination.
        
        Parameters
        ----------
        w : torch.tensor
            weights

        """
        self.w = w.to(self.device_) 

    def set_offset(self, b):
        """
        Set linear offset
        
        Parameters
        ----------
        b : torch.tensor
            offset

        """
        self.b = b.to(self.device_) 

    def get_params(self):
        """
        Return saved parameters.
        
        Returns
        -------
        out : namedtuple
            Parameters
        """
        return vars(self)
    
    def set_params(self,dict_params):
        """
        Set parameters.

        Parameters
        ----------
        dict_params : dictionary
            Parameters
        
        """

        for key in dict_params.keys():
            if hasattr(self,key):
                setattr(self,key,dict_params[key])
            else:
                raise AttributeError(f'{self.__class__.__name__} has no attribute {key}.')

    def plumed_input(self):
        """
        Generate PLUMED input file
        
        Returns
        -------
        out : string
            PLUMED input file
        """
        # count number of output components
        weights = self.w.cpu().numpy()
        offset = self.b.cpu().numpy()
        n_cv = 1 if weights.ndim == 1 else weights.shape[1]

        out = "" 
        for i in range(n_cv):
            if n_cv==1:
                out += f'{self.name_}: COMBINE ARG='
            else:
                out += f'{self.name_}{i+1}: COMBINE ARG='
            for j in range(self.n_features):
                    out += f'{self.features_names[j]},'
            out = out [:-1]

            out += " COEFFICIENTS="
            for j in range(self.n_features):
                out += str(np.round(weights[j,i],6))+','
            out = out [:-1]
            if not np.all(offset == 0):
                out += " PARAMETERS="
                for j in range(self.n_features):
                    out += str(np.round(offset[j,i],6))+','
            out = out [:-1]

            out += ' PERIODIC=NO'
        return out 

class NeuralNetworkCV(torch.nn.Module):
    """
    Neural Network CV base class

    Attributes
    ----------
    nn : nn.Module
        Neural network module
    n_input: int
        No. of inputs
    dtype_: torch.dtype
        Type of tensors
    device_: torch.device
        Device (cpu or cuda)
    opt_: torch.optimizer
        Optimizer
    earlystopping_: optim.EarlyStopping
        EarlyStopping scheduler
    normIn: bool
        Normalize inputs
    normOut: bool
        Normalize outputs

    Methods
    -------
    forward(x)
        Compute model output
    forward_nn(x)
        Compute nn module output
    set_device(device)
        Set torch.device
    set_optimizer(opt)
        Save optimizer
    set_earlystopping(patience,min_delta, ...)
        Enable EarlyStopping
    standardize_inputs(x)
        Standardize inputs over dataset
    standardize_outputs(x)
        Standardize outputs over dataset
    get_params()
        Return attached parameters
    print_info()
        Display information about model
    print_log()
        Utility function for training log

    Examples
    --------
    Create a neural-network with 20 inputs, one hidden layer with 10 nodes and 5 outputs
    >>> net = NeuralNetworkCV(layers=[20,10,5], activation = 'relu')
    >>> net
    NeuralNetworkCV(
      (nn): Sequential(
        (0): Linear(in_features=20, out_features=10, bias=True)
        (1): ReLU(inplace=True)
        (2): Linear(in_features=10, out_features=5, bias=True)
      )
    )
    """

    def __init__(self, layers , activation='relu', device = 'auto', dtype = torch.float32 ):
        '''
        Define a neural network module given the list of layers.
        
        Parameters
        ----------
        layers : list
            Number of neurons per layer
        activation : string
            Activation function (relu, tanh, elu, linear) 

        '''
        super().__init__()

        #get activation function
        activ=None
        if   activation == 'relu':
            activ=torch.nn.ReLU(True)
        elif activation == 'elu':
            activ=torch.nn.ELU(True)
        elif activation == 'tanh':
            activ=torch.nn.Tanh()
        elif activation == 'linear':
            print('WARNING: linear activation selected')
        else:
            raise ValueError("Unknown activation. options: 'relu','elu','tanh','linear'. ")

        #Create architecture
        modules=[]
        for i in range( len(layers)-1 ):
            if( i<len(layers)-2 ):
                modules.append( torch.nn.Linear(layers[i], layers[i+1]) )
                if activ is not None:
                    modules.append( activ )
            else:
                modules.append( torch.nn.Linear(layers[i], layers[i+1]) )

        # Device and dtype
        self.dtype_ = dtype
        if device == 'auto':
            self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device_ = device

        # Initialize parameters
        self.nn = torch.nn.Sequential(*modules)
        self.n_features = layers[0]
        self.n_hidden = layers[-1]

        # Linear projection output
        weight = torch.eye(self.n_hidden, dtype = self.dtype_, device = self.device_)
        offset = torch.zeros(self.n_hidden, dtype = self.dtype_, device = self.device_)
        self.register_buffer('w', weight)
        self.register_buffer('b', offset)
        
        # Flags 
        self.normIn = False
        self.normOut = False
        self.output_hidden = False
        
        # Optimizer
        self.opt_ = None
        self.earlystopping_ = None

        # Generic attributes
        self.name_ = 'NN_CV'
        self.features_names = ['x'+str(i) for i in range(self.n_features)]
        
    # Forward pass
    def forward_nn(self, x: torch.tensor) -> (torch.tensor):
        """
        Compute outputs of neural network module
        
        Parameters
        ----------
        x : torch.tensor
            input data

        Returns
        -------
        z : torch.tensor
            NN output

        See Also
        --------
        forward : compute forward pass of the mdoel
        """
        if(self.normIn):
            x = self._normalize(x,self.MeanIn,self.RangeIn)   
        z = self.nn(x)
        return z
      
    def forward(self, x: torch.tensor) -> (torch.tensor):
        """
        Compute model output. First propagate input through the NN and then linear projection.

        Parameters
        ----------
        x : torch.tensor
            input data

        Returns
        -------
        z : torch.tensor
            CVs

        See Also
        --------
        forward_nn : compute forward pass of NN module

        """
        z = self.forward_nn(x)
        if (not self.output_hidden ):
            z = self.transform(z)
        return z

    def transform(self,X):
        """
        Project data along linear components.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features)
            Inference data.
               
        Returns
        -------
        s : array-like of shape (n_samples, n_classes-1)
            Linear projection of inputs. 
        """

        s = torch.matmul(X-self.b,self.w)
    
        return s

    # Optimizer

    def set_optimizer(self,opt):
        """
        Set optimizer object. If not set uses ADAM with default parameters.

        Parameters
        ----------
        opt : torch.optimizer
            Optimizer
        """
        self.opt_ = opt
        
    def _default_optimizer(self):
        """
        Initialize default optimizer (ADAM).
        """
        self.opt_ = torch.optim.Adam(self.parameters())
        
    def set_earlystopping(self,patience=5, min_delta = 0, consecutive=True, log=False, save_best_model=True):
        """
        Enable earlystopping. 

        Parameters
        ----------
        patience : int
            how many epochs to wait before stopping when loss is not improving (default = 5)
        min_delta : float, optional 
            minimum difference between new loss and old loss to be considered as an improvement (default = 0)
        consecutive: bool, optional
            whether to consider cumulative or consecutive patience
        log: bool, optional
            print info when counter increases
        save_best: bool, optional
            store the best model
        """
        self.earlystopping_ = optim.EarlyStopping(patience,min_delta,consecutive,log,save_best_model)
        self.best_valid = None
        self.best_model = None

    # Input / output standardization

    def _compute_mean_range(self, x: torch.Tensor, print_values = False):
        '''
        Compute mean and range of values over dataset (for inputs / outputs standardization)
        '''
        Max, _=torch.max(x,dim=0)
        Min, _=torch.min(x,dim=0)

        Mean=(Max+Min)/2.
        Range=(Max-Min)/2.

        if print_values:
            print( "Standardization enabled.")
            print('Mean:',Mean.shape,'-->',Mean)
            print('Range:',Range.shape,'-->',Range)
        if (Range<1e-6).nonzero().sum() > 0 :
            print( "[Warninthe follig] Normalization: owing features have a range of values < 1e-6:", (Range<1e-6).nonzero() )
            Range[Range<1e-6]=1.

        return Mean,Range

    def standardize_inputs(self, x: torch.Tensor, print_values=False):
        """
        Enable the standardization of inputs based on max and min over set.
        
        Parameters
        ----------
        x : torch.tensor
            reference set over which compute the standardization
        """

        Mean, Range = self._compute_mean_range(x,print_values)

        self.normIn = True
        self.MeanIn = Mean
        self.RangeIn = Range

    def standardize_outputs(self, input: torch.Tensor, print_values=False):
        """
        Enable the standardization of outputs based on max and min over set.
        
        Parameters
        ----------
        x : torch.tensor
            reference set over which compute the standardization
        """
        #disable normOut for unbiased cv evaluation
        self.normOut = False
        with torch.no_grad():
            x = self.forward(input)

        Mean, Range = self._compute_mean_range(x,print_values)

        self.normOut = True
        self.MeanOut = Mean
        self.RangeOut = Range

    def _normalize(self, x: torch.Tensor, Mean: torch.Tensor, Range: torch.Tensor) -> (torch.Tensor):
        ''' Compute standardized inputs/outputs '''

        batch_size = x.size(0)
        x_size = x.size(1)

        Mean_ = Mean.unsqueeze(0).expand(batch_size, x_size)
        Range_ = Range.unsqueeze(0).expand(batch_size, x_size)

        return x.sub(Mean_).div(Range_)
    
        # Parameters

    def get_params(self):
        """
        Return saved parameters.
        
        Returns
        -------
        out : namedtuple
            Parameters
        """
        return vars(self)
    
    def set_params(self,dict_params):
        """
        Set parameters.

        Parameters
        ----------
        dict_params : dictionary
            Parameters
        """

        for key in dict_params.keys():
            if hasattr(self,key):
                setattr(self,key,dict_params[key])
            else:
                raise AttributeError(f'{self.__class__.__name__} has no attribute {key}.')
    
    # Info

    def print_info(self):
        """Display information about model"""

        print('================INFO================')
        print('[MODEL]')
        print(self)
        print('\n[OPTIMIZER]')
        print(self.opt_)
        print('\n[PARAMETERS]')
        print(self.get_params())
        print('====================================')
    
    # Log 

    def print_log(self,log_dict,spacing=None,decimals=3):
        """
        Utility function for training log.

        Parameters
        ----------
        log_dict : dict
            training log values
        spacing: int
            columns width
        decimals: int
            number of decimals in log

        """
        if spacing is None:
            spacing = [16 for i in range(len(log_dict))]
        if self.log_header:
            for i,key in enumerate(log_dict.keys()):
                print("{0:<{width}s}".format(key,width=spacing[i]),end='')
            print('')
            self.log_header = False
            
        for i,value in enumerate(log_dict.values()):
            if type(value) == int:
                print("{0:<{width}d}".format(value,width=spacing[i]),end='')
                
            if (type(value) == torch.Tensor) or (type(value) == torch.nn.parameter.Parameter) :
                value = value.cpu().numpy()
                if value.shape == ():
                    print("{0:<{width}.{dec}f}".format(value,width=spacing[i],dec=decimals),end='')
                else:
                    for v in value:
                        print("{0:<6.3f}".format(v),end=' ')
        print('')