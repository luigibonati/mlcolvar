"""Base neural network class."""

import torch
from . import optim

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
    output_hidden: bool
        Output hidden layer instead of CVs

    Methods
    -------
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

    def __init__(self, layers , activation='relu'):
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

        # nn
        self.nn = torch.nn.Sequential(*modules)

        # n_input 
        self.n_input = layers[0]
        
        # options
        self.normIn = False
        self.normOut = False
        self.output_hidden = False
        
        #type and device
        self.dtype_ = torch.float32
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device_)
        
        #optimizer
        self.opt_ = None
        self.earlystopping_ = None

    # Device / optimizer / earlystopping

    def set_device(self,device):
        """
        Set desired device: torch.device('cpu') or torch.device('cuda').

        Parameters
        ----------
        device : torch.device
            Device object
        """
        self.device_ = device
        self.to(device)

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
        Set default optimizer (ADAM).
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

    def standardize_outputs(self, input: torch.Tensor, print_info=False):
        """
        Enable the standardization of outputs based on max and min over set.
        
        Parameters
        ----------
        x : torch.tensor
            reference set over which compute the standardization

        Returns
        -------
        out : dictionary
            Parameters
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
        Template function for attached parameters.
        
        Returns
        -------
        out : dictionary
            Parameters
        """
        out = dict()
        return out
    
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
                value = value.numpy()
                if value.shape == ():
                    print("{0:<{width}.{dec}f}".format(value,width=spacing[i],dec=decimals),end='')
                else:
                    for v in value:
                        print("{0:<6.3f}".format(v),end=' ')
        print('')