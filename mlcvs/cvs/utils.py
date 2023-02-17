import torch

from typing import Any

class CV_utils:
    """
    
    To inherit from this class, the class must define a BLOCKS class attribute.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_block_defaults(self, options : dict = None):
        """
        Initialize the blocks as attributes of the CV class.
        Set the options of each block to empty dict.

        Parameters
        ----------
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default None.
        """
        if options is None:
            options = {}

        for b in self.BLOCKS:
            self.__setattr__(b,None)
            options.setdefault(b,{})
        
        return options

    def initialize_default_Adam_opt(self) -> torch.optim:
        """
        Initialize a default Adam optimizer.
        If self.lr is not defined sets lr=1e-3 .

        Returns
        -------
        torch.optim
            Torch optimizer
        """
        if not hasattr(self, 'lr'):
            self.lr =  1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return self.optimizer

    def define_in_features_out_features(self, in_features : int, out_features : int):
        """
        Initialize self.in_features and self.out_features of the CV class
        Initialize self.example_input_array accordingly

        Parameters
        ----------
        in_features : int
            Number of inputs of the CV model
        out_features : int
            Number of outputs of the CV model, should be the number of CVs
        """
        self.in_features = in_features
        self.out_features = out_features
        self.example_input_array = torch.ones(self.in_features)

    def forward(self, x : torch.tensor) -> (torch.tensor):
        """
        Execute all the blocks in self.BLOCKS unless they have been deactivated with options dict.

        Parameters
        ----------
        x : torch.tensor
            Input of the forward operation of the model
        Returns
        -------
        torch.tensor
            Output of the forward operation of the model
        """
        for b in self.BLOCKS:
            block = getattr(self, b)
            if block is not None:
                x = block(x)
        return x

    def validation_step(self, val_batch, batch_idx):
        """ Equal to training step if not overridden. Different behaviors can be enforced based on the self.training variable. 
        """
        self.training_step(val_batch, batch_idx)

    def set_loss_options(self, options : dict = None, **kwargs):
        """
        Save loss functions options to be used in train/valid step. It can either take a dictionary or kwargs. 

        Examples:
        > cvs.set_loss_options(options = {'a' : 1, 'b' : 2})
        > cvs.set_loss_options(a=1,b=2)

        Parameters
        ----------
        options : dict
            Dictionary of options (allowed keys depend on the loss used)
        """
        #add kwargs to options dict
        if options is None:
            options = {}
        self.loss_options = {**options, **locals()['kwargs']}
    
    def set_loss_fn(self, fn):
        """
        Overload loss function with given function. 'fn' need to have the following signature:
        def f(x : torch.Tensor, options : dict = {} ) -> torch.Tensor 
        where x is the same input as in the loss_function implemented in the CV.
        Lambda functions can also be used: fn = lambda x, options : -x.sum()

        Parameters
        ----------
        fn : function
            Loss function to be used in train/valid
        """
        self.loss_function = fn
