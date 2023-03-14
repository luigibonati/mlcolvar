import torch
from mlcvs.core.transform import Transform

class BaseCV:
    """
    To inherit from this class, the class must define a BLOCKS class attribute.
    """

    def __init__(self, in_features, out_features, 
                        preprocessing : torch.nn.Module = None, postprocessing : torch.nn.Module = None,
                        *args, **kwargs):
        """ Base CV class options.

        Parameters
        ----------
        in_features : int
            Number of inputs of the CV model
        out_features : int
            Number of outputs of the CV model, should be the number of CVs
        preprocessing : torch.nn.Module, optional
            Preprocessing module, default None
        postprocessing : torch.nn.Module, optional
            Postprocessing module, default None
            
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        # MODEL 
        self.initialize_blocks()

        # Set pre/post processing 
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

        # adapt no. input and output features based on pre/post processing
        self.in_features = in_features if preprocessing is None else preprocessing.in_features
        self.out_features = out_features if postprocessing is None else postprocessing.out_features

        self.example_input_array = torch.randn(self.in_features)

        # OPTIM
        self.optim_name = 'Adam'
        self.optim_options = {}

        # LOSS
        self.loss_options = {}

    def sanitize_options(self, options : dict = None):
        """
        Parameters
        ----------
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default None.
        """
        if options is None:
            options = {}

        for b in self.BLOCKS:
            options.setdefault(b,{})

        for o in options.keys():
            if o not in self.BLOCKS:
                if o == 'loss':
                    self.set_loss_options(options[o])
                elif o == 'optim':
                    self.set_optim_options(options[o])
                else:
                    raise ValueError(f'The key {o} is not available in this class. The available keys are: {",".join(self.BLOCKS)},loss,optim ')

        return options

    def initialize_blocks(self):
        """
        Initialize the blocks as attributes of the CV class.
        """
        for b in self.BLOCKS:
            self.__setattr__(b,None)

    def setup(self, stage=None):
        if stage == "fit":
            self.initialize_transforms(self.trainer.datamodule)

    def initialize_transforms(self, datamodule):
        for b in self.BLOCKS:
            if isinstance(getattr(self,b), Transform): 
                getattr(self,b).setup_from_datamodule(datamodule)

    def forward(self, x : torch.tensor) -> (torch.tensor):
        """
        Evaluation of the CV

        - Apply preprocessing if any
        - Execute sequentially all the blocks in self.BLOCKS unless they are not initialized
        - Apply postprocessing if any

        Parameters
        ----------
        x : torch.tensor
            Input of the forward operation of the model

        Returns
        -------
        torch.tensor
            Output of the forward operation of the model
        """

        if self.preprocessing is not None:
            x = self.preprocessing(x)

        x = self.forward_blocks(x)

        if self.postprocessing is not None:
            x = self.postprocessing(x)

        return x

    def forward_blocks(self, x : torch.tensor) -> (torch.tensor):
        """
        Execute sequentially all the blocks in self.BLOCKS unless they are not initialized.
        
        No pre/post processing will be executed here. This is supposed to be called during training/validation and to be overloaded if necessary.

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
        """ 
        Equal to training step if not overridden. Different behaviors for train/valid step can be enforced in training_step() based on the self.training variable. 
        """
        self.training_step(val_batch, batch_idx)

    def test_step(self, test_batch, batch_idx):
        """ 
        Equal to training step if not overridden. Different behaviors for train/valid step can be enforced in training_step() based on the self.training variable. 
        """
        self.training_step(test_batch, batch_idx)

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

    def set_loss_options(self, options : dict = None, **kwargs):

        """
        Save loss functions options to be used in train/valid step. It can either take a dictionary or arguments. 

        Examples:
        >>> cvs.set_loss_options(options = {'a' : 1, 'b' : 2})
        >>> cvs.set_loss_options(a=1,b=2)

        Parameters
        ----------
        options : dict
            Dictionary of options to be passed to the loss during train/valid steps.
        """
        if options is None:
            options = {}
        #update saved options based on both provided dict options and kwargs
        self.loss_options.update({**options, **locals()['kwargs']})

    def set_optim_name(self, optim_name : str): 
        """Choose optimizer. Options can be set using set_optim_options. Actual optimizer will be return from configure_optimizer function.

        Parameters
        ----------
        optim : str
            Name of the torch.optim optimizer
        """
        if not hasattr(torch.optim, optim_name):
            raise AttributeError (f'torch.optim does not have a {optim_name} optimizer.')
        self.optim_name = optim_name

    def set_optim_options(self, options : dict = None, **kwargs):
        """
        Save options to be used for creating optimizer in configure_optimizer function.

        Examples:
        >>> cvs.set_optim_options(options = {'weight_decay' : 1e-5, 'lr' : 1e-3})
        >>> cvs.set_optim_options(lr=1e-3)

        Parameters
        ----------
        options : dict
            Dictionary of options
        """
        if options is None:
            options = {}
        #update saved options based on both provided dict options and kwargs
        self.optim_options.update({**options, **locals()['kwargs']})

    def configure_optimizers(self): 
        """
        Initialize the optimizer based on self.optim_name and self.optim_options.

        Returns
        -------
        torch.optim
            Torch optimizer
        """ 
        optimizer = getattr(torch.optim,self.optim_name)(self.parameters(),**self.optim_options)
        return optimizer
