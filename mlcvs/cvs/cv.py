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
        self.in_features        = in_features
        self.out_features       = out_features
    
        # OPTIM
        self._optimizer_name     = 'Adam'
        self.optimizer_kwargs    = {}

        # LOSS
        self.loss_kwargs         = {}

        # PRE/POST
        self.preprocessing      = preprocessing
        self.postprocessing     = postprocessing

    @property
    def n_cvs(self):
        """Number of CVs."""
        return self.out_features

    @property
    def example_input_array(self):
        return torch.randn(self.in_features if self.preprocessing is None else self.preprocessing.in_features)

    def parse_options(self, options : dict = None):
        """
        Sanitize options and create defaults ({}) if not in options.
        Furthermore, sets loss and optimizer kwargs if given. 

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
                    self.loss_kwargs.update(options[o])
                elif o == 'optimizer':
                    self.optimizer_kwargs.update(options[o])
                else:
                    raise ValueError(f'The key {o} is not available in this class. The available keys are: {",".join(self.BLOCKS)}, + (loss,optimizer) ')

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

    def forward(self, x : torch.Tensor) -> (torch.Tensor):
        """
        Evaluation of the CV

        - Apply preprocessing if any
        - Execute sequentially all the blocks in self.BLOCKS unless they are not initialized
        - Apply postprocessing if any

        Parameters
        ----------
        x : torch.Tensor
            Input of the forward operation of the model

        Returns
        -------
        torch.Tensor
            Output of the forward operation of the model
        """

        if self.preprocessing is not None:
            x = self.preprocessing(x)

        x = self.forward_cv(x)

        if self.postprocessing is not None:
            x = self.postprocessing(x)

        return x

    def forward_cv(self, x : torch.Tensor) -> (torch.Tensor):
        """
        Execute sequentially all the blocks in self.BLOCKS unless they are not initialized.
        
        No pre/post processing will be executed here. This is supposed to be called during training/validation and to be overloaded if necessary.

        Parameters
        ----------
        x : torch.Tensor
            Input of the forward operation of the model

        Returns
        -------
        torch.Tensor
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

    @property
    def optimizer_name(self) -> str:
        """Optimizer name. Options can be set using optimizer_kwargs. Actual optimizer will be return during training from configure_optimizer function.
        """
        return self._optimizer_name
    
    @optimizer_name.setter
    def optimizer_name(self, optimizer_name : str): 
        if not hasattr(torch.optim, optimizer_name):
            raise AttributeError (f'torch.optim does not have a {optimizer_name} optimizer.')
        self._optimizer_name = optimizer_name

    def configure_optimizers(self): 
        """
        Initialize the optimizer based on self._optimizer_name and self.optimizer_kwargs.

        Returns
        -------
        torch.optim
            Torch optimizer
        """ 
        optimizer = getattr(torch.optim,self._optimizer_name)(self.parameters(),**self.optimizer_kwargs)
        return optimizer
