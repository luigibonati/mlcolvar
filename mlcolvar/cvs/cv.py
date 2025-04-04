import torch
from mlcolvar.core.transform import Transform
from typing import Union, List
from mlcolvar.core.nn import FeedForward, BaseGNN
from mlcolvar.data.graph.utils import create_test_graph_input


class BaseCV:
    """
    Base collective variable class.

    To inherit from this class, the class must define a BLOCKS class attribute.
    """

    DEFAULT_BLOCKS = []
    MODEL_BLOCKS = []

    def __init__(
        self,
        model: Union[List[int], FeedForward, BaseGNN],
        preprocessing: torch.nn.Module = None,
        postprocessing: torch.nn.Module = None,
        *args,
        **kwargs,
    ):
        """Base CV class options.

        Parameters
        ----------
        preprocessing : torch.nn.Module, optional
            Preprocessing module, default None
        postprocessing : torch.nn.Module, optional
            Postprocessing module, default None

        """
        super().__init__(*args, **kwargs)

        # The parent class sets in_features and out_features based on their own
        # init arguments so we don't need to save them here (see #103).
        
        # TODO check if need
        self.save_hyperparameters(ignore=['in_features', 'out_features'])

        # MODEL
        self.parse_model(model=model)
        self.initialize_blocks()

        # OPTIM
        self._optimizer_name = "Adam"
        self.optimizer_kwargs = {}
        self.lr_scheduler_kwargs = {}

        # PRE/POST
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    @property
    def n_cvs(self):
        """Number of CVs."""
        return self.out_features

    @property
    def example_input_array(self):
        if self.in_features is not None:
            return torch.randn(
                (1,self.in_features)
                if self.preprocessing is None
                or not hasattr(self.preprocessing, "in_features")
                else self.preprocessing.in_features
            )
        else:
            return create_test_graph_input(output_type='tracing_example', n_atoms=3, n_samples=1, n_states=1)


    # TODO add general torch.nn.Module
    def parse_model(self, model: Union[List[int], FeedForward, BaseGNN]):
        if isinstance(model, list):
            self.layers = model
            self.BLOCKS = self.DEFAULT_BLOCKS
            self._override_model = False
            self.in_features = self.layers[0]
            self.out_features = self.layers[-1]
        elif isinstance(model, FeedForward) or isinstance(model, BaseGNN):
            self.BLOCKS = self.MODEL_BLOCKS
            self._override_model = True
            self.in_features = model.in_features
            self.out_features = model.out_features
            # save buffers for the interface for PLUMED
            if isinstance(model, BaseGNN):
                self.register_buffer('n_out', model.n_out)    
                self.register_buffer('cutoff', model.cutoff)
                self.register_buffer('atomic_numbers', model.atomic_numbers)
        else:
            raise ValueError(
                f"Keyword model can either accept type list, FeedForward or BaseGNN. Found {type(model)}"
            )

    def parse_options(self, options: dict = None):
        """
        Sanitize options and create defaults ({}) if not in options.
        Furthermore, it sets the optimizer kwargs, if given.

        Parameters
        ----------
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default None.
        """
        if options is None:
            options = {}
        else:
            for o in options.keys():
                if o in self.DEFAULT_BLOCKS and self._override_model:
                    raise ValueError(
                        "Options on blocks are disabled if a model is provided!"
                        )
            
        for b in self.BLOCKS:
            options.setdefault(b, {})

        for o in options.keys():
            if o not in self.BLOCKS:
                if o == "optimizer":
                    self.optimizer_kwargs.update(options[o])
                elif o == "lr_scheduler":
                    self.lr_scheduler_kwargs.update(options[o])
                else:
                    raise ValueError(
                        f'The key {o} is not available in this class. The available keys are: {", ".join(self.BLOCKS)}, optimizer and lr_scheduler.'
                    )

        return options

    def initialize_blocks(self):
        """
        Initialize the blocks as attributes of the CV class.
        """
        for b in self.BLOCKS:
            self.__setattr__(b, None)

    def setup(self, stage=None):
        if stage == "fit":
            self.initialize_transforms(self.trainer.datamodule)

    def initialize_transforms(self, datamodule):
        for b in self.BLOCKS:
            if isinstance(getattr(self, b), Transform):
                getattr(self, b).setup_from_datamodule(datamodule)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
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
        """Optimizer name. Options can be set using optimizer_kwargs. Actual optimizer will be return during training from configure_optimizer function."""
        return self._optimizer_name

    @optimizer_name.setter
    def optimizer_name(self, optimizer_name: str):
        if not hasattr(torch.optim, optimizer_name):
            raise AttributeError(
                f"torch.optim does not have a {optimizer_name} optimizer."
            )
        self._optimizer_name = optimizer_name

    def configure_optimizers(self):
        """
        Initialize the optimizer based on self._optimizer_name and self.optimizer_kwargs.

        Returns
        -------
        torch.optim
            Torch optimizer
        """

        optimizer = getattr(torch.optim, self._optimizer_name)(
            self.parameters(), **self.optimizer_kwargs
        )

        if self.lr_scheduler_kwargs:
            scheduler_cls = self.lr_scheduler_kwargs['scheduler']
            scheduler_kwargs = {k: v for k, v in self.lr_scheduler_kwargs.items() if k != 'scheduler'}
            lr_scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
            return [optimizer] , [lr_scheduler]
        else: 
            return optimizer

    def __setattr__(self, key, value):
        # PyTorch overrides __setattr__ to raise a TypeError when you try to assign
        # an attribute that is a Module to avoid substituting the model's component
        # by mistake. This means we can't simply assign to loss_fn a lambda function
        # after it's been assigned a Module, but we need to delete the Module first.
        #    https://github.com/pytorch/pytorch/issues/51896
        #    https://stackoverflow.com/questions/61116433/maybe-i-found-something-strange-on-pytorch-which-result-in-property-setter-not
        try:
            super().__setattr__(key, value)
        except TypeError as e:
            # We make an exception only for loss_fn.
            if (key == "loss_fn") and ("cannot assign" in str(e)):
                del self.loss_fn
                super().__setattr__(key, value)

    def _setup_graph_data(self, train_batch, key : str='data_list'):
            data = train_batch[key]
            data['positions'].requires_grad_(True)
            data['node_attrs'].requires_grad_(True)
            return data