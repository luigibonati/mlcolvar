from pathlib import Path
import torch
import lightning
from lightning.pytorch.core.module import _jit_is_scripting, get_filesystem
from mlcolvar.core.transform import Transform
from typing import Any, Dict, Optional, Union, List
from torch.jit import ScriptModule
from mlcolvar.core.nn import FeedForward, BaseGNN
from mlcolvar.data.graph.utils import create_graph_tracing_example


class BaseCV(lightning.LightningModule):
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
        # It is needed for compatibility with multiclass CVs
        self.save_hyperparameters(ignore=['in_features', 'out_features'])

        # MODEL
        self.parse_model(model=model)
        self.initialize_blocks()

        # OPTIM
        self._optimizer_name = "Adam"
        self.optimizer_kwargs = {}
        self.lr_scheduler_kwargs = {}
        self.lr_scheduler_config = {}

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
            return create_graph_tracing_example(n_species=len(self.atomic_numbers))


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
                elif o == "lr_scheduler_config":
                    self.lr_scheduler_config.update(options[o])
                else:
                    raise ValueError(
                        f'The key {o} is not available in this class. The available keys are: {", ".join(self.BLOCKS)}, optimizer, lr_scheduler, and lr_scheduler_config.'
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

    def forward(self, x: torch.Tensor, cell=None) -> torch.Tensor:
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
            x = self._apply_module(self.preprocessing, x, cell=cell)

        x = self.forward_cv(x)

        if self.postprocessing is not None:
            x = self._apply_module(self.postprocessing, x)

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
                x = self._apply_module(block, x)

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
        It also adds the learning rate scheduler if self.lr_scheduler_kwargs is not empty.
        The scheduler is given as a dictionary with the key 'scheduler' containing the scheduler class
        and the rest of the keys are config options for the scheduler.

        Returns
        -------
        torch.optim
            Torch optimizer
            
        dict, optional
            Learning rate scheduler configuration (if any)
        """

        # Create the optimizer from the optimizer name and kwargs
        optimizer = getattr(torch.optim, self._optimizer_name)(
            self.parameters(), **self.optimizer_kwargs
        )
        
        # Return just the optimizer if no scheduler is defined
        if not self.lr_scheduler_kwargs:
            return optimizer
        
        # Create the scheduler from the lr_scheduler_kwargs if any
        if "scheduler" not in self.lr_scheduler_kwargs:
            raise ValueError("lr_scheduler_kwargs must include a 'scheduler' key with the scheduler class.")

        scheduler_cls = self.lr_scheduler_kwargs["scheduler"]
        scheduler_kwargs = {
            k: v for k, v in self.lr_scheduler_kwargs.items() if k != "scheduler"
        }
        lr_scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
        lr_scheduler_config = {
            "scheduler": lr_scheduler
        }

        # Add possible additional config options
        if self.lr_scheduler_config:
            if "scheduler" in self.lr_scheduler_config:
                raise ValueError("lr_scheduler_config cannot override the 'scheduler' entry.")
            lr_scheduler_config.update(self.lr_scheduler_config)
        return [optimizer], [lr_scheduler_config]

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
    
    def _apply_module(self, module: torch.nn.Module, x, cell=None):
        if module is None:
            return x
        if cell is not None:
            return module(x, cell=cell)
        return module(x)

    @staticmethod
    def _get_batch_cell(batch):
        if isinstance(batch, dict):
            return batch.get("cell", None)
        return None

    @torch.no_grad()
    def to_torchscript(
        self,
        file_path: Optional[Union[str, Path]] = None,
        method: Optional[str] = "script",
        example_inputs: Optional[Any] = None,
        **kwargs: Any,
    ) -> Union[ScriptModule, Dict[str, torch.ScriptModule]]:
        """By default compiles the whole model to a `torch.jit.ScriptModule` Tracing can be used with the
        argument `method='trace'`. In case, you can provide and `example_inputs`, otherwise, the default 
        `example_input_array` will be used. 

        Args:
            file_path: Path where to save the torchscript. Default: None (no file saved).
            method: Whether to use TorchScript's script or trace method. Default: 'script'
            example_inputs: An input to be used to do tracing when method is set to 'trace'.
              Default: None (uses :attr:`example_input_array`)
            **kwargs: Additional arguments that will be passed to the :func:`torch.jit.script` or
              :func:`torch.jit.trace` function.

        Return:
            This LightningModule as a torchscript, regardless of whether `file_path` is
            defined or not.
        """
        # check if preprocessing has varible cells
        if self.preprocessing is not None:
            if hasattr(self.preprocessing, "cell"):
                Warning("Found a descriptor-based preprocessing module. If the same descriptors can be computed with PLUMED,"
                        "it is recommended for performance to export the model without the preprocessing and compute the descriptors with PLUMED."
                    )
                if self.preprocessing.cell is None:
                    raise ValueError(
                        "Found a descriptor-based preprocessing module without a defined cell, as it was passed at runtime."
                        "Tracing or scripting of preprocessing modules with variable cells is not supported yet."
                        "If changing cell is NOT needed, you can set the fixed cell during the inizialization of the descriptor module"
                        "and overwrite the model.preprocessing with the same module with the fixed cell."
                    )
                
        mode = self.training

        if method == "script":
            with _jit_is_scripting():
                torchscript_module = torch.jit.script(self.eval(), **kwargs)
        elif method == "trace":
            # if no example inputs are provided, try to see if model has example_input_array set
            if example_inputs is None:
                if self.example_input_array is None:
                    raise ValueError(
                        "Choosing method=`trace` requires either `example_inputs`"
                        " or `model.example_input_array` to be defined."
                    )
                example_inputs = self.example_input_array

            # automatically send example inputs to the right device and use trace
            example_inputs = self._on_before_batch_transfer(example_inputs)
            example_inputs = self._apply_batch_transfer_handler(example_inputs)
            with _jit_is_scripting():
                torchscript_module = torch.jit.trace(func=self.eval(), example_inputs=example_inputs, **kwargs)
        else:
            raise ValueError(f"The 'method' parameter only supports 'script' or 'trace', but value given was: {method}")

        self.train(mode)

        if file_path is not None:
            fs = get_filesystem(file_path)
            with fs.open(file_path, "wb") as f:
                torch.jit.save(torchscript_module, f)

        return torchscript_module
