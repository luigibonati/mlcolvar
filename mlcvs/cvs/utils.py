import torch

from typing import Any

from mlcvs.core.utils.decorators import decorate_methods,call_submodules_hooks,allowed_hooks

__all__ = ["initialize_block_defaults", "define_n_in_n_out", "forward_all_blocks"] 

def initialize_block_defaults(self, options : dict[str, Any] = {}):
    """
    Initialize the blocks as attributes of the CV class
    Set the options of each block to empty dict

    Parameters
    ----------
    self : 
        Self instance of the CV class 
    options : dict[str, Any], optional
        Options for the building blocks of the model, by default {}.
    """
    for b in self.blocks:
        self.__setattr__(b,None)
        options.setdefault(b,{})

def define_n_in_n_out(self, n_in : int, n_out : int):
    """
    Initialize self.n_in and self.n_out of the CV class
    Initialize self.example_input_array accordingly

    Parameters
    ----------
    self : 
        Self instance of the CV class 
    n_in : int
        Number of inputs of the CV model
    n_out : int
        Number of outputs of the CV model, should be the number of CVs
    """
    self.n_in = n_in
    self.n_out = n_out
    self.example_input_array = torch.ones(self.n_in)

def forward_all_blocks(self, x : torch.tensor) -> (torch.tensor):
    """
    Execute all the blocks in self.blocks unless they have been deactivated with options dict

    Parameters
    ----------
    self : 
        Self instance of the CV class 
    x : torch.tensor
        Input of the forward operation of the model
    Returns
    -------
    torch.tensor
        Output of the forward operation of the model
    """
    for b in self.blocks:
        block = getattr(self, b)
        if block is not None:
            x = block(x)
    return x