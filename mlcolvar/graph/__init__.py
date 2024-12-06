"""The graph neural network module."""

from . import utils
from . import core
from . import data
from . import cvs
from . import explain

utils.torch_tools.set_default_dtype('float32')
# torch_scatter will compile problems.
__import__('torch_geometric').typing.WITH_TORCH_SCATTER = False
