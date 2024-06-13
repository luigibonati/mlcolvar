"""The graph neural network module."""

from . import utils
from . import core
from . import data
from . import cvs
from . import explain

utils.torch_tools.set_default_dtype('float32')
