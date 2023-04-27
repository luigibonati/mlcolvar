__all__ = ["BaseCV", "DeepLDA", "DeepTICA","DeepTDA","AutoEncoderCV","RegressionCV","MultiTaskCV"]

from .cv import BaseCV
from .unsupervised import *
from .supervised import *
from .timelagged import *
from .multitask import *