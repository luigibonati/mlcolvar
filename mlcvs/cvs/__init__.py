__all__ = ["BaseCV", "DeepLDA_CV", "DeepTICA_CV","DeepTDA_CV","AutoEncoder_CV","Regression_CV"]

from .cv import BaseCV
from .unsupervised import *
from .supervised import *
from .timelagged import *