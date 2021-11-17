"""Linear base class"""

__all__ = ["LinearCV"]

import torch
import numpy as np
import pandas as pd
from .utils import normalize

class LinearCV(torch.nn.Module):
    """
    Linear CV base class.

    Attributes
    ----------
    w : torch.Tensor
        Weights array of linear model
    b : torch.Tensor
        Offset array 
    n_features : int
        Number of input features
    feature_names : list
        List of input features names
    device_ : torch.Device
        Device used for the model

    Methods
    -------
    __init__(n_features)
        Create a linear model.
    forward(X)
        Project data along linear model
    get_params()
        Return saved parameters
    set_params(dict)
        Set parameters via dictionaries
    set_weights(w)
        Set coefficients
    set_offset(b)
        Set linear model
    plumed_input()
        Generate PLUMED input file
    """

    def __init__(self, n_features, **kwargs):
        super().__init__(**kwargs)

        # Initialize parameters
        self.n_features = n_features
        weight = torch.eye(n_features)
        offset = torch.zeros(n_features)
        self.register_buffer("w", weight)
        self.register_buffer("b", offset)

        # Generic attributes
        self.name_ = "LinearCV"
        self.feature_names = ["x" + str(i) for i in range(n_features)]

        # Input normalization
        self.normIn = False
        self.normOut = False

    def forward(self, X):
        """
        Project data along linear components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features)
            Inference data.

        Returns
        -------
        s : array-like of shape (n_samples, n_classes-1)
            Linear projection of inputs.
        """
        if type(X) == pd.DataFrame:
            X = torch.Tensor(X.values)
        elif type(X) != torch.Tensor:
            X = torch.Tensor(X)

        if self.normIn:
            X = normalize(X, self.MeanIn, self.RangeIn)

        s = torch.matmul(X - self.b, self.w)

        if self.normOut:
            s = normalize(s, self.MeanOut, self.RangeOut)

        return s

    def set_weights(self, w):
        """
        Set weights of linear combination.

        Parameters
        ----------
        w : torch.tensor
            weights

        """
        self.w = w

    def set_offset(self, b):
        """
        Set linear offset

        Parameters
        ----------
        b : torch.tensor
            offset

        """
        self.b = b

    def get_params(self):
        """
        Return saved parameters.

        Returns
        -------
        out : namedtuple
            Parameters
        """
        return vars(self)

    def set_params(self, dict_params):
        """
        Set parameters.

        Parameters
        ----------
        dict_params : dictionary
            Parameters

        """

        for key in dict_params.keys():
            if hasattr(self, key):
                setattr(self, key, dict_params[key])
            else:
                raise AttributeError(
                    f"{self.__class__.__name__} has no attribute {key}."
                )

    def plumed_input_combine(self): # TODO REMOVE
        """
        Generate PLUMED input file

        Deprecated (todo remove)

        Returns
        -------
        out : string
            PLUMED input file
        """

        # count number of output components
        weights = self.w.cpu().numpy()
        offset = self.b.cpu().numpy()
        n_cv = 1 if weights.ndim == 1 else weights.shape[1]

        out = ""
        for i in range(n_cv):
            if n_cv == 1:
                out += f"{self.name_}: COMBINE ARG="
            else:
                out += f"{self.name_}{i+1}: COMBINE ARG="
            for j in range(self.n_features):
                out += f"{self.feature_names[j]},"
            out = out[:-1]

            out += " COEFFICIENTS="
            for j in range(self.n_features):
                out += str(np.round(weights[j, i], 6)) + ","
            out = out[:-1]
            if not np.all(offset == 0):
                out += " PARAMETERS="
                for j in range(self.n_features):
                    out += str(np.round(offset[j, i], 6)) + ","
            out = out[:-1]

            out += " PERIODIC=NO\n"
            
        return out

    def plumed_input(self):
        """
        Generate PLUMED input file

        Returns
        -------
        out : string
            PLUMED input file
        """
        
        # count number of output components
        weights = self.w.cpu().numpy()
        #offset = self.b.cpu().numpy()
        n_cv = 1 if weights.ndim == 1 else weights.shape[1]

        # get normalization
        if self.normIn:
            mean = self.MeanIn.cpu().numpy()

        out = ""
        for i in range(n_cv):
            if n_cv == 1:
                out += f"{self.name_}: CUSTOM ARG="
            else:
                out += f"{self.name_}{i+1}: CUSTOM ARG="
            for j in range(self.n_features):
                out += f"{self.feature_names[j]},"
            out = out[:-1]

            out += " VAR="
            for j in range(self.n_features):
                out += f"x{j},"
            out = out[:-1]

            out += " FUNC="
            for j in range(self.n_features):
                w = weights[j, i]
                s = "+" if w > 0 else ""
                if self.normIn:
                    m = mean[j]
                    s2 = "+" if m < 0 else "-"
                    out += f'{s}{w:.6f}*(x{j}{s2}{np.abs(m):.6f})'
                else:
                    out += f"{s}{w:.6f}*x{j}"

            out += " PERIODIC=NO\n"
        return out