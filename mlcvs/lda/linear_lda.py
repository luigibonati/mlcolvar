"""Linear discriminant analysis-based CVs."""

__all__ = ["LDA_CV"]

import torch
import pandas as pd

from .lda import LDA
from ..models import LinearCV

class LDA_CV(LinearCV):
    """
    Linear Discriminant-based collective variable.

    Attributes
    ----------
    lda: LDA object
        Linear discriminant analysis instance.

    Methods
    -------
    fit(X,y)
        Fit LDA given data and classes
    fit_predict(X,y)
        Fit LDA and project along components
    """

    def __init__(self, n_features, harmonic_lda = False, **kwargs):
        """
        Create a LDA_CV object

        Parameters
        ----------
        n_features : int
            Number of input features
        harmonic_lda : bool
            Build a HLDA CV.
        **kwargs : dict
            Additional parameters for LinearCV object 
        """
        super().__init__(n_features=n_features, **kwargs)

        self.name_ = "hlda_cv" if harmonic_lda else "lda_cv"
        self.lda = LDA(harmonic_lda)

    def fit(self, X, y):
        """
        Fit LDA given data and classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Classes labels.
        """
        
        #TODO : warning, this works only if device is CPU
        # if DataFrame save feature names
        if type(X) == pd.DataFrame:
            self.feature_names = X.columns.values
            X = torch.Tensor(X.values)
        elif type(X) != torch.Tensor:
            X = torch.Tensor(X)

        # class
        if type(y) == pd.DataFrame:
            y = torch.Tensor(y.values)
        elif type(y) != torch.Tensor:
            y = torch.Tensor(y)

        _, eigvecs = self.lda.compute_LDA(X, y)
        # save parameters for estimator
        self.w = eigvecs

    def fit_predict(self, X, y):
        """
        Fit LDA and project along components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Classes labels.

        Returns
        -------
        s : array-like of shape (n_samples, n_classes-1)
            Linear projection of inputs.

        """
        self.fit(X, y)
        return self.forward(X)

    def set_regularization(self, sw_reg = 0.05):
        """
        Set regularization for within-scatter matrix.

        Parameters
        ----------
        sw_reg : float
            Regularization value.

        Notes
        -----
        .. math:: S_w = S_w + \mathtt{sw_reg}\ \mathbf{1}.

        """
        self.lda.sw_reg = sw_reg

