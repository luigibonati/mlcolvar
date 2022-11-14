"""Linear discriminant analysis-based CVs."""

__all__ = ["DeepLDA_CV"]

import torch
from torch.utils.data import DataLoader,TensorDataset,random_split

from .lda import LDA
from ..models import NeuralNetworkCV
from ..utils.data import FastTensorDataLoader

class DeepLDA_CV(NeuralNetworkCV):
    """
    Neural network based discriminant CV.
    Perform a non-linear featurization of the inputs with a neural-network and optimize it as to maximize Fisher's discriminant ratio.

    Attributes
    ----------
    lda : mlcvs.lda.LDA 
        linear discriminant analysis class object
    lorentzian_reg: float
        Magnitude of lorentzian regularization
    epochs : int
        Number of performed epochs of training
    loss_train : list
        Loss function for train data over training. 
    loss_valid : list
        Loss function for validation data over training.

    """

    def __init__(self, layers, device=None, activation="relu", **kwargs):
        """
        Initialize a DeepLDA_CV object

        Parameters
        ----------
        layers : list
            Number neurons per layers.
        **kwargs : dict
            Additional parameters for NeuralNetworkCV class.
        """
        
        super().__init__(
            layers=layers, activation=activation, **kwargs 
        )
        self.name_ = "deeplda_cv"
        self.lda = LDA()

        # lorentzian regularization
        self.lorentzian_reg = 0        
        self.set_regularization(0.05)

        # send model to device
        self.set_device(device) 

    def set_regularization(self, sw_reg=0.05, lorentzian_reg=None):
        """
        Set magnitude of regularizations for the training:
        - add identity matrix multiplied by `sw_reg` to within scatter S_w.
        - add lorentzian regularization to NN outputs with magnitude `lorentzian_reg`

        If `lorentzian_reg` is None, set it equal to `2./sw_reg`.

        Parameters
        ----------
        sw_reg : float
            Regularization value for S_w.
        lorentzian_reg: float
            Regularization for lorentzian on NN outputs.

        Notes
        -----
        These regularizations are described in [1]_.
        .. [1] Luigi Bonati, Valerio Rizzi, and Michele Parrinello, J. Phys. Chem. Lett. 11, 2998-3004 (2020).

        - S_w
        .. math:: S_w = S_w + \mathtt{sw_reg}\ \mathbf{1}.

        - Lorentzian

        TODO Add equation

        """
        self.lda.sw_reg = sw_reg
        if lorentzian_reg is None:
            self.lorentzian_reg = 2.0 / sw_reg
        else:
            self.lorentzian_reg = lorentzian_reg

    def regularization_lorentzian(self, H):
        """
        Compute lorentzian regularization on NN outputs.

        Parameters
        ----------
        x : float
            input data
        """
        reg_loss = H.pow(2).sum().div(H.size(0))
        reg_loss_lor = -self.lorentzian_reg / (1 + (reg_loss - 1).pow(2))
        return reg_loss_lor

    def loss_function(self, H, y, save_params=False):
        """
        Loss function for the DeepLDA CV. Correspond to maximizing the eigenvalue(s) of LDA plus a regularization on the NN outputs.
        If there are C classes the C-1 eigenvalue will be maximized.

        Parameters
        ----------
        H : torch.tensor
            NN output
        y : torch.tensor
            labels
        save_params: bool
            save the eigenvalues/vectors of LDA into the model

        Returns
        -------
        loss : torch.tensor
            loss function
        """
        eigvals, eigvecs = self.lda.compute_LDA(H, y, save_params)
        if save_params:
            self.w = eigvecs

        # TODO add sum option for multiclass

        # if two classes loss is equal to the single eigenvalue
        if self.lda.n_classes == 2:
            loss = -eigvals
        # if more than two classes loss equal to the smallest of the C-1 eigenvalues
        elif self.lda.n_classes > 2:
            loss = -eigvals[self.lda.n_classes - 2]
        else:
            raise ValueError("The number of classes for LDA must be greater than 1")

        if self.lorentzian_reg > 0:
            loss += self.regularization_lorentzian(H)

        return loss

    def evaluate_dataset(self, dataset, save_params=False, unravel_dataset = False):
        """
        Evaluate loss function on dataset.

        Parameters
        ----------
        dataset : dataloader or list of batches
            dataset
        save_params: bool
            save the eigenvalues/vectors of LDA into the model
        unravel_dataset: bool, optional
            unravel dataset to calculate loss on all dataset instead of averaging over batches   

        Returns
        -------
        loss : torch.tensor
            loss value
        """
        with torch.no_grad():
            loss = 0
            n_batches = 0

            if unravel_dataset:
                batches = [batch for batch in dataset] # to deal with shuffling
                batches = [torch.cat([batch[i] for batch in batches]) for i in range(2)] 
            elif type(dataset) == list:
                batches = [dataset]
            else: 
                batches = dataset

            for batch in batches:
                # =================get data===================
                X = batch[0].to(self.device_)
                y = batch[1].to(self.device_)
                H = self.forward_nn(X)
                # ===================loss=====================
                loss += self.loss_function(H, y, save_params)
                n_batches +=1

        return loss/n_batches