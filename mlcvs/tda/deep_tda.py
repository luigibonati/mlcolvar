"""Targeted Discriminant Analysis-based CVs"""

__all__ = ["DeepTDA_CV"]

import torch
import numpy as np
from ..models import NeuralNetworkCV

class DeepTDA_CV(NeuralNetworkCV):
    """
    Neural network based targeted discriminant CV.
    Perform a non-linear featurization of the inputs with a neural-network and optimize it in a way such that the data are distributed accordingly to a target distribution.

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

    def __init__(self, layers, states_num, cvs_num, target_centers, target_sigmas, alfa=1, beta=250, activation="relu",
                 device=None, **kwargs):
        """
        Initialize a DeepTDA_CV object

        Parameters
        ----------
        layers : list
            Number neurons per layers.
        states_num: int
            Number of states to be used for the training.
        cvs_num: int
            Number of output CVs
        target_centers: list
            Centers of the gaussian targets (with shape: [states_num, cvs_num])
        target_sigma: list
            Standard deviations of the gaussian targets (with shape: [states_num, cvs_num])
        alfa: float
            Prefactor for center contributions to loss function (default = 1)
        beta: float
            Prefactor for sigma contributions to loss function (default = 250)
        **kwargs : dict
            Additional parameters for NeuralNetworkCV class.
        """
        layers.append(cvs_num)

        super().__init__(
            layers=layers, activation=activation, **kwargs
        )
        self.name_ = "deeptda_cv"

        # set number of states
        self.states_num = states_num

        # set number of CVs
        self.cvs_num = cvs_num

        # set targets
        self.target_centers = np.array(target_centers)
        self.target_sigmas = np.array(target_sigmas)

        # set loss prefactors
        self.alfa = alfa
        self.beta = beta

        # send model to device
        self.set_device(device) 

    def loss_function(self, H, y, save_params = None):
        """
        Loss function for the DeepTDA CV. Corresponds to minimizing the distance of each states from a target Gaussian distribution in the CV space given by the NN output.
        The loss is written only in terms of the mean, mu, and the standard deviation, sigma, of the data computed along the components of the CVs space.

        Parameters
        ----------
        H : torch.tensor
            NN output
        y : torch.tensor
            labels
        save_params: 
            not used
        -------
        loss : torch.tensor
            loss function
        """
        lossMu, lossSigma = torch.zeros(self.target_centers.shape, device=self.device_), torch.zeros(
        self.target_centers.shape, device=self.device_)

        for i in range(self.states_num):
            # check which elements belong to class i
            H_red = H[torch.nonzero(y == i).view(-1)]

            # compute mean over the class i
            mu = torch.mean(H_red, 0)
            # compute standard deviation over class i
            sigma = torch.std(H_red, 0)

            # compute loss function contribute for class i
            lossMu[i] = self.alfa * (mu - torch.tensor(self.target_centers[i], device=self.device_)).pow(2)
            lossSigma[i] = self.beta * (sigma - torch.tensor(self.target_sigmas[i], device=self.device_)).pow(2)

        loss = torch.sum(lossMu) + torch.sum(lossSigma)
        # to output each contribute of the loss uncomment here
        # lossMu, lossSigma = torch.reshape(lossMu, (self.states_num, self.cvs_num)), torch.reshape(lossSigma, (self.states_num, self.cvs_num))

        return loss

# TODO
    # def visualize_results(self, data_loader):
    #     '''
    #     Plot points distributions in the DeepTDA cv space

    #     Parameters
    #     ----------
    #     dataset : dataloader or list of batches
    #         dataset

    #     '''