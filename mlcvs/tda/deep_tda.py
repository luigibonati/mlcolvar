"""Targeted Discriminant Analysis-based CVs"""

__all__ = ["DeepTDA_CV"]

import torch
from torch.utils.data import TensorDataset, random_split

import numpy as np

from ..models import NeuralNetworkCV
from ..utils.data import FastTensorDataLoader


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

    def loss_function(self, H, y):
        """
        Loss function for the DeepTDA CV. Corresponds to minimizing the distance of each states from a target Gaussian distribution in the CV space given by the NN output.
        The loss is written only in terms of the mean, mu, and the standard deviation, sigma, of the data computed along the components of the CVs space.

        Parameters
        ----------
        H : torch.tensor
            NN output
        y : torch.tensor
            labels
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

    def train_epoch(self, loader):
        """
        Auxiliary function for training an epoch.

        Parameters
        ----------
        loader: DataLoader
            training set
        """
        for data in loader:
            # =================get data===================
            X = data[0].to(self.device_)
            y = data[1].to(self.device_)
            # =================forward====================
            H = self.forward_nn(X)
            # =================tda loss=================== # ??? can I add the contributes also?
            loss = self.loss_function(H, y)
            # =================backprop===================
            self.opt_.zero_grad()
            loss.backward()
            self.opt_.step()
        # ===================log======================
        self.epochs += 1

    def fit(
            self,
            train_loader=None,
            valid_loader=None,
            X=None,
            y=None,
            standardize_inputs=True,
            standardize_outputs=True,
            batch_size=0,
            nepochs=1000,
            log_every=1,
            info=False,
            earlystopping=None
    ):
        """
        Train Deep-TDA CVs. Takes as input a FastTensorDataLoader/standard Dataloader constructed from a TensorDataset, or even a tuple of (colvar,labels) data.

        Parameters
        ----------
        train_data: FastTensorDataLoader/DataLoader, or tuple of torch.tensors (X:input, y:labels)
            training set
        valid_data: tuple of torch.tensors (X:input, y:labels) #TODO add dataloader option?
            validation set
        X: np.array or torch.Tensor, optional
            input data, alternative to train_loader (default = None)
        y: np.array or torch.Tensor, optional
            labels (default = None)
        standardize_inputs: bool
            whether to standardize input data
        standardize_outputs: bool
            whether to standardize CVs
        batch_size: bool, optional
            number of points per batch (default = -1, single batch)
        nepochs: int, optional
            number of epochs (default = 1000)
        log_every: int, optional
            frequency of log (default = 1)
        print_info: bool, optional
            print debug info (default = False)

        See Also
        --------
        loss_function
            Loss functions for training Deep-LDA CVs
        """

        # check optimizer
        if self.opt_ is None:
            self._set_default_optimizer()

        # check device
        if self.device_ is None:
            self.device_ = next(self.nn.parameters()).device

        # set earlystopping variable
        self.earlystopping_ = earlystopping

        # assert to avoid redundancy
        if (train_loader is not None) and (X is not None):
            raise KeyError('Only one between train_loader and X can be used.')

        # create dataloader if not given
        if X is not None:
            if y is None:
                raise KeyError('labels (y) must be given.')

            if type(X) != torch.Tensor:
                X = torch.Tensor(X)
            if type(y) != torch.Tensor:
                y = torch.Tensor(y)

            dataset = TensorDataset(X, y)
            train_size = int(0.9 * len(dataset))
            valid_size = len(dataset) - train_size

            train_data, valid_data = random_split(dataset, [train_size, valid_size])
            train_loader = FastTensorDataLoader(train_data, batch_size)
            valid_loader = FastTensorDataLoader(valid_data)
            print('Training   set:', len(train_data))
            print('Validation set:', len(valid_data))

        # standardize inputs (unravel dataset to compute average)
        x_train = torch.cat([batch[0] for batch in train_loader])
        if standardize_inputs:
            self.standardize_inputs(x_train)

        # print info
        if info:
            self.print_info()

        # train
        for ep in range(nepochs):
            self.train_epoch(train_loader)

            loss_train = self.evaluate_dataset(train_loader) 
            loss_valid = self.evaluate_dataset(valid_loader)
            self.loss_train.append(loss_train)
            self.loss_valid.append(loss_valid)

            # standardize output
            if standardize_outputs:
                self.standardize_outputs(x_train)

            # earlystopping
            if self.earlystopping_ is not None:
                if valid_loader is None:
                    raise ValueError('EarlyStopping requires validation data')
                self.earlystopping_(loss_valid, model=self.state_dict())
            else:
                self.set_earlystopping(patience=1e30)

            # log
            if ((ep + 1) % log_every == 0) or (self.earlystopping_.early_stop):
                self.print_log(
                    {
                        "Epoch": ep + 1,
                        "Train Loss": loss_train,
                        "Valid Loss": loss_valid,
                    },
                    spacing=[6, 12, 12],
                    decimals=2,
                )

            # check whether to stop
            if (self.earlystopping_ is not None) and (self.earlystopping_.early_stop):
                self.load_state_dict(self.earlystopping_.best_model)
                break

    def evaluate_dataset(self, dataset, unravel_dataset=False):
        """
        Evaluate loss function on dataset.

        Parameters
        ----------
        dataset : dataloader or list of batches
            dataset
        unravel_dataset: bool, optional
            unravel dataset to calculate LDA loss on all dataset instead of averaging over batches

        Returns
        -------
        loss : torch.tensor
            loss value
        """
        with torch.no_grad():
            loss = 0
            n_batches = 0

            if unravel_dataset:
                batches = [batch for batch in dataset]  # to deal with shuffling
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
                loss += self.loss_function(H, y)
                n_batches += 1

        return loss / n_batches

# TODO
    # def visualize_results(self, data_loader):
    #     '''
    #     Plot points distributions in the DeepTDA cv space

    #     Parameters
    #     ----------
    #     dataset : dataloader or list of batches
    #         dataset

    #     '''