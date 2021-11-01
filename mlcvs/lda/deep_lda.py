"""Linear discriminant analysis-based CVs."""

__all__ = ["DeepLDA_CV"]

import torch
from torch.utils.data import DataLoader

from .lda import LDA
from ..models import NeuralNetworkCV
from ..utils.data import LabeledDataset

class DeepLDA_CV(NeuralNetworkCV):
    """
    Neural network based discriminant CV.
    Perform a non-linear featurization of the inputs with a neural-network and optimize it as to maximize Fisher's discriminant ratio.

    Attributes
    ----------
    lda : LDA object
        Number of classes
    lorentzian_reg: float
        Magnitude of lorentzian regularization
    epochs : int
        Number of performed epochs of training
    loss_train : list
        Loss function for train data over training. 
    loss_valid : list
        Loss function for validation data over training.

    Methods
    -------
    __init__(layers,activation,**kwargs)
        Create a DeepLDA_CV object
    set_regularization(sw_reg,lorentzian_reg)
        Set magnitudes of regularizations.
    regularization_lorentzian(H)
        Apply regularization to NN output.
    loss_function(H,y)
        Loss function for the DeepLDA CV.
    set_loss_function(func)
        Custom loss function.
    train()
        Train Deep-LDA CVs.
    evaluate_dataset(x,label)
        Evaluate loss function on dataset.
    """

    def __init__(self, layers, activation="relu", **kwargs):
        """
        Create a DeepLDA_CV object

        Parameters
        ----------
        n_features : int
            Number of input features
        **kwargs : dict
            Additional parameters for LinearCV object
        """
        
        super().__init__(
            layers=layers, activation=activation, **kwargs 
        )
        self.name_ = "deeplda_cv"
        self.lda = LDA()

        # custom loss function
        self.custom_loss = None

        # lorentzian regularization
        self.lorentzian_reg = 0

        # training logs
        self.epochs = 0
        self.loss_train = []
        self.loss_valid = []
        self.log_header = True

    def set_regularization(self, sw_reg=0.02, lorentzian_reg=None):
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
        if self.custom_loss is None:

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

        else: 
            loss = self.custom_loss(self,H,y,save_params)

        return loss

    def set_loss_function(self, func):
        """Set custom loss function

        TODO document with an example

        Parameters
        ----------
        func : function
            custom loss function
        """
        self.custom_loss = func

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
            X, y = data
            # =================forward====================
            H = self.forward_nn(X)
            # =================lda loss===================
            loss = self.loss_function(H, y, save_params=False)
            # =================backprop===================
            self.opt_.zero_grad()
            loss.backward()
            self.opt_.step()
        # ===================log======================
        self.epochs += 1

    def train(
        self,
        train_data,
        valid_data=None,
        standardize_inputs=True,
        batch_size=-1,
        nepochs=1000,
        log_every=1,
        info=False,
    ):
        """
        Train Deep-LDA CVs. Takes as input a dataloader constructed from a LabeledDataset, directly the dataloader or even a tuple of (colvar,labels) data.

        Parameters
        ----------
        train_data: DataLoader, LabeledDataset or tuple of torch.tensors (X:input, y:labels)
            training set
        valid_data: tuple of torch.tensors (X:input, y:labels) #TODO add dataloader option?
            validation set
        standardize_inputs: bool
            whether to standardize input data
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
        LabeledDataset
            Dataset with 
        """

        # check optimizer
        if self.opt_ is None:
            self.default_optimizer()

        # create dataloader
        create_loader = True
        if type(train_data) == DataLoader:
            train_loader = train_data
            create_loader = False
        elif type(train_data) == LabeledDataset:
            train_dataset = train_data
        else: 
            train_dataset = LabeledDataset(colvar = train_data[0], labels = train_data[1])

        if create_loader:
            # determine batch size
            if batch_size == -1:
                batch_size = len(train_data[0])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # standardize inputs
        if standardize_inputs:
            self.standardize_inputs(train_data[0])

        # print info
        if info:
            self.print_info()

        # train
        for ep in range(nepochs):
            self.train_epoch(train_loader)

            loss_train = self.evaluate_dataset(train_data, save_params=True)
            loss_valid = self.evaluate_dataset(valid_data)
            self.loss_train.append(loss_train)
            self.loss_valid.append(loss_valid)

            # earlystopping
            if self.earlystopping_ is not None:
                self.earlystopping_(loss_valid, self.parameters)

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
                self.parameters = self.earlystopping_.best_model
                break

    def evaluate_dataset(self, data, save_params=False):
        """
        Evaluate loss function on dataset.

        Parameters
        ----------
        data : array-like (data,labels)
            validation dataset
        save_params: bool
            save the eigenvalues/vectors of LDA into the model

        Returns
        -------
        loss : torch.tensor
            loss value
        """
        with torch.no_grad():
            X, y = data
            H = self.forward_nn(X)
            loss = self.loss_function(H, y, save_params)
        return loss


