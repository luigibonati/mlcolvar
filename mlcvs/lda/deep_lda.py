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

    def __init__(self, layers, activation="relu", device = None, **kwargs):
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

        # set device 
        self.device_ = device

        # custom loss function
        self.custom_loss = None

        # lorentzian regularization
        self.lorentzian_reg = 0

        # training logs
        self.epochs = 0
        self.loss_train = []
        self.loss_valid = []
        self.log_header = True

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
            X = data[0].to(self.device_)
            y = data[1].to(self.device_)
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

    def fit(
        self,
        train_loader=None,
        valid_loader=None,
        X = None,
        y = None,
        standardize_inputs=True,
        standardize_outputs=True,
        batch_size=0,
        nepochs=1000,
        log_every=1,
        info=False,
    ):
        """
        Train Deep-LDA CVs. Takes as input a FastTensorDataLoader/standard Dataloader constructed from a TensorDataset, or even a tuple of (colvar,labels) data.

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
                            
            dataset = TensorDataset(X,y)
            train_size = int(0.9 * len(dataset))
            valid_size = len(dataset) - train_size

            train_data, valid_data = random_split(dataset,[train_size,valid_size])
            train_loader = FastTensorDataLoader(train_data,batch_size)
            valid_loader = FastTensorDataLoader(valid_data)
            print('Training   set:' ,len(train_data))
            print('Validation set:' ,len(valid_data))

        if self.lda.sw_reg == 1e-6: # default value
            self.set_regularization(0.05)
            print('Sw regularization:' ,self.lda.sw_reg)
            print('Lorentzian reg.  :' ,self.lorentzian_reg)
            print('')

        # standardize inputs (unravel dataset to compute average)
        x_train = torch.cat([batch[0] for batch in train_loader])
        if standardize_inputs:
            self.standardize_inputs( x_train )

        # print info
        if info:
            self.print_info()

        # train
        for ep in range(nepochs):
            self.train_epoch(train_loader)

            loss_train = self.evaluate_dataset(train_loader, save_params=True)
            loss_valid = self.evaluate_dataset(valid_loader)
            self.loss_train.append(loss_train)
            self.loss_valid.append(loss_valid)

            #standardize output
            if standardize_outputs:
                self.standardize_outputs(x_train)

            # earlystopping
            if self.earlystopping_ is not None:
                if valid_loader is None:
                    raise ValueError('EarlyStopping requires validation data')
                self.earlystopping_(loss_valid, model=self.state_dict() )

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
                self.load_state_dict( self.earlystopping_.best_model )
                break


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