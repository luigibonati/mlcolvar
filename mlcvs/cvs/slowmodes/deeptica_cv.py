"""Time-lagged independent component analysis-based CV"""

__all__ = ["DeepTICA_CV"] 

import torch
import pytorch_lightning as pl
# cv
from mlcvs.utils.decorators import decorate_methods, allowed_hooks, call_submodules_hooks
from mlcvs.core.models import FeedForward
from mlcvs.core.stats import TICA
from mlcvs.core.transform import Normalization
from mlcvs.cvs.utils import CV_utils
from mlcvs.core.loss.eigvals import reduce_eigenvalues


@decorate_methods(call_submodules_hooks, methods=allowed_hooks)
class DeepTICA_CV(pl.LightningModule,CV_utils):
    """Time-lagged independent component analysis-based CV."""
    
    def __init__(self, layers : list , out_features : int = None, options : dict = {}, **kwargs): 
        """ 
        Neural network-based TICA CV.
        Perform a non-linear featurization of the inputs with a neural-network and optimize it as to maximize autocorrelation (e.g. eigenvalues of the transfer operator approximation).

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        n_eig : int, optional
            Number of cvs to optimize, default None (= last layer)
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['normIn','nn','tica','normOut'] .
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(**kwargs)

        # Members
        self.blocks = ['normIn','nn','normNN','tica','normOut'] 
        self.initialize_block_defaults(options=options)

        # Parse info from args
        self.define_in_features_out_features(in_features=layers[0], out_features=out_features if out_features is not None else layers[-1])

        # initialize normIn
        o = 'normIn'
        if ( options[o] is not False ) and (options[o] is not None):
            self.normIn = Normalization(self.in_features, **options[o]) 

        # initialize nn
        o = 'nn'
        self.nn = FeedForward(layers, **options[o])

        # initialize lda
        o = 'tica'
        self.tica = TICA(layers[-1], self.out_features, **options[o])

        # initialize normOut
        o = 'normOut'
        if ( options[o] is not False ) and (options[o] is not None):
            self.normOut = Normalization(self.out_features,**options[o]) 

    def forward(self, x: torch.tensor) -> (torch.tensor):
        return self.forward_all_blocks(x)
        
    def forward_nn(self, x: torch.tensor) -> (torch.tensor):
        if self.normIn is not None:
            x = self.normIn(x)
        x = self.nn(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def set_regularization(self, c0_reg=1e-6):
        """
        Add identity matrix multiplied by `c0_reg` to correlation matrix C(0) to avoid instabilities.
        
        Parameters
        ----------
        c0_reg : float
            Regularization value for C_0.
        """
        self.tica.reg_c0 = c0_reg

    def loss_function(self, eigenvalues, options = {'mode':'sum2'}):
        """
        Loss function for the DeepTICA CV. Correspond to maximizing the eigenvalue(s) of TICA.
        By default the sum of the squares is maximized.

        Parameters
        ----------
        eigenvalues : torch.tensor
            TICA eigenvalues

        Returns
        -------
        loss : torch.tensor
            loss function
        """
        loss = - reduce_eigenvalues(eigenvalues, options)

        return loss

    def training_step(self, train_batch, batch_idx):
        """
        1) Calculate the NN output
        2) Remove average (inside forward_nn)
        3) Compute TICA
        """
        # =================get data===================
        x_t   = train_batch['data']
        x_lag = train_batch['data_lag']
        w_t   = train_batch['weights']
        w_lag = train_batch['weights_lag']
        # =================forward====================
        f_t = self.forward_nn(x_t)
        f_lag = self.forward_nn(x_lag)
        # ===================tica=====================
        eigvals, _ = self.tica.compute(data = [f_t,f_lag], 
                                                    weights = [w_t,w_lag],
                                                    save_params=True)
        # ===================loss=====================
        loss = self.loss_function(eigvals)
        # ====================log=====================            
        loss_dict = {'train_loss' : loss}
        eig_dict = { f'train_eigval_{i+1}' : eigvals[i] for i in range(len(eigvals))}
        self.log_dict(dict(loss_dict, **eig_dict), on_step=True, on_epoch=True)
        # ===================norm=====================     
        z = self.forward(x_t) # to accumulate info on normOut
        return loss

    def validation_step(self, val_batch, batch_idx):
        # =================get data===================
        x_t   = val_batch['data']
        x_lag = val_batch['data_lag']
        w_t   = val_batch['weights']
        w_lag = val_batch['weights_lag']
        # =================forward====================
        f_t = self.forward_nn(x_t)
        f_lag = self.forward_nn(x_lag)
        # ===================tica=====================
        eigvals, _ = self.tica.compute(data = [f_t,f_lag], weights = [w_t,w_lag])
        # ===================loss=====================
        loss = self.loss_function(eigvals)
        # ====================log=====================     
        loss_dict = {'valid_loss' : loss}
        eig_dict = { f'valid_eigval_{i+1}' : eigvals[i] for i in range(len(eigvals))}
        self.log_dict(dict(loss_dict, **eig_dict), on_step=True, on_epoch=True)
        

def test_deep_tica():
    # tests
    import numpy as np
    from mlcvs.utils.data import TensorDataModule, Build_TimeLagged_Dataset
    from mlcvs.utils.data import DictionaryDataset

    # create dataset
    X = np.loadtxt('mlcvs/tests/data/mb-mcmc.dat')
    X = torch.Tensor(X)
    dataset = Build_TimeLagged_Dataset(X,lag_time=10)
    datamodule = TensorDataModule(dataset, batch_size = 1024)

    # create cv
    layers = [2,10,10,2]
    model = DeepTICA_CV(layers,out_features=1)

    # create trainer and fit
    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )

    model.eval()
    with torch.no_grad():
        s = model(X).numpy()
    print(X.shape,'-->',s.shape)

if __name__ == '__main__':
    test_deep_tica()