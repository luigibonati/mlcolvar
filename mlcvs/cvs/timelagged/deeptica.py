import torch
import pytorch_lightning as pl
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward,Normalization
from mlcolvar.core.stats import TICA
from mlcolvar.core.loss import ReduceEigenvaluesLoss

__all__ = ["DeepTICA"] 

class DeepTICA(BaseCV, pl.LightningModule):
    """Time-lagged independent component analysis-based CV."""
    
    BLOCKS = ['norm_in','nn','tica'] 

    def __init__(self, layers : list , n_cvs : int = None, options : dict = None, **kwargs): 
        """ 
        Neural network-based TICA CV.
        Perform a non-linear featurization of the inputs with a neural-network and optimize it as to maximize autocorrelation (e.g. eigenvalues of the transfer operator approximation).

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        n_cvs : int, optional
            Number of cvs to optimize, default None (= last layer)
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['norm_in','nn','tica'].
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(in_features=layers[0], 
                         out_features=n_cvs if n_cvs is not None else layers[-1], 
                         **kwargs)

        # =======   LOSS  =======
        # Maximize the squared sum of all the TICA eigenvalues.
        self.loss_fn = ReduceEigenvaluesLoss(mode='sum2')

        # ======= OPTIONS ======= 
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======

        # initialize norm_in
        o = 'norm_in'
        if ( options[o] is not False ) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features, **options[o]) 

        # initialize nn
        o = 'nn'
        self.nn = FeedForward(layers, **options[o])

        # initialize lda
        o = 'tica'
        self.tica = TICA(layers[-1], n_cvs, **options[o])
        
    def forward_nn(self, x: torch.Tensor) -> (torch.Tensor):
        if self.norm_in is not None:
            x = self.norm_in(x)
        x = self.nn(x)
        return x

    def set_regularization(self, c0_reg=1e-6):
        """
        Add identity matrix multiplied by `c0_reg` to correlation matrix C(0) to avoid instabilities in performin Cholesky and .
        
        Parameters
        ----------
        c0_reg : float
            Regularization value for C_0.
        """
        self.tica.reg_C_0 = c0_reg

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
        loss = self.loss_fn(eigvals)
        # ====================log=====================          
        name = 'train' if self.training else 'valid'       
        loss_dict = {f'{name}_loss' : loss}
        eig_dict = { f'{name}_eigval_{i+1}' : eigvals[i] for i in range(len(eigvals))}
        self.log_dict(dict(loss_dict, **eig_dict), on_step=True, on_epoch=True)  
        return loss

def test_deep_tica():
    # tests
    import numpy as np
    from mlcolvar.data import DictionaryDataModule
    from mlcolvar.utils.timelagged import create_timelagged_dataset

    # create dataset
    X = np.loadtxt('mlcolvar/tests/data/mb-mcmc.dat')
    X = torch.Tensor(X)
    dataset = create_timelagged_dataset(X,lag_time=1)
    datamodule = DictionaryDataModule(dataset, batch_size = 10000)

    # create cv
    layers = [2,10,10,2]
    model = DeepTICA(layers,n_cvs=1)

    # change loss options
    model.loss_fn.mode = 'sum2'

    # create trainer and fit
    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )

    model.eval()
    with torch.no_grad():
        s = model(X).numpy()
    print(X.shape,'-->',s.shape)

if __name__ == '__main__':
    test_deep_tica()