import torch
import pytorch_lightning as pl
from mlcvs.core.utils.decorators import decorate_methods, allowed_hooks, call_submodules_hooks
from mlcvs.core.models import FeedForward
from mlcvs.core.transform import Normalization
from mlcvs.utils.data import TensorDataModule
from torch.utils.data import TensorDataset
from mlcvs.cvs.utils import CV_utils
from mlcvs.core.utils.lda import LDA

__all__ = ["DeepLDA_CV"]

@decorate_methods(call_submodules_hooks, methods=allowed_hooks)
class DeepLDA_CV(pl.LightningModule,CV_utils):
    """Neural network-based discriminant collective variables."""
    
    def __init__(self, n_states : int, layers : list , options : dict = {}, **kwargs):
        """ 
        Define a Deep Linear Discriminant Analysis (Deep-LDA) CV.

        Parameters
        ----------
        n_states : int
            Number of states for the training
        layers : list
            Number of neurons per layer
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['normIn','nn','lda','normOut'] .
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(**kwargs)

        # Members
        self.blocks = ['normIn','nn','lda','normOut'] 
        self.initialize_block_defaults(options=options)

        # Parse info from args
        self.define_n_in_n_out(n_in=layers[0], n_out=layers[-1])

        # Save n_states
        self.n_states = n_states

        # initialize normIn
        o = 'normIn'
        if ( options[o] is not False ) and (options[o] is not None):
            self.normIn = Normalization(self.n_in, **options[o]) 

        # initialize nn
        o = 'nn'
        self.nn = FeedForward(layers, **options[o])

        # initialize lda
        o = 'lda'
        self.lda = LDA(layers[-1], n_states, **options[o])
        # initialize normOut
        o = 'normOut'

        if ( options[o] is not False ) and (options[o] is not None):
            self.normOut = Normalization(self.n_out,**options[o]) 

        # regularization
        self.lorentzian_reg = 40 # == 2/sw_reg, see set_regularization   
        self.set_regularization(sw_reg=0.05)

    def forward(self, x: torch.tensor) -> (torch.tensor):
        for b in self.blocks:
            block = getattr(self, b)
            if block is not None:
                x = block(x)
        return x

    def forward_nn(self, x: torch.tensor) -> (torch.tensor):
        if self.normIn is not None:
            x = self.normIn(x)
        x = self.nn(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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

    def loss_function(self, eigenvalues):
        """
        Loss function for the DeepLDA CV. Correspond to maximizing the eigenvalue(s) of LDA plus a regularization on the NN outputs.
        If there are C classes the C-1 eigenvalue will be maximized.

        Parameters
        ----------
        eigenvalues : torch.tensor
            LDA eigenvalues

        Returns
        -------
        loss : torch.tensor
            loss function
        """
        # if two classes the loss is equal to the single eigenvalue
        if self.lda.n_states == 2:
            loss = -eigenvalues
        # if more than two classes the loss is equal to the smallest of the C-1 eigenvalues
        # TODO ADD OPTION FOR SUM EIGVALS MULTICLASS
        elif self.lda.n_states > 2:
            loss = -eigenvalues[self.lda.n_states - 2]
        else:
            raise ValueError(f"The number of classes for LDA must be greater than 1 (detected: {self.lda.n_states})")

        return loss

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        h = self.forward_nn(x)
        S_b, S_w = self.lda.compute_scatter_matrices(h,y)
        eigvals,_ = self.lda.compute_eigenvalues(S_b,S_w,save_params=True)
        loss = self.loss_function(eigvals)
        if self.lorentzian_reg > 0:
            lorentzian_reg = self.regularization_lorentzian(h)
            loss += lorentzian_reg
        
        loss_dict = {'train_loss' : loss, 'train_lorentzian_reg' : lorentzian_reg}
        eig_dict = { f'train_eigval_{i+1}' : eigvals[i] for i in range(len(eigvals))}
        self.log_dict(dict(loss_dict, **eig_dict) ,on_step=True, on_epoch=True)
        z = self.forward(x) # to accumulate info on normOut
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        h = self.forward_nn(x)
        S_b, S_w = self.lda.compute_scatter_matrices(h,y)
        eigvals,_ = self.lda.compute_eigenvalues(S_b,S_w)
        loss = self.loss_function(eigvals)
        if self.lorentzian_reg > 0:
            lorentzian_reg = self.regularization_lorentzian(h)
            loss += lorentzian_reg
        loss_dict = {'valid_loss' : loss, 'valid_lorentzian_reg' : lorentzian_reg}
        eig_dict = { f'valid_eigval_{i+1}' : eigvals[i] for i in range(len(eigvals))}
        self.log_dict(dict(loss_dict, **eig_dict) ,on_step=True, on_epoch=True)

def test_deeplda(n_states=2):
    n_in, n_out = 2, n_states-1
    layers = [n_in, 50, 50, n_out]

    # create dataset
    n_points= 500
    X, y = [],[]
    for i in range(n_states):
        X.append( torch.randn(n_points,n_in)*(i+1) + torch.tensor([10*i,(i-1)*10]) )
        y.append( torch.ones(n_points)*i )

    X = torch.cat(X,dim=0)
    y = torch.cat(y,dim=0)
    
    datamodule = TensorDataModule(TensorDataset(X,y),
                                    lengths = [0.8,0.2], batch_size=n_states*n_points)

    # initialize CV
    opts = { 'normIn'  : { 'mode'   : 'std' } ,
             'nn' :      { 'activation' : 'relu' },
             'lda' :     {} ,
             'normOut' : {} , 
           } 
    model = DeepLDA_CV( n_states, layers, options=opts )

    # create trainer and fit
    trainer = pl.Trainer(accelerator='gpu', devices=1, 
                        callbacks=[pl.callbacks.early_stopping.EarlyStopping(monitor="valid_loss", patience=100, mode='min', min_delta=0.1, verbose=False)],
                        max_epochs=1, log_every_n_steps=2)#,logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )

    # eval
    model.eval()
    with torch.no_grad():
        s = model(X).numpy()

if __name__ == "__main__":
    test_deeplda(n_states=2)
    test_deeplda(n_states=3) 