import torch
import pytorch_lightning as pl
from mlcvs.utils.decorators import decorate_methods, allowed_hooks, call_submodules_hooks
from mlcvs.core.models import FeedForward
from mlcvs.core.transform import Normalization
from mlcvs.utils.data import DictionaryDataModule
from torch.utils.data import TensorDataset
from mlcvs.cvs.utils import BaseCV
from mlcvs.core.stats.lda import LDA
from mlcvs.core.loss.eigvals import reduce_eigenvalues

__all__ = ["DeepLDA_CV"]

@decorate_methods(call_submodules_hooks, methods=allowed_hooks)
class DeepLDA_CV(BaseCV, pl.LightningModule):
    """Neural network-based discriminant collective variables.
    
    For the training it requires a DictionaryDataset with the keys 'data' and 'labels'.
    """

    BLOCKS = ['normIn', 'nn', 'lda', 'normOut']
    
    def __init__(self, layers : list , n_states : int, options : dict = None, **kwargs):
        """ 
        Define a Deep Linear Discriminant Analysis (Deep-LDA) CV.

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        n_states : int
            Number of states for the training
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['normIn','nn','lda','normOut'] .
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs)

        # ===== BLOCKS =====

        options = self.sanitize_options(options)

        # Save n_states
        self.n_states = n_states

        # initialize normIn
        o = 'normIn'
        if ( options[o] is not False ) and (options[o] is not None):
            self.normIn = Normalization(self.in_features, **options[o]) 

        # initialize nn
        o = 'nn'
        self.nn = FeedForward(layers, **options[o])

        # initialize lda
        o = 'lda'
        self.lda = LDA(layers[-1], n_states, **options[o])
        # initialize normOut
        o = 'normOut'

        if ( options[o] is not False ) and (options[o] is not None):
            self.normOut = Normalization(self.out_features,**options[o]) 

        # regularization
        self.lorentzian_reg = 40 # == 2/sw_reg, see set_regularization   
        self.set_regularization(sw_reg=0.05)

        # ===== LOSS OPTIONS =====
        self.loss_options = {'mode':'sum'}      # eigenvalue reduction mode

    def forward_nn(self, x: torch.tensor) -> (torch.tensor):
        if self.normIn is not None:
            x = self.normIn(x)
        x = self.nn(x)
        return x

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

    def loss_function(self, eigenvalues, **kwargs):
        """
        Loss function for the DeepLDA CV. Correspond to maximizing the eigenvalue(s) of LDA.
        If there are C classes the sum of the C-1 eigenvalues will be maximized.

        Parameters
        ----------
        eigenvalues : torch.tensor
            LDA eigenvalues

        Returns
        -------
        loss : torch.tensor
            loss function
        """
        loss = - reduce_eigenvalues(eigenvalues, **kwargs)

        return loss

    def training_step(self, train_batch, batch_idx):
        # =================get data===================
        x = train_batch['data']
        y = train_batch['labels']
        # =================forward====================
        h = self.forward_nn(x)
        # ===================lda======================
        eigvals,_ = self.lda.compute(h,y,save_params=True if self.training else False) 
        # ===================loss=====================
        loss = self.loss_function(eigvals, **self.loss_options)
        if self.lorentzian_reg > 0:
            lorentzian_reg = self.regularization_lorentzian(h)
            loss += lorentzian_reg
        # ====================log=====================
        name = 'train' if self.training else 'valid'    
        loss_dict = {f'{name}_loss' : loss, f'{name}_lorentzian_reg' : lorentzian_reg}
        eig_dict = { f'{name}_eigval_{i+1}' : eigvals[i] for i in range(len(eigvals))}
        self.log_dict(dict(loss_dict, **eig_dict) ,on_step=True, on_epoch=True)
        # ===================norm=====================     
        if self.training:
            z = self.forward(x) # to accumulate info on normOut
        return loss


def test_deeplda(n_states=2):
    from mlcvs.utils.data import DictionaryDataset

    in_features, out_features = 2, n_states-1
    layers = [in_features, 50, 50, out_features]

    # create dataset
    n_points= 500
    X, y = [],[]
    for i in range(n_states):
        X.append( torch.randn(n_points,in_features)*(i+1) + torch.tensor([10*i,(i-1)*10]) )
        y.append( torch.ones(n_points)*i )

    X = torch.cat(X,dim=0)
    y = torch.cat(y,dim=0)
    
    dataset = DictionaryDataset({'data':X, 'labels':y})
    datamodule = DictionaryDataModule(dataset, lengths = [0.8,0.2], batch_size=n_states*n_points)

    # initialize CV
    opts = { 'normIn'  : { 'mode'   : 'mean_std' } ,
             'nn' :      { 'activation' : 'relu' },
             'lda' :     {} ,
             'normOut' : {} , 
           } 
    model = DeepLDA_CV( layers, n_states, options=opts )

    # create trainer and fit
    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=2,logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )

    # eval
    model.eval()
    with torch.no_grad():
        s = model(X).numpy()

if __name__ == "__main__":
    test_deeplda(n_states=2)
    test_deeplda(n_states=3) 