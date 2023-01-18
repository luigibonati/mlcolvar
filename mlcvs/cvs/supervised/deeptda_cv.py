import torch 
import pytorch_lightning as pl
from typing import Any

from mlcvs.core import FeedForward, Normalization
from mlcvs.core import TDA_loss

from mlcvs.utils.data import TensorDataModule
from torch.utils.data import TensorDataset

from mlcvs.core.utils.decorators import decorate_methods,call_submodules_hooks,allowed_hooks

@decorate_methods(call_submodules_hooks,methods=allowed_hooks)
class DeepTDA_CV(pl.LightningModule):
    """
    Define Deep Targeted Discriminant Analysis (Deep-TDA) CV.
    Combine the inputs with a neural-network and optimize it in a way such that the data are distributed accordingly to a target distribution.
    """
    def __init__(self,
                n_states : int,
                n_cvs : int,
                target_centers : list, 
                target_sigmas : list, 
                layers : list, 
                options : dict[str, Any] = {}, 
                **kwargs):
        """
        Define Deep Targeted Discriminant Analysis (Deep-TDA) CV.

        Parameters
        ----------
        n_states : int
            Number of states for the training
        n_cvs : int
            Numnber of collective variables to be trained
        target_centers : list
            Centers of the Gaussian targets
        target_sigmas : list
            Standard deviations of the Gaussian targets
        layers : list
            Number of neurons per layer
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['normIn', 'nn'].
            Set 'block_name' = None or False to turn off that block
        """

        super().__init__(**kwargs)

        # Members
        self.blocks = ['normIn', 'nn']

        # Initialize defaults #BASE_CV?
        for b in self.blocks:
            self.__setattr__(b,None)
            options.setdefault(b,{})
        
        # Parse info from args
        self.n_in = layers[0]
        self.n_states = n_states
        self.n_out = n_cvs
        self.n_cvs = n_cvs
        #TODO assert that n_cvs = layers[-1]
        self.target_centers = target_centers
        self.target_sigmas =  target_sigmas
        #TODO assert that size of target makes sense with n_cvs and n_states

        # Initialize normIn
        o = 'normIn'
        if ( not options[o] ) and (options[o] is not None):
            self.normIn = Normalization(self.n_in,**options[o])

        # initialize NN
        o = 'nn'
        self.nn = FeedForward(layers, **options[o])

        self.example_input_array = torch.ones(self.n_in) #BASE_CV?

        # parameters
        self.lr = 1e-3

    def forward(self, x: torch.tensor) -> (torch.tensor): #BASE_CV?
        for b in self.blocks:
            block = getattr(self, b)
            if block is not None:
                x = block(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss_function(self, input, labels):
        # TDA loss
        loss, loss_centers, loss_sigmas = TDA_loss(input, labels, self.n_states, self.target_centers, self.target_sigmas)
        return loss, loss_centers, loss_sigmas

    def training_step(self, train_batch, batch_idx):
        x, labels = train_batch
        y = self(x)
        loss, loss_centers, loss_sigmas = self.loss_function(y,labels)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_loss_centers', loss_centers, on_epoch=True)
        self.log('train_loss_sigmas', loss_sigmas, on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch
        y = self(x)
        loss, loss_centers, loss_sigmas = self.loss_function(y,labels)
        self.log('valid_loss', loss, on_epoch=True)
        self.log('valid_loss_centers', loss_centers, on_epoch=True)
        self.log('valid_loss_sigmas', loss_sigmas, on_epoch=True)


def test_deeptda_cv():
    """
    Create a synthetic dataset and test functionality of the DeepTDA_CV class
    """
    n_in, n_out = 2,1 
    layers = [n_in, 24, 12, n_out]

    # initialize via dictionary
    options= { 'FeedForward' : { 'activation' : 'relu' } }

    model = DeepTDA_CV(n_states = 2,
                        n_cvs = 1,
                        target_centers = [-10, 10],
                        target_sigmas = [0.2, 0.2],
                        layers = layers
                        )
    print('----------')
    print(model)

    # create dataset
    X = torch.randn((100,2))
    # make two 'states' different from each other
    X[:50] += 10
    X[50:] -= 10
    # create labels
    y = torch.zeros(X.shape[0])
    y[50:] += 1
    
    dataset = TensorDataset(X,y)
    datamodule = TensorDataModule(dataset,lengths=[0.75,0.2,0.05], batch_size=25)
    # train model
    trainer = pl.Trainer(accelerator='cpu',max_epochs=2,logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )
    # trace model
    traced_model = model.to_torchscript(file_path=None, method='trace', example_inputs=X[0])
    model.eval()
    assert torch.allclose(model(X),traced_model(X))

if __name__ == "__main__":
    test_deeptda_cv() 