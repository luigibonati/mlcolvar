from typing import Any
import torch
import pytorch_lightning as pl

from mlcvs.core import FeedForward, Normalization
from mlcvs.utils.data import TensorDataModule
from torch.utils.data import TensorDataset

from mlcvs.utils.decorators import decorate_methods,call_submodules_hooks,allowed_hooks

from mlcvs.cvs.utils import CV_utils

__all__ = ["Regression_CV"]

@decorate_methods(call_submodules_hooks,methods=allowed_hooks)
class Regression_CV(pl.LightningModule, CV_utils):
    """
    Example of collective variable obtained with a regression task.
    Combine the inputs with a neural-network and optimize it to match a target function
         """
    
    def __init__(self, 
                layers : list, 
                options : dict = {},
                **kwargs):
        """Example of collective variable obtained with a regression task.

        Parameters
        ----------
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
        self.initialize_block_defaults(options=options)

        # Parse info from args
        self.define_n_in_n_out(n_in=layers[0], n_out=layers[-1])

        # Initialize normIn
        o = 'normIn'
        if ( not options[o] ) and (options[o] is not None):
            self.normIn = Normalization(self.n_in,**options[o])

        # initialize NN
        o = 'nn'
        self.nn = FeedForward(layers, **options[o])

    def forward(self, x: torch.tensor) -> (torch.tensor):
        return self.forward_all_blocks(x=x)

    def configure_optimizers(self):
        return self.initialize_default_Adam_opt()

    def loss_function(self, input, target): 
        # MSE LOSS
        loss = (input-target).square().mean()
        return loss

    def training_step(self, train_batch, batch_idx):
        x, labels = train_batch
        y = self(x)
        loss = self.loss_function(y,labels)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch
        y = self(x)
        loss = self.loss_function(y,labels)
        self.log('val_loss', loss, on_epoch=True)

def test_regression_cv():
    """
    Create a synthetic dataset and test functionality of the Regression_CV class
    """
    n_in, n_out = 2,1 
    layers = [n_in, 5, 10, n_out]

    # initialize via dictionary
    options= { 'FeedForward' : { 'activation' : 'relu' } }

    model = Regression_CV( layers = layers,
                      options = options)
    print('----------')
    print(model)

    # create dataset
    X = torch.randn((100,2))
    y = X.square().sum(1)
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
    test_regression_cv() 