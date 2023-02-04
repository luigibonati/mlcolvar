from typing import Any
import torch
import pytorch_lightning as pl
from mlcvs.utils.decorators import decorate_methods, allowed_hooks, call_submodules_hooks
from mlcvs.core.models import FeedForward
from mlcvs.core.transform import Normalization
from mlcvs.cvs.utils import CV_utils

__all__ = ["AutoEncoder_CV"]

@decorate_methods(call_submodules_hooks, methods=allowed_hooks)
class AutoEncoder_CV(pl.LightningModule, CV_utils):
    """AutoEncoding Collective Variable."""
    
    def __init__(self,
                encoder_layers : list, 
                decoder_layers : list = None, 
                options : dict = {}, 
                **kwargs):
        """
        Train a CV defined as the output layer of the encoder of an autoencoder model (latent space). 
        The decoder part is used only during the training for the reconstruction loss.

        Parameters
        ----------
        encoder_layers : list
            Number of neurons per layer of the encoder
        decoder_layers : list, optional
            Number of neurons per layer of the decoder, by default None
            If not set it takes automaically the reversed architecture of the encoder
        options : dict[str,Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['normIn', 'encoder','normOut','decoder'].
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(**kwargs)

        # Members
        self.blocks = ['normIn','encoder','normOut','decoder'] 
        self.initialize_block_defaults(options=options)

        # parse info from args
        self.define_n_in_n_out(n_in=encoder_layers[0], n_out=encoder_layers[-1])
        if decoder_layers is None:
            decoder_layers = encoder_layers[::-1]

        # initialize normIn
        o = 'normIn'
        if ( options[o] is not False ) and (options[o] is not None):
            self.normIn = Normalization(self.n_in,**options[o]) 

        # initialize encoder
        o = 'encoder'
        self.encoder = FeedForward(encoder_layers, **options[o])

         # initialize normOut
        o = 'normOut'
        if ( options[o] is not False ) and (options[o] is not None):
            self.normOut = Normalization(self.n_out,**options[o]) 

        # initialize encoder
        o = 'decoder'
        self.decoder = FeedForward(decoder_layers, **options[o])

    def forward(self, x: torch.tensor) -> (torch.tensor):
        if self.normIn is not None:
            x = self.normIn(x)
        x = self.encoder(x)
        if self.normOut is not None:
            x = self.normOut(x)
        return x

    def encode_decode(self, x: torch.tensor) -> (torch.tensor):
        x = self.forward(x)
        x = self.decoder(x)
        if self.normIn is not None:
            x = self.normIn.inverse(x)
        return x
    
    def configure_optimizers(self):
        return self.initialize_default_Adam_opt()

    def loss_function(self, diff, options = {}):
        # Reconstruction (MSE) loss
        if 'weights' in options:
            w = options['weights']
            if w.ndim == 1:
                w = w.unsqueeze(1)
            loss = (diff*w).square().mean()
        else:
            loss = (diff).square().mean()
        return loss

    def training_step(self, train_batch, batch_idx):
        options = {}
        # get data
        x = train_batch['data']
        if 'weights' in train_batch:
            options['weights'] = train_batch['weights'] 
        # forward
        x_hat = self.encode_decode(x)
        # loss
        diff = x - x_hat
        loss = self.loss_function(diff, options)
        # log
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        options = {}
        # get data
        x = val_batch['data']
        if 'weights' in val_batch:
            options['weights'] = val_batch['weights'] 
        # forward
        x_hat = self.encode_decode(x)
        # loss
        diff = x - x_hat
        loss = self.loss_function(diff, options)
        # log
        self.log('val_loss', loss, on_epoch=True)

def test_autoencodercv():
    from mlcvs.utils.data import DictionaryDataset, TensorDataModule
    import numpy as np

    n_in, n_out = 8,2
    layers = [n_in, 6, 4, n_out]

    # initialize via dictionary
    opts = { 'normIn'  : None,
             'encoder' : { 'activation' : 'relu' },
             'normOut' : { 'mode'   : 'mean_std' },
           } 
    model = AutoEncoder_CV( encoder_layers=layers, options=opts )
    print(model)

    # train
    print('train 1 - no weights')
    dataset = DictionaryDataset({'data': torch.randn(100,n_in) })
    datamodule = TensorDataModule(dataset)
    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=2,logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )

    # train with weights
    print('train 2 - weights')
    dataset = DictionaryDataset({'data': torch.randn(100,n_in), 'weights' : np.arange(100) })
    datamodule = TensorDataModule(dataset)
    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=2,logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )

if __name__ == "__main__":
    test_autoencodercv() 