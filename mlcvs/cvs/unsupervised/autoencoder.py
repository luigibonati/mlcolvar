from typing import Any
import torch
import pytorch_lightning as pl
from mlcvs.core.utils.decorators import decorate_methods, allowed_hooks, call_submodules_hooks
from mlcvs.core.models import FeedForward
from mlcvs.core.transform import Normalization

from mlcvs.cvs.utils import CV_utils

@decorate_methods(call_submodules_hooks, methods=allowed_hooks)
class AutoEncoder_CV(pl.LightningModule, CV_utils):
    """AutoEncoding Collective Variable."""
    
    def __init__(self, encoder_layers : list , decoder_layers : list = None, options : dict[str,Any] = {}, **kwargs ):
        """
        Train a CV defined as the output layer of the encoder of an autoencoder model

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
        super().__init__()

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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return self.optimizer

    def loss_function(self, input, target):
        # Reconstruction (MSE) loss
        loss = (input-target).square().mean()
        return loss

    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        x_hat = self.encode_decode(x)
        loss = self.loss_function(x_hat,x)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch[0]
        x_hat = self.encode_decode(x)
        loss = self.loss_function(x_hat,x)
        self.log('val_loss', loss, on_epoch=True)

def test_autoencodercv():
    n_in, n_out = 8,2
    layers = [n_in, 6, 4, n_out]

    # initialize via dictionary
    opts = { 'normIn'  : None,
             'encoder' : { 'activation' : 'relu' },
             'normOut' : { 'mode'   : 'std' },
           } 
    model = AutoEncoder_CV( encoder_layers=layers, options=opts )
    print(model)
    print('----------')

if __name__ == "__main__":
    test_autoencodercv() 