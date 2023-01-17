from typing import Any,Union
import torch
import pytorch_lightning as pl
from mlcvs.core.utils.decorators import decorate_methods, allowed_hooks, call_submodules_hooks
from mlcvs.core.models import FeedForward
from mlcvs.core.transform import Normalization

@decorate_methods(call_submodules_hooks, methods=allowed_hooks)
class AutoEncoderCV(pl.LightningModule):
    """AutoEncoding Collective Variable."""
    
    def __init__(self, encoder_layers : list , decoder_layers : list = None, options : dict[str,Any] = {} ):
        """TODO 

        """
        super().__init__()

        # Members
        blocks = ['normIn','encoder','normOut','decoder'] 

        # Initialize defaults
        for b in blocks:
            self.__setattr__(b,None)
            options.setdefault(b,{})

        # parse info from args
        if decoder_layers is None:
            decoder_layers = encoder_layers[::-1]
        self.n_in = encoder_layers[0]
        self.n_out = encoder_layers[-1]

        # initialize normIn
        o = 'normIn'
        if ( not options[o] ) and (options[o] is not None):
            self.normIn = Normalization(self.n_in,**options[o]) 

        # initialize encoder
        o = 'encoder'
        self.encoder = FeedForward(encoder_layers, **options[o])

         # initialize normOut
        o = 'normOut'
        if ( not options[o] ) and (options[o] is not None):
            self.normOut = Normalization(self.n_out,**options[o]) 

        # initialize encoder
        o = 'decoder'
        self.decoder = FeedForward(decoder_layers, **options[o])

        # set input example
        self.example_input_array = torch.ones(self.n_in)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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
    model = AutoEncoderCV( encoder_layers=layers, options=opts )
    print(model)
    print('----------')

if __name__ == "__main__":
    test_autoencodercv() 