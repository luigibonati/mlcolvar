from typing import Any
import torch
import pytorch_lightning as pl
from mlcvs.utils.decorators import decorate_methods, allowed_hooks, call_submodules_hooks
from mlcvs.core.nn import FeedForward
from mlcvs.core.transform import Normalization
from mlcvs.cvs.cv import BaseCV
from mlcvs.core.loss import MSE_loss

__all__ = ["AutoEncoder_CV"]

@decorate_methods(call_submodules_hooks, methods=allowed_hooks)
class AutoEncoder_CV(BaseCV, pl.LightningModule):
    """AutoEncoding Collective Variable.
    
    For training it requires a DictionaryDataset with the key 'data' and optionally 'weights'.
    """
    
    BLOCKS = ['normIn','encoder','decoder'] 
    
    def __init__(self,
                encoder_layers : list, 
                decoder_layers : list = None, 
                options : dict = None, 
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
            Options for the building blocks of the model, by default None.
            Available blocks: ['normIn', 'encoder','decoder'].
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(in_features=encoder_layers[0], out_features=encoder_layers[-1], **kwargs)

        # ===== BLOCKS =====

        options = self.sanitize_options(options)

        # parse info from args
        if decoder_layers is None:
            decoder_layers = encoder_layers[::-1]

        # initialize normIn
        o = 'normIn'
        if ( options[o] is not False ) and (options[o] is not None):
            self.normIn = Normalization(self.in_features,**options[o]) 

        # initialize encoder
        o = 'encoder'
        self.encoder = FeedForward(encoder_layers, **options[o])

        # initialize encoder
        o = 'decoder'
        self.decoder = FeedForward(decoder_layers, **options[o])

        # ===== LOSS OPTIONS =====
        self.loss_options = {}   

    def forward_blocks(self, x: torch.tensor) -> (torch.tensor):
        if self.normIn is not None:
            x = self.normIn(x)
        x = self.encoder(x)
        return x

    def encode_decode(self, x: torch.tensor) -> (torch.tensor):
        x = self.forward(x)
        x = self.decoder(x)
        if self.normIn is not None:
            x = self.normIn.inverse(x)
        return x

    def loss_function(self, diff, **kwargs):
        # Reconstruction (MSE) loss
        return MSE_loss(diff,**kwargs)

    def training_step(self, train_batch, batch_idx):
        options = self.loss_options
        # =================get data===================
        x = train_batch['data']
        if 'weights' in train_batch:
            options['weights'] = train_batch['weights'] 
        # =================forward====================
        x_hat = self.encode_decode(x)
        # ===================loss=====================
        diff = x - x_hat
        loss = self.loss_function(diff, **options)
        # ====================log=====================     
        name = 'train' if self.training else 'valid'       
        self.log(f'{name}_loss', loss, on_epoch=True)
        return loss

def test_autoencodercv():
    from mlcvs.data import DictionaryDataset, DictionaryDataModule
    import numpy as np

    in_features, out_features = 8,2
    layers = [in_features, 6, 4, out_features]

    # initialize via dictionary
    opts = { 'normIn'  : None,
             'encoder' : { 'activation' : 'relu' },
           } 
    model = AutoEncoder_CV( encoder_layers=layers, options=opts )
    print(model)

    # train
    print('train 1 - no weights')
    X = torch.randn(100,in_features) 
    dataset = DictionaryDataset({'data': X})
    datamodule = DictionaryDataModule(dataset)
    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=2,logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )
    model.eval()
    X_hat = model(X)

    # train with weights
    print('train 2 - weights')
    dataset = DictionaryDataset({'data': torch.randn(100,in_features), 'weights' : np.arange(100) })
    datamodule = DictionaryDataModule(dataset)
    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=2,logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )
    

if __name__ == "__main__":
    test_autoencodercv() 