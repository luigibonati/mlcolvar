import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization, NormalizationInverse
from mlcolvar.core.loss import MSELoss

__all__ = ["AutoEncoderCV"]

class AutoEncoderCV(BaseCV, lightning.LightningModule):
    """AutoEncoding Collective Variable. It is composed by a first neural network (encoder) which projects 
    the input data into a latent space (the CVs). Then a second network (decoder) takes 
    the CVs and tries to reconstruct the input data based on them. It is an unsupervised learning approach, 
    typically used when no labels are available.
    Furthermore, it can also be used lo learn a representation which can be used not to reconstruct the data but 
    to predict, e.g. future configurations. 

    For training it requires a DictDataset with the key 'data' and optionally 'weights'. If a 'target' 
    key is present this will be used as reference for the output of the decoder, otherway this will be compared
    with the input 'data'.
    """
    
    BLOCKS = ['norm_in','encoder','decoder'] 
    
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
            Available blocks: ['norm_in', 'encoder','decoder'].
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(in_features=encoder_layers[0], out_features=encoder_layers[-1], **kwargs)

        # =======   LOSS  =======
        # Reconstruction (MSE) loss
        self.loss_fn = MSELoss()

        # ======= OPTIONS ======= 
        # parse and sanitize
        options = self.parse_options(options)

        # if decoder is not given reverse the encoder
        if decoder_layers is None:
            decoder_layers = encoder_layers[::-1]

        # ======= BLOCKS =======

        # initialize norm_in
        o = 'norm_in'
        if ( options[o] is not False ) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features,**options[o]) 

        # initialize encoder
        o = 'encoder'
        self.encoder = FeedForward(encoder_layers, **options[o])

        # initialize encoder
        o = 'decoder'
        self.decoder = FeedForward(decoder_layers, **options[o])

    def forward_cv(self, x: torch.Tensor) -> (torch.Tensor):
        if self.norm_in is not None:
            x = self.norm_in(x)
        x = self.encoder(x)
        return x

    def encode_decode(self, x: torch.Tensor) -> (torch.Tensor):
        x = self.forward(x)
        x = self.decoder(x)
        if self.norm_in is not None:
            x = self.norm_in.inverse(x)
        return x

    def training_step(self, train_batch, batch_idx):
        # =================get data===================
        x = train_batch['data']
        loss_kwargs = {}
        if 'weights' in train_batch:
            loss_kwargs['weights'] = train_batch['weights']
        # =================forward====================
        x_hat = self.encode_decode(x)
        # ===================loss=====================
        # Reference output (compare with a 'target' key
        # if any, otherwise with input 'data')
        if 'target' in train_batch:
            x_ref = train_batch['target']
        else:
            x_ref = x 
        loss = self.loss_fn(x_hat, x_ref, **loss_kwargs)
        # ====================log=====================     
        name = 'train' if self.training else 'valid'       
        self.log(f'{name}_loss', loss, on_epoch=True)
        return loss
    
    def return_decoder_model(self):
        """Return a torch model with the decoder and the normalization inverse"""
        if self.norm_in is not None:
            inv_norm = NormalizationInverse(norm=self.norm_in, in_features=self.in_features)
            decoder_model = torch.nn.Sequential(*[self.decoder, inv_norm])
        else:
            decoder_model = self.decoder
        return decoder_model

def test_autoencodercv():
    from mlcolvar.data import DictDataset, DictModule
    import numpy as np

    in_features, out_features = 8,2
    layers = [in_features, 6, 4, out_features]

    # initialize via dictionary
    opts = { #'norm_in'  : None,
             'encoder' : { 'activation' : 'relu' },
           } 
    model = AutoEncoderCV( encoder_layers=layers, options=opts )
    print(model)

    # train
    print('train 1 - no weights')
    X = torch.randn(100,in_features) 
    dataset = DictDataset({'data': X})
    datamodule = DictModule(dataset)
    trainer = lightning.Trainer(max_epochs=1, log_every_n_steps=2,logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )
    # model.eval()
    X_hat = model(X)

    # test export of decoder_model
    decoder_model = model.return_decoder_model()
    # print(model.encode_decode(X) - decoder_model(X_hat))

    # train with weights
    print('train 2 - weights')
    dataset = DictDataset({'data': torch.randn(100,in_features), 'weights' : np.arange(100) })
    datamodule = DictModule(dataset)
    trainer = lightning.Trainer(max_epochs=1, log_every_n_steps=2,logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )
    
    # train with different input and ouput 
    print('train 3 - timelagged')
    dataset = DictDataset({'data': torch.randn(100,in_features), 'target' : torch.randn(100,in_features) })
    datamodule = DictModule(dataset)
    trainer = lightning.Trainer(max_epochs=1, log_every_n_steps=2,logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )

if __name__ == "__main__":
    test_autoencodercv() 