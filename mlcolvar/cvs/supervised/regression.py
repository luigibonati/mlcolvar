import torch
import pytorch_lightning as pl
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.loss import MSELoss

__all__ = ["RegressionCV"]

class RegressionCV(BaseCV, pl.LightningModule):
    """
    Example of collective variable obtained with a regression task.
    Combine the inputs with a neural-network and optimize it to match a target function.

    For the training it requires a DictionaryDataset with the keys 'data' and 'target' and optionally 'weights'.
    MSE Loss is used to optimize it.
    """

    BLOCKS = ['norm_in', 'nn']

    def __init__(self, 
                layers : list, 
                options : dict = None,
                **kwargs):
        """Example of collective variable obtained with a regression task.

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default None.
            Available blocks: ['norm_in', 'nn'].
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs)

        # =======   LOSS  =======
        self.loss_fn = MSELoss()

        # ======= OPTIONS ======= 
        # parse and sanitize
        options = self.parse_options(options)

        # Initialize norm_in
        o = 'norm_in'
        if ( options[o] is not False ) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features,**options[o])

        # initialize NN
        o = 'nn'
        self.nn = FeedForward(layers, **options[o])

    def training_step(self, train_batch, batch_idx):
        # =================get data===================
        x = train_batch['data']
        labels = train_batch['target']
        loss_kwargs = {}
        if 'weights' in train_batch:
            loss_kwargs['weights'] = train_batch['weights']
        # =================forward====================
        y = self.forward_cv(x)
        # ===================loss=====================
        loss = self.loss_fn(y, labels, **loss_kwargs)
        # ====================log===================== 
        name = 'train' if self.training else 'valid'       
        self.log(f'{name}_loss', loss, on_epoch=True)
        return loss


def test_regression_cv():
    """
    Create a synthetic dataset and test functionality of the RegressionCV class
    """
    from mlcolvar.data import DictionaryDataset, DictionaryDataModule

    in_features, out_features = 2,1 
    layers = [in_features, 5, 10, out_features]

    # initialize via dictionary
    options= { 'nn' : { 'activation' : 'relu' } }

    model = RegressionCV( layers = layers,
                        options = options)
    print('----------')
    print(model)

    # create dataset
    X = torch.randn((100,2))
    y = X.square().sum(1)
    dataset = DictionaryDataset({'data':X,'target':y})
    datamodule = DictionaryDataModule(dataset,lengths=[0.75,0.2,0.05], batch_size=25)
    # train model
    model.optimizer_name ='SGD'
    model.optimizer_kwargs.update(dict(lr=1e-2))
    trainer = pl.Trainer(accelerator='cpu',max_epochs=1,logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )
    model.eval()
    # trace model
    traced_model = model.to_torchscript(file_path=None, method='trace', example_inputs=X[0])
    assert torch.allclose(model(X),traced_model(X))

    # weighted loss
    print('weighted loss') 
    w = torch.randn((100))
    dataset_weights = DictionaryDataset({'data':X, 'target':y, 'weights':w})
    datamodule_weights = DictionaryDataModule(dataset_weights, lengths=[0.75,0.2,0.05], batch_size=25)
    trainer.fit(model, datamodule_weights)
        
    # use custom loss
    print('custom loss')
    trainer = pl.Trainer(accelerator='cpu',max_epochs=1,logger=None, enable_checkpointing=False)

    model = RegressionCV( layers = [2,10,10,1])
    model.loss_fn = lambda y,y_ref: (y-y_ref).abs().mean() 
    trainer.fit( model, datamodule )

if __name__ == "__main__":
    test_regression_cv() 