import torch 
import pytorch_lightning as pl
from typing import Any

from mlcvs.core import FeedForward, Normalization
from mlcvs.core import TDA_loss

from mlcvs.utils.data import TensorDataModule
from torch.utils.data import TensorDataset

from mlcvs.utils.decorators import decorate_methods,call_submodules_hooks,allowed_hooks

from mlcvs.cvs.utils import BaseCV

__all__ = ["DeepTDA_CV"]

@decorate_methods(call_submodules_hooks,methods=allowed_hooks)
class DeepTDA_CV(BaseCV, pl.LightningModule):
    """
    Define Deep Targeted Discriminant Analysis (Deep-TDA) CV.
    Combine the inputs with a neural-network and optimize it in a way such that the data are distributed accordingly to a target distribution.
    """

    BLOCKS = ['normIn', 'nn']

    def __init__(self,
                n_states : int,
                n_cvs : int,
                target_centers : list, 
                target_sigmas : list, 
                layers : list, 
                options : dict = {}, 
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

        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs)

        # Members
        options = self.initialize_block_defaults(options=options)
        
        self.n_states = n_states
        if self.out_features != n_cvs:
            raise ValueError("Number of neurons of last layer should match the number of CVs!")
        self.n_cvs = n_cvs
        
        # TODO pass to options
        self.target_centers = torch.tensor(target_centers)
        self.target_sigmas =  torch.tensor(target_sigmas)

        if self.target_centers.shape != self.target_sigmas.shape:
            raise ValueError(f"Size of target_centers and target_sigmas should be the same!")
        if n_states != self.target_centers.shape[0]:
            raise ValueError(f"Size of target_centers at dimension 0 should match the number of states! Expected {n_states} found {self.target_centers.shape[0]}")
        if len(self.target_centers.shape) == 2:
            if n_cvs != self.target_centers.shape[1]:
                raise ValueError((f"Size of target_centers at dimension 1 should match the number of cvs! Expected {n_cvs} found {self.target_centers.shape[1]}"))

        # Initialize normIn
        o = 'normIn'
        if ( not options[o] ) and (options[o] is not None):
            self.normIn = Normalization(self.in_features,**options[o])

        # initialize NN
        o = 'nn'
        self.nn = FeedForward(layers, **options[o])
    
    # TODO change to have standard signature?
    def loss_function(self, input, labels):
        loss, loss_centers, loss_sigmas = TDA_loss(input, labels, self.n_states, self.target_centers, self.target_sigmas)
        
        return loss, loss_centers, loss_sigmas

    def training_step(self, train_batch, batch_idx):
        # =================get data===================
        x = train_batch['data']
        labels = train_batch['labels']
        # =================forward====================
        z = self(x)
        # ===================loss=====================
        loss, loss_centers, loss_sigmas = self.loss_function(z,labels)
        # ====================log=====================+
        name = 'train' if self.training else 'valid'
        self.log(f'{name}_loss', loss, on_epoch=True)
        self.log(f'{name}_loss_centers', loss_centers, on_epoch=True)
        self.log(f'{name}_loss_sigmas', loss_sigmas, on_epoch=True)
        return loss

# TODO signature of tests?
import numpy as np
def test_deeptda_cv():
    from mlcvs.utils.data import DictionaryDataset

    for states_and_cvs in [  [2, 1], [3, 1], [3, 2], [5, 4] ]:
        # get the number of states and cvs for the test run
        n_states = states_and_cvs[0]
        n_cvs = states_and_cvs[1]
        
        in_features, out_features = 2, n_cvs 
        layers = [in_features, 4, 2, out_features]
        target_centers = np.random.randn(n_states, n_cvs)
        target_sigmas = np.random.randn(n_states, n_cvs)

        # test initialize via dictionary
        options= { 'nn' : { 'activation' : 'relu' } }

        model = DeepTDA_CV(n_states = n_states, n_cvs = n_cvs, target_centers = target_centers, target_sigmas = target_sigmas, layers = layers, options=options)
        
        print('----------')
        print(model)

        # create dataset
        samples = 50
        X = torch.randn((samples * n_states, 2))

        # create labels
        y = torch.zeros(X.shape[0])
        for i in range(1, n_states):
            y[samples*i:] += 1
        
        dataset = DictionaryDataset({'data': X, 'labels' : y})
        datamodule = TensorDataModule(dataset,lengths=[0.75,0.2,0.05], batch_size=samples)
        # train model
        trainer = pl.Trainer(accelerator='cpu', max_epochs=2, logger=None, enable_checkpointing=False)
        trainer.fit( model, datamodule )

        # trace model
        traced_model = model.to_torchscript(file_path=None, method='trace', example_inputs=X[0])
        model.eval()
        assert torch.allclose(model(X),traced_model(X))

if __name__ == "__main__":
    test_deeptda_cv()