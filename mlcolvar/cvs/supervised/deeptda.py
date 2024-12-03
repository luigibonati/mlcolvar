import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, BaseGNN, Normalization
from mlcolvar.core.loss import TDALoss
from mlcolvar.data import DictModule
from typing import Union, List

__all__ = ["DeepTDA"]

class DeepTDA(BaseCV, lightning.LightningModule):
    """
    Deep Targeted Discriminant Analysis (Deep-TDA) CV.
    Combine the inputs with a neural-network and optimize it in a way such that
    the data are distributed accordingly to a mixture of Gaussians. The method is described in [1]_.
    **Data**: for training it requires a DictDataset with the keys 'data' and 'labels'.
    **Loss**: distance of the samples of each class from a set of Gaussians (TDALoss)
    References
    ----------
    .. [1] E. Trizio and M. Parrinello, "From enhanced sampling to reaction profiles",
        The Journal of Physical Chemistry Letters 12, 8621– 8626 (2021).
    See also
    --------
    mlcolvar.core.loss.TDALoss
        Distance from a simple Gaussian target distribution.
    """

    DEFAULT_BLOCKS = ["norm_in", "nn"]
    MODEL_BLOCKS = ["nn"]


    # TODO n_states optional?
    def __init__(
        self,
        n_states: int,
        n_cvs: int,
        target_centers: list,
        target_sigmas: list,
        model: Union[List[int], FeedForward, BaseGNN],
        options: dict = None,
        **kwargs,
    ):
        """
        Define Deep Targeted Discriminant Analysis (Deep-TDA) CV composed by a neural network module.
        By default a module standardizing the inputs is also used.
        Parameters
        ----------
        n_states : int
            Number of states for the training
        n_cvs : int
            Number of collective variables to be trained
        target_centers : list
            Centers of the Gaussian targets
        target_sigmas : list
            Standard deviations of the Gaussian targets
        layers : list
            Number of neurons per layer
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['norm_in', 'nn'].
            Set 'block_name' = None or False to turn off that block
        """        
        super().__init__(model, **kwargs)
        self.save_hyperparameters(ignore=['model'])

        # =======   LOSS  =======
        self.loss_fn = TDALoss(
            n_states=n_states,
            target_centers=target_centers,
            target_sigmas=target_sigmas,
        )

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)
        # Save n_states
        self.n_states = n_states
        if self.out_features != n_cvs:
            raise ValueError(
                "Number of neurons of last layer should match the number of CVs!"
            )

        # check size  and type of targets
        if not isinstance(target_centers, torch.Tensor):
            target_centers = torch.Tensor(target_centers)
        if not isinstance(target_sigmas, torch.Tensor):
            target_sigmas = torch.Tensor(target_sigmas)

        if target_centers.shape != target_sigmas.shape:
            raise ValueError(
                f"Size of target_centers and target_sigmas should be the same!"
            )
        if n_states != target_centers.shape[0]:
            raise ValueError(
                f"Size of target_centers at dimension 0 should match the number of states! Expected {n_states} found {target_centers.shape[0]}"
            )
        if len(target_centers.shape) == 2:
            if n_cvs != target_centers.shape[1]:
                raise ValueError(
                    (
                        f"Size of target_centers at dimension 1 should match the number of cvs! Expected {n_cvs} found {target_centers.shape[1]}"
                    )
                )

        # ======= BLOCKS =======
        if not self._override_model: 
            # Initialize norm_in
            o = "norm_in"
            if (options[o] is not False) and (options[o] is not None):
                self.norm_in = Normalization(self.in_features, **options[o])

            # initialize NN
            o = "nn"
            self.nn = FeedForward(self.layers, **options[o])
        elif self._override_model:
            self.nn = model

    def training_step(self, train_batch, *args, **kwargs) -> torch.Tensor:
        """Compute and return the training loss and record metrics."""
        if isinstance(self.nn, FeedForward):
            # =================get data===================
            x = train_batch["data"]
            labels = train_batch["labels"]
            # =================forward====================
            z = self.forward_cv(x)

        elif isinstance(self.nn, BaseGNN):
            # =================get data===================
            data = self._setup_graph_data(train_batch)
            labels = data['graph_labels'].squeeze()
            # =================forward====================
            z = self.forward(data)

        # ===================loss=====================
        loss, loss_centers, loss_sigmas = self.loss_fn(z, 
                                                        labels, 
                                                        return_loss_terms=True
                                                        )
        
        # ====================log=====================
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_centers", loss_centers, on_epoch=True)
        self.log(f"{name}_loss_sigmas", loss_sigmas, on_epoch=True)

        return loss

# TODO signature of tests?
import numpy as np


def test_deeptda_cv():
    from mlcolvar.data import DictDataset

    # feedforward with layers
    for states_and_cvs in [[2, 1], [3, 1], [3, 2], [5, 4]]:
        print(states_and_cvs)
        # get the number of states and cvs for the test run
        n_states = states_and_cvs[0]
        n_cvs = states_and_cvs[1]

        in_features, out_features = 2, n_cvs
        layers = [in_features, 4, 2, out_features]
        target_centers = np.random.randn(n_states, n_cvs)
        target_sigmas = np.random.randn(n_states, n_cvs)

        # test initialize via dictionary
        options = {"nn": {"activation": "relu"}}

        print()
        print('NORMAL')
        print()
        model = DeepTDA(
            n_states=n_states,
            n_cvs=n_cvs,
            target_centers=target_centers,
            target_sigmas=target_sigmas,
            model=layers,
            options=options,
        )

        # create dataset
        samples = 100
        X = torch.randn((samples * n_states, 2))

        # create labels
        y = torch.zeros(X.shape[0])
        for i in range(1, n_states):
            y[samples * i :] += 1

        dataset = DictDataset({"data": X, "labels": y})
        datamodule = DictModule(dataset, lengths=[0.75, 0.2, 0.05], batch_size=samples)
        # train model
        trainer = lightning.Trainer(
            accelerator="cpu", max_epochs=2, logger=None, enable_checkpointing=False,  enable_model_summary=False
        )
        trainer.fit(model, datamodule)

        # trace model
        traced_model = model.to_torchscript(
            file_path=None, method="trace")
        model.eval()
        assert torch.allclose(model(X), traced_model(X))

        print()
        print('EXTERNAL FEEDFORWARD')
        print()
        # feedforward external
        ff_model = FeedForward(layers=layers)
        model = DeepTDA(
            n_states=n_states,
            n_cvs=n_cvs,
            target_centers=target_centers,
            target_sigmas=target_sigmas,
            model=ff_model
        )

        # train model
        trainer = lightning.Trainer(
            accelerator="cpu", max_epochs=2, logger=None, enable_checkpointing=False, enable_model_summary=False
        )
        trainer.fit(model, datamodule)

        # trace model
        traced_model = model.to_torchscript(
            file_path=None, method="trace")
        model.eval()
        assert torch.allclose(model(X), traced_model(X))

        print()
        print('EXTERNAL GNN')
        print()
        # gnn external 
        from mlcolvar.core.nn.graph.schnet import SchNetModel
        from mlcolvar.data.graph.utils import create_test_graph_input
        gnn_model = SchNetModel(n_cvs, 0.1, [1, 8])
        model = DeepTDA(
            n_states=n_states,
            n_cvs=n_cvs,
            target_centers=target_centers,
            target_sigmas=target_sigmas,
            model=gnn_model
        )
        datamodule = create_test_graph_input(output_type='datamodule', n_samples=100, n_states=n_states)

        # train model
        trainer = lightning.Trainer(
            accelerator="cpu", max_epochs=2, logger=False, enable_checkpointing=False, enable_model_summary=False
        )
        trainer.fit(model, datamodule)

        # trace model
        traced_model = model.to_torchscript(
            file_path=None, method="trace")
        
        # check on a different number of atoms
        example_input_graph_test = create_test_graph_input(output_type='example', n_atoms=4, n_samples=3, n_states=2)
        assert torch.allclose(model(example_input_graph_test), traced_model(example_input_graph_test))

if __name__ == "__main__":
    test_deeptda_cv()

