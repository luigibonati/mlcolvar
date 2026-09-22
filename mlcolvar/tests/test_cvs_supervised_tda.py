import lightning
import numpy as np
import torch

from mlcolvar.core.nn import FeedForward
from mlcolvar.core.nn.graph.schnet import SchNetModel
from mlcolvar.cvs.supervised.deeptda import DeepTDA
from mlcolvar.data import DictDataset, DictModule
from mlcolvar.data.graph.utils import create_test_graph_input


def test_deeptda_cv():
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
        gnn_model = SchNetModel(n_out=n_cvs, cutoff=0.1, atomic_numbers=[1, 8])
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