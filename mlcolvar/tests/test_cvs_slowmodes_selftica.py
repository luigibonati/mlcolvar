import lightning
import numpy as np
import torch

from mlcolvar.core.nn.graph.schnet import SchNetModel
from mlcolvar.cvs.timelagged.selftica import SelfTICA
from mlcolvar.data import DictModule
from mlcolvar.data.graph.utils import create_test_graph_input
from mlcolvar.tests import data_dir
from mlcolvar.utils.timelagged import create_timelagged_dataset


def test_self_tica():
    # loss modes
    loss_modes = ["l2", "kl_DV", "kl_NWJ"]

    with data_dir() as data_folder:
        X = np.loadtxt(data_folder / "mb-mcmc.dat")

    X = torch.Tensor(X)

    for mode in loss_modes:
        print("FNN")
        print(f"\nTesting loss mode: {mode}")

        # create dataset
        dataset = create_timelagged_dataset(X, lag_time=1)
        datamodule = DictModule(dataset, batch_size=10000)

        # create model
        layers = [2, 10, 10, 2]
        model = SelfTICA(layers, n_cvs=1)

        # change loss
        model.loss_fn.mode = mode

        trainer = lightning.Trainer(
            max_epochs=1,
            log_every_n_steps=2,
            logger=None,
            enable_checkpointing=False,
        )

        trainer.fit(model, datamodule)

        # test TICA computation
        datamodule.setup()

        eigvals, eigvecs = model.compute_tica(datamodule, lag_time=10, update_optimal=True)
        print("TICA eigenvalues:", eigvals)

        # trace model
        traced_model = model.to_torchscript(
            file_path=None,
            method="trace",
        )

        model.eval()
        assert torch.allclose(model(X), traced_model(X), atol=1e-6)

    # gnn external
    print()
    print('GNN')
    print()

    gnn_model = SchNetModel(n_out=2, cutoff=0.1, atomic_numbers=[1, 8])
    model = SelfTICA(gnn_model, n_cvs=1)

    # change loss options
    model.loss_fn.mode = "l2"

    # create trainer and fit
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=False, enable_checkpointing=False, enable_model_summary=False,
    )

    dataset = create_test_graph_input(output_type='dataset', n_samples=200, n_states=2)
    lagged_dataset = create_timelagged_dataset(dataset, logweights=torch.randn(len(dataset)))
    
    datamodule = DictModule(dataset=lagged_dataset)
    trainer.fit(model, datamodule)

    model.eval()

    # test TICA computation
    datamodule.setup()

    eigvals, eigvecs = model.compute_tica(datamodule, lag_time=10, update_optimal=True)
    print("TICA eigenvalues:", eigvals)

    example_input_graph_test = create_test_graph_input(output_type='example', n_atoms=4, n_samples=3, n_states=2)
    traced_model = model.to_torchscript(
        file_path=None,
        method="trace",
    )
    assert torch.allclose(model(example_input_graph_test), traced_model(example_input_graph_test), atol=1e-6)