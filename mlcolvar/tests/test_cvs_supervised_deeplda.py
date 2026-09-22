import lightning
import pytest
import torch

from mlcolvar.core.nn import FeedForward
from mlcolvar.core.nn.graph.schnet import SchNetModel
from mlcolvar.cvs.supervised.deeplda import DeepLDA
from mlcolvar.data import DictDataset, DictModule
from mlcolvar.data.graph.utils import create_test_graph_input


@pytest.mark.parametrize("n_states", [2, 3])
def test_deeplda(n_states):
    in_features, out_features = 2, n_states - 1
    layers = [in_features, 50, 50, out_features]
    
    # create dataset
    n_points = 500
    X, y = [], []
    for i in range(n_states):
        X.append(
            torch.randn(n_points, in_features) * (i + 1)
            + torch.Tensor([10 * i, (i - 1) * 10])
        )
        y.append(torch.ones(n_points) * i)

    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    dataset = DictDataset({"data": X, "labels": y})
    datamodule = DictModule(dataset, lengths=[0.8, 0.2], batch_size=n_states * n_points)

    # initialize CV
    opts = {
        "norm_in": {"mode": "mean_std"},
        "nn": {"activation": "relu"},
        "lda": {},
    }
    print()
    print('NORMAL')
    print()
    model = DeepLDA(layers, n_states, options=opts)

    # create trainer and fit
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False
    )
    trainer.fit(model, datamodule)

    # eval
    model.eval()
    with torch.no_grad():
        _ = model(X).numpy()

    
    # feedforward external
    print()
    print('EXTERNAL')
    print()
    ff_model = FeedForward(layers=layers)
    model = DeepLDA(ff_model, n_states)

    # create trainer and fit
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False
    )
    trainer.fit(model, datamodule)

    # eval
    model.eval()
    with torch.no_grad():
        s = model(X).numpy()
    print(s)

    # gnn external
    print()
    print('GNN')
    print()
    gnn_model = SchNetModel(n_out=2, cutoff=0.1, atomic_numbers=[1, 8])
    model = DeepLDA(gnn_model, n_states)

    datamodule = create_test_graph_input(output_type='datamodule', n_samples=200, n_states=n_states)

    # create trainer and fit
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=False, enable_checkpointing=False, enable_model_summary=False
    )
    trainer.fit(model, datamodule)

    traced_model = model.to_torchscript(
    file_path=None, method="trace")

    example_input_graph_test = create_test_graph_input(output_type='example', n_atoms=4, n_samples=3, n_states=n_states)
    assert torch.allclose(model(example_input_graph_test), traced_model(example_input_graph_test))

    # eval
    model.eval()
    with torch.no_grad():
        s = model(example_input_graph_test).numpy()
    print(s)