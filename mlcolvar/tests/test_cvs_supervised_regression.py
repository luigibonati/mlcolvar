import lightning
import torch

from mlcolvar.core.nn import FeedForward
from mlcolvar.core.nn.graph.schnet import SchNetModel
from mlcolvar.cvs.supervised.regression import RegressionCV
from mlcolvar.data import DictDataset, DictModule
from mlcolvar.data.graph.utils import create_test_graph_input


def test_regression_cv():
    """
    Create a synthetic dataset and test functionality of the RegressionCV class
    """
    in_features, out_features = 2, 1
    layers = [in_features, 5, 10, out_features]

    print()
    print('NORMAL')
    print()
    # initialize via dictionary
    options = {"nn": {"activation": "relu"}}

    model = RegressionCV(model=layers, options=options)
    print("----------")
    print(model)

    # create dataset
    X = torch.randn((100, 2))
    y = X.square().sum(1)
    dataset = DictDataset({"data": X, "target": y})
    datamodule = DictModule(dataset, lengths=[0.75, 0.2, 0.05], batch_size=25)
    # train model
    model.optimizer_name = "SGD"
    model.optimizer_kwargs.update(dict(lr=1e-2))
    trainer = lightning.Trainer(
        accelerator="cpu", max_epochs=1, logger=None, enable_checkpointing=False
    )
    trainer.fit(model, datamodule)
    model.eval()
    # trace model
    traced_model = model.to_torchscript(
        file_path=None, method="trace", example_inputs=X[0]
    )
    assert torch.allclose(model(X), traced_model(X))

    # weighted loss
    print("weighted loss")
    w = torch.randn((100))
    dataset_weights = DictDataset({"data": X, "target": y, "weights": w})
    datamodule_weights = DictModule(
        dataset_weights, lengths=[0.75, 0.2, 0.05], batch_size=25
    )
    trainer.fit(model, datamodule_weights)

    # use custom loss
    print("custom loss")
    trainer = lightning.Trainer(
        accelerator="cpu", max_epochs=1, logger=None, enable_checkpointing=False
    )

    model = RegressionCV(model=[2, 10, 10, 1])
    model.loss_fn = lambda y, y_ref: (y - y_ref).abs().mean()
    trainer.fit(model, datamodule)

    print()
    print('EXTERNAL FEEDFORWARD')
    print()
    ff_model = FeedForward(layers=layers)
    # create model
    model = RegressionCV(model=ff_model)

    # create dataset
    X = torch.randn((100, 2))
    y = X.square().sum(1)
    dataset = DictDataset({"data": X, "target": y})
    datamodule = DictModule(dataset, lengths=[0.75, 0.2, 0.05], batch_size=25)
    # train model
    model.optimizer_name = "SGD"
    model.optimizer_kwargs.update(dict(lr=1e-2))
    trainer = lightning.Trainer(
        accelerator="cpu", max_epochs=1, logger=None, enable_checkpointing=False
    )
    trainer.fit(model, datamodule)
    model.eval()
    # trace model
    traced_model = model.to_torchscript(
        file_path=None, method="trace", example_inputs=X[0]
    )
    assert torch.allclose(model(X), traced_model(X))

    # weighted loss
    print("weighted loss")
    w = torch.randn((100))
    dataset_weights = DictDataset({"data": X, "target": y, "weights": w})
    datamodule_weights = DictModule(
        dataset_weights, lengths=[0.75, 0.2, 0.05], batch_size=25
    )
    trainer.fit(model, datamodule_weights)

    # use custom loss
    print("custom loss")
    trainer = lightning.Trainer(
        accelerator="cpu", max_epochs=1, logger=None, enable_checkpointing=False
    )

    model = RegressionCV(model=ff_model)
    model.loss_fn = lambda y, y_ref: (y - y_ref).abs().mean()
    trainer.fit(model, datamodule)

    print()
    print('EXTERNAL GNN')
    print()
    # gnn external
    gnn_model = SchNetModel(n_out=1, cutoff=0.1, atomic_numbers=[1, 8])
    # create model
    model = RegressionCV(model=gnn_model)

    datamodule = create_test_graph_input(output_type='datamodule', n_samples=100, n_states=2)
    # train model
    trainer = lightning.Trainer(
        accelerator="cpu", max_epochs=1, logger=False, enable_checkpointing=False, enable_model_summary=False
    )
    trainer.fit(model, datamodule)
    model.eval()
    # trace model
    traced_model = model.to_torchscript(file_path=None, method="trace")
    example_input_graph_test = create_test_graph_input(output_type='example', n_atoms=4, n_samples=3, n_states=2)
    assert torch.allclose(model(example_input_graph_test), traced_model(example_input_graph_test))

    # weighted loss
    print("weighted loss")
    datamodule_weights = create_test_graph_input(output_type='datamodule', n_samples=100, n_states=2, random_weights=True)
    trainer.fit(model, datamodule_weights)

    # use custom loss
    print("custom loss")
    trainer = lightning.Trainer(
        accelerator="cpu", max_epochs=1, logger=False, enable_checkpointing=False, enable_model_summary=False
    )

    model = RegressionCV(model=gnn_model)
    model.loss_fn = lambda y, y_ref: (y - y_ref).abs().mean()
    trainer.fit(model, datamodule)

    # node-level regression with GNN configured without pooling
    print("node-level")
    gnn_model_node = SchNetModel(n_out=1, cutoff=0.1, atomic_numbers=[1, 8], pooling_operation=None)
    model = RegressionCV(model=gnn_model_node, graph_target_key="node_labels")
    trainer = lightning.Trainer(
        accelerator="cpu", max_epochs=1, logger=False, enable_checkpointing=False, enable_model_summary=False
    )
    trainer.fit(model, datamodule)
    model.eval()
    traced_model = model.to_torchscript(file_path=None, method="trace")
    example_input_graph_test = create_test_graph_input(output_type='example', n_atoms=4, n_samples=3, n_states=2)
    assert torch.allclose(model(example_input_graph_test), traced_model(example_input_graph_test))
