import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization, BaseGNN
from mlcolvar.core.loss import MSELoss
from typing import Union, List


__all__ = ["RegressionCV"]


class RegressionCV(BaseCV):
    """
    Example of collective variable obtained with a regression task.
    Combine the inputs with a neural-network and optimize it to match a target function.

    **Data**: for training it requires a DictDataset containing:
        - If using descriptors as input, the keys 'data', 'target' and optionally 'weights'.
        - If using graphs as input, `torch_geometric.data` with either 'graph_labels' (graph-level)
          or 'node_labels' (node-level) as regression target, and optionally 'weight' in 'data_list'.

    **Loss**: least squares (MSELoss).

    See also
    --------
    mlcolvar.core.loss.MSELoss
        (weighted) Mean Squared Error (MSE) loss function.
    """

    DEFAULT_BLOCKS = ["norm_in", "nn"]
    MODEL_BLOCKS = ["nn"]

    def __init__(
        self,
        model: Union[List[int], FeedForward, BaseGNN],
        options: dict = None,
        graph_target_key: str = "graph_labels",
        **kwargs,
    ):
        """Example of collective variable obtained with a regression task.
        By default a module standardizing the inputs is used.

        Parameters
        ----------
        model : list or FeedForward or BaseGNN
            Determines the underlying machine-learning model. One can pass:
            1. A list of integers corresponding to the number of neurons per layer of a feed-forward NN.
               The model Will be automatically intialized using a `mlcolvar.core.nn.feedforward.FeedForward` object.
               The CV class will be initialized according to the DEFAULT_BLOCKS.
            2. An externally intialized model (either `mlcolvar.core.nn.feedforward.FeedForward` or `mlcolvar.core.nn.graph.BaseGNN` object).
               The CV class will be initialized according to the MODEL_BLOCKS.
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default None.
            Available blocks: ['norm_in', 'nn'].
            Set 'block_name' = None or False to turn off that block.
        graph_target_key : str, optional
            Graph regression target key, either 'graph_labels' or 'node_labels', by default 'graph_labels'.
            Only used when `model` is a `BaseGNN` and should match the model output level
            configured through `model.pooling_operation`.
        """
        super().__init__(model, **kwargs)

        allowed_graph_targets = {"graph_labels", "node_labels"}
        if graph_target_key not in allowed_graph_targets:
            raise ValueError(
                f"`graph_target_key` must be one of {allowed_graph_targets}, found '{graph_target_key}'."
            )
        self.graph_target_key = graph_target_key

        # =======   LOSS  =======
        self.loss_fn = MSELoss()

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

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

    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        # =================get data===================
        loss_kwargs = {}
        if isinstance(self.nn, FeedForward):
            x = train_batch["data"]
            labels = train_batch["target"]
            if "weights" in train_batch:
                loss_kwargs["weights"] = train_batch["weights"]
        elif isinstance(self.nn, BaseGNN):
            x = self._setup_graph_data(train_batch)
            if self.graph_target_key not in x:
                raise KeyError(
                    f"Missing '{self.graph_target_key}' in graph batch. Available keys: {list(x.keys())}"
                )
            labels = x[self.graph_target_key]
            if self.graph_target_key == "graph_labels" and "weight" in x:
                loss_kwargs["weights"] = x["weight"]

        # =================forward====================
        y = self.forward_cv(x)

        # Keep compatibility with scalar targets stored with an extra singleton dim.
        if y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        if labels.ndim > 1 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)

        # ===================loss=====================
        try:
            loss = self.loss_fn(y, labels, **loss_kwargs)
        except TypeError as e:
            if "unexpected keyword argument 'weights'" in str(e):
                loss = self.loss_fn(y, labels)
            else:
                raise
        # ====================log=====================
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        return loss


def test_regression_cv():
    """
    Create a synthetic dataset and test functionality of the RegressionCV class
    """
    from mlcolvar.data import DictDataset, DictModule

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
    from mlcolvar.core.nn.graph.schnet import SchNetModel
    from mlcolvar.data.graph.utils import create_test_graph_input
    gnn_model = SchNetModel(1, 0.1, [1, 8])
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
    gnn_model_node = SchNetModel(1, 0.1, [1, 8], pooling_operation=None)
    model = RegressionCV(model=gnn_model_node, graph_target_key="node_labels")
    trainer = lightning.Trainer(
        accelerator="cpu", max_epochs=1, logger=False, enable_checkpointing=False, enable_model_summary=False
    )
    trainer.fit(model, datamodule)
    model.eval()
    traced_model = model.to_torchscript(file_path=None, method="trace")
    example_input_graph_test = create_test_graph_input(output_type='example', n_atoms=4, n_samples=3, n_states=2)
    assert torch.allclose(model(example_input_graph_test), traced_model(example_input_graph_test))
