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

        # Keep compatibility with scalar targets stored with extra singleton dims.
        y, labels = self._align_regression_tensors(y, labels)
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

    @staticmethod
    def _squeeze_trailing_singletons(x: torch.Tensor) -> torch.Tensor:
        """Remove trailing singleton dimensions, preserving multi-target tensors."""
        while x.ndim > 1 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x

    def _align_regression_tensors(
        self, y: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align prediction and target shapes before computing the regression loss."""
        y = self._squeeze_trailing_singletons(y)
        labels = self._squeeze_trailing_singletons(labels)

        if y.shape != labels.shape:
            raise ValueError(
                "Prediction/target shape mismatch in RegressionCV: "
                f"pred shape={tuple(y.shape)}, target shape={tuple(labels.shape)}. "
                "Check `graph_target_key`, pooling configuration, and label tensor shapes."
            )

        return y, labels
