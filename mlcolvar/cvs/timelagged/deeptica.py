import torch
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, BaseGNN, Normalization
from mlcolvar.core.estimators import TICA
from mlcolvar.core.loss import ReduceEigenvaluesLoss
from typing import Union, List

__all__ = ["DeepTICA"]


class DeepTICA(BaseCV):
    """
    Neural network-based time-lagged independent component analysis (DeepTICA).

    DeepTICA is a nonlinear generalization of TICA in which a feature map is
    learned by a neural network by maximizing the eigenvalues of the transfer
    operator approximated through TICA. The method is described in Ref. [1]_.
    From an architectural perspective, DeepTICA is closely related to the
    state-free reversible VAMPnet (SRV) approach [2]_.

    The model supports both descriptor-based and graph-based neural networks.

    Data
    ----
    For descriptor-based models, the training dataset should contain
    ``data`` and ``data_lag`` for configurations at times ``t`` and
    ``t + lag``, together with the corresponding ``weights`` and
    ``weights_lag``.

    For graph-based models, the dataset should contain ``data_list`` and
    ``data_list_lag``, with the corresponding graph weights.

    Time-lagged datasets can be constructed with
    ``mlcolvar.utils.timelagged.create_timelagged_dataset``.

    Loss
    ----
    DeepTICA maximizes the TICA eigenvalues using
    ``ReduceEigenvaluesLoss``.

    References
    ----------
    .. [1] L. Bonati, G. Piccini, and M. Parrinello.
       "Deep learning the slow modes for rare events sampling."
       Proceedings of the National Academy of Sciences 118,
       e2113533118 (2021).

    .. [2] W. Chen, H. Sidky, and A. L. Ferguson.
       "Nonlinear discovery of slow molecular modes using state-free
       reversible VAMPnets." Journal of Chemical Physics 150,
       214114 (2019).

    See Also
    --------
    mlcolvar.core.estimators.TICA
        Time-lagged independent component analysis.
    mlcolvar.core.loss.ReduceEigenvaluesLoss
        Reduce multiple eigenvalues to a scalar loss.
    mlcolvar.utils.timelagged.create_timelagged_dataset
        Create datasets of time-lagged configurations.
    """

    DEFAULT_BLOCKS = ["norm_in", "nn", "tica"]
    MODEL_BLOCKS = ["nn", "tica"]

    def __init__(self, 
                 model: Union[List[int], FeedForward, BaseGNN], 
                 n_cvs: int = None, 
                 options: dict = None, **kwargs):
        """
        Define a Deep-TICA CV, composed of a neural network module and a TICA object.
        By default a module standardizing the inputs is also used.

        Parameters
        ----------
        model : list or FeedForward or BaseGNN
            Determines the underlying machine-learning model. One can pass:
            1. A list of integers corresponding to the number of neurons per layer of a feed-forward NN.
               The model Will be automatically intialized using a `mlcolvar.core.nn.feedforward.FeedForward` object.
               The CV class will be initialized according to the DEFAULT_BLOCKS.
            2. An externally intialized model (either `mlcolvar.core.nn.feedforward.FeedForward` or `mlcolvar.core.nn.graph.BaseGNN` object).
               The CV class will be initialized according to the MODEL_BLOCKS.
        n_cvs : int, optional
            Number of cvs to optimize, default None (= last layer)
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['norm_in','nn','tica'].
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(model, **kwargs)

        # =======   LOSS  =======
        # Maximize the squared sum of all the TICA eigenvalues.
        self.loss_fn = ReduceEigenvaluesLoss(mode="sum2")
        # here we need to override the self.out_features attribute
        if n_cvs is None:
            n_cvs = self.out_features

        self.out_features = n_cvs
        self.n_cvs.fill_(n_cvs)
        
        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======

        if not self._override_model:
            # initialize norm_in
            o = "norm_in"
            if (options[o] is not False) and (options[o] is not None):
                self.norm_in = Normalization(self.in_features, **options[o])

            # initialize nn
            o = "nn"
            self.nn = FeedForward(self.layers, **options[o])
        
        elif self._override_model:
            self.nn = model
            
        # initialize tica
        o = "tica"
        self.tica = TICA(self.nn.out_features, n_cvs, **options[o])

    def forward_nn(self, x: torch.Tensor) -> torch.Tensor:
        if not self._override_model:
            if self.norm_in is not None:
                x = self._apply_module(self.norm_in, x)
        x = self._apply_module(self.nn, x)
        return x

    def set_regularization(self, c0_reg=1e-6):
        """
        Add identity matrix multiplied by `c0_reg` to correlation matrix C(0) to avoid instabilities in performin Cholesky and .

        Parameters
        ----------
        c0_reg : float
            Regularization value for C_0.
        """
        self.tica.reg_C_0 = c0_reg

    def evaluate_loss(
        self,
        batch,
        batch_idx: int,
        update_state: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute and return the training loss and record metrics.
        1) Calculate the NN output
        2) Remove average (inside forward_nn)
        3) Compute TICA
        """
        # ================= get data =================
        if isinstance(self.nn, FeedForward):
            x_t = batch["data"]
            x_lag = batch["data_lag"]
            w_t = batch["weights"]
            w_lag = batch["weights_lag"]
        elif isinstance(self.nn, BaseGNN):
            x_t = self._setup_graph_data(
                batch,
                key="data_list",
            )
            x_lag = self._setup_graph_data(
                batch,
                key="data_list_lag",
            )
            w_t = x_t["weight"]
            w_lag = x_lag["weight"]

        # ================= forward ==================
        f_t = self.forward_nn(x_t)
        f_lag = self.forward_nn(x_lag)

        # ================== TICA ====================
        eigvals, _ = self.tica.compute(
            data=[f_t, f_lag],
            weights=[w_t, w_lag],
            save_params=update_state,
        )

        # ================== loss ====================
        loss = self.loss_fn(eigvals)

        # ================= metrics ==================
        output = {
            "loss": loss,
        }
        output.update(
            {
                f"eigval_{i + 1}": eigval
                for i, eigval in enumerate(eigvals)
            }
        )
        return output
