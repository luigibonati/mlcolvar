import torch
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, BaseGNN, Normalization
from mlcolvar.core.estimators import LDA
from mlcolvar.core.loss import ReduceEigenvaluesLoss
from typing import Union, List


__all__ = ["DeepLDA"]


class DeepLDA(BaseCV):
    """Deep Linear Discriminant Analysis (Deep-LDA) CV.
    Non-linear generalization of LDA in which a feature map is learned by a neural network optimized
    as to maximize the classes separation. The method is described in [1]_.

    **Data**: for training it requires a DictDataset containing:
        - If using descriptors as input, the keys 'data' and 'labels'
        - If using graphs as input, `torch_geometric.data` with 'graph_labels' in their 'data_list'.

    **Loss**: maximize LDA eigenvalues (ReduceEigenvaluesLoss)

    References
    ----------
    .. [1] L. Bonati, V. Rizzi, and M. Parrinello, "Data-driven collective variables for enhanced
        sampling", JPCL 11, 2998–3004 (2020).

    See also
    --------
    mlcolvar.core.estimators.LDA
        Linear Discriminant Analysis method
    mlcolvar.core.loss.ReduceEigenvalueLoss
        Eigenvalue reduction to a scalar quantity
    """

    DEFAULT_BLOCKS = ["norm_in", "nn", "lda"]
    MODEL_BLOCKS = ["nn", "lda"]

    def __init__(self, 
                 model: Union[List[int], FeedForward, BaseGNN],
                 n_states: int, 
                 options: dict = None, 
                 **kwargs):
        """
        Define a Deep Linear Discriminant Analysis (Deep-LDA) CV composed by a
        neural network module and a LDA object.
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
        n_states : int
            Number of states for the training
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['norm_in','nn','lda'] .
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(model=model, **kwargs)

        # =======   LOSS  =======
        # Maximize the sum of all the LDA eigenvalues.
        self.loss_fn = ReduceEigenvaluesLoss(mode="sum")

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # Save n_states
        self.n_states = n_states

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

        # initialize lda
        o = "lda"
        self.lda = LDA(self.nn.out_features, n_states, **options[o])

        # regularization
        self.lorentzian_reg = 40  # == 2/sw_reg, see set_regularization
        self.set_regularization(sw_reg=0.05)

    def forward_nn(self, x: torch.Tensor) -> torch.Tensor:
        if not self._override_model:
            if self.norm_in is not None:
                x = self._apply_module(self.norm_in, x)
        x = self._apply_module(self.nn, x)
        return x

    def set_regularization(self, sw_reg=0.05, lorentzian_reg=None):
        r"""
        Set magnitude of regularizations for the training:
        - add identity matrix multiplied by `sw_reg` to within scatter S_w.
        - add lorentzian regularization to NN outputs with magnitude `lorentzian_reg`

        If `lorentzian_reg` is None, set it equal to `2./sw_reg`.

        Parameters
        ----------
        sw_reg : float
            Regularization value for S_w.
        lorentzian_reg: float
            Regularization for lorentzian on NN outputs.

        Notes
        -----
        These regularizations are described in [1]_.

        - S_w
        .. math:: S_w = S_w + \mathtt{sw_reg}\ \mathbf{1}.

        - Lorentzian

        .. math:: \text{reg}_{lor}=\alpha \left( 1+( \mathbb{E}\left[||\mathbf{s}||^2\right]-1)^2 \right)^{-1}

        """
        self.lda.sw_reg = sw_reg
        if lorentzian_reg is None:
            self.lorentzian_reg = 2.0 / sw_reg
        else:
            self.lorentzian_reg = lorentzian_reg

    def regularization_lorentzian(self, x):
        """
        Compute lorentzian regularization on the CVs.

        Parameters
        ----------
        x : float
            input data
        """
        reg_loss = x.pow(2).sum().div(x.size(0))
        reg_loss_lor = -self.lorentzian_reg / (1 + (reg_loss - 1).pow(2))
        return reg_loss_lor

    def evaluate_loss(
        self,
        batch,
        batch_idx: int,
        update_state: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute the Deep-LDA loss and associated metrics."""
        if isinstance(self.nn, FeedForward):
            x = batch["data"]
            labels = batch["labels"]
        elif isinstance(self.nn, BaseGNN):
            x = self._setup_graph_data(batch)
            labels = x["graph_labels"].squeeze()

        h = self.forward_nn(x)
        eigvals, _ = self.lda.compute(h, labels, save_params=update_state)
        loss = self.loss_fn(eigvals)

        # Keep this metric defined when regularization is disabled.
        lorentzian_reg = torch.zeros_like(loss)
        if self.lorentzian_reg > 0:
            s = self.lda(h)
            lorentzian_reg = self.regularization_lorentzian(s)
            loss = loss + lorentzian_reg

        output = {"loss": loss, "lorentzian_reg": lorentzian_reg}
        output.update(
            {f"eigval_{i + 1}": eigval for i, eigval in enumerate(eigvals)}
        )
        return output
