import torch 
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, BaseGNN, Normalization
from mlcolvar.core.estimators import TICA
from mlcolvar.core.loss import ContrastiveLoss
from typing import Union, List

__all__ = ["SelfTICA"]
    

class SelfTICA(BaseCV):
    """
    Self-supervised time-lagged independent component analysis (SelfTICA).

    SelfTICA learns dynamical representations from time-lagged configurations
    using contrastive learning and subsequently applies TICA to the learned
    latent space to extract slow collective variables.

    The model supports both descriptor-based and graph-based encoders.

    Data
    ----
    For descriptor-based models, the training dataset should contain
    ``data`` and ``data_lag`` for configurations at times ``t`` and
    ``t + lag``, together with the corresponding ``weights`` and
    ``weights_lag``.

    For graph-based models, the dataset should contain ``data_list`` and
    ``data_list_lag`` with the corresponding graph weights.

    Time-lagged datasets can be constructed with
    ``mlcolvar.utils.timelagged.create_timelagged_dataset``.

    Loss
    ----
    SelfTICA uses ``ContrastiveLoss`` to encourage temporal consistency and
    decorrelation of the learned representations. The contrastive objective is
    related to the VAMP-2 score.

    References
    ----------
    Zhu, K., Zhang, J., Novelli, P., Hou, T., & Bonati, L.
    "Contrastive Learning of Dynamical Representations for Enhanced
    Molecular Sampling." arXiv:2606.15495 (2026).

    Turri, G., Bonati, L., Zhu, K., Pontil, M., & Novelli, P.
    "Self-Supervised Evolution Operator Learning for High-Dimensional
    Dynamical Systems." International Conference on Learning
    Representations (ICLR), 2026.

    See Also
    --------
    mlcolvar.core.estimators.TICA
        Time-lagged independent component analysis.
    mlcolvar.core.loss.ContrastiveLoss
        Contrastive loss for learning time-lagged representations.
    mlcolvar.utils.timelagged.create_timelagged_dataset
        Create datasets of time-lagged configurations.
    """

    DEFAULT_BLOCKS = ["norm_in", "nn", "predictor", "tica"]
    MODEL_BLOCKS = ["nn", "predictor", "tica"]

    def __init__(
        self,
        model: Union[List[int], FeedForward, BaseGNN],
        n_cvs: int = 1,
        regularization: float = 1e-5,
        predictor_depth: int = 2,
        options: dict = None,
        **kwargs,
    ):
        """
        Initialize a SelfTICA model.

        Parameters
        ----------
        model : list[int] or FeedForward or BaseGNN
            Neural-network architecture used as the encoder. If a list is
            provided, it defines the layer sizes of a ``FeedForward`` model.
            A ``FeedForward`` or ``BaseGNN`` instance can also be provided
            directly.
        n_cvs : int, optional
            Number of collective variables, by default 1.
        regularization : float, optional
            L2 regularization strength used in the contrastive loss,
            by default 1e-5.
        predictor_depth : int, optional
            Number of layers in the predictor. A value of 2 corresponds to a
            linear predictor, ``FeedForward([d, d])``, where ``d`` is the
            latent dimension. Larger values add hidden layers of width ``d``,
            by default 2.
        options : dict, optional
            Options for the model blocks. Available blocks are ``norm_in``,
            ``nn``, ``predictor``, and ``tica``. Set a block to ``None`` or
            ``False`` to disable it where supported.
        **kwargs
            Additional keyword arguments passed to ``BaseCV``.
        """
        super().__init__(model, **kwargs)      
        
        # encoder output dimension
        out_dim = int(self.out_features)  

        # =======   LOSS  =======
        self.loss_fn = ContrastiveLoss(reg=regularization, mode="l2")

        # check n_cvs
        if not isinstance(n_cvs, int) or n_cvs < 1:
            raise ValueError("n_cvs must be a positive integer (>= 1)")

        # final CV dimension
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

        # initalize predictor
        o = "predictor"
        
        if not isinstance(predictor_depth, int) or predictor_depth < 2:
            raise ValueError("predictor_depth must be an integer greater than or equal to 2.")
        
        pred_layers = [out_dim] * predictor_depth
        self.predictor = FeedForward(
           layers=pred_layers,
           **options[o]
        )

        # initialize TICA
        o = "tica"
        self.tica = TICA(out_dim, n_cvs, **options[o])

        self.register_buffer('current_evecs', torch.eye(out_dim, n_cvs))
        self.register_buffer('current_means', torch.zeros(out_dim))
        self.register_buffer('optimal_lag_time', torch.tensor(-1.0))

    def compute_tica(
        self,
        datamodule: lightning.LightningDataModule,
        lag_time: float = None,
        update_optimal: bool = False,
    ):
        """
        Compute TICA components from the learned representations.

        Parameters
        ----------
        datamodule : lightning.LightningDataModule
            Data module containing time-lagged configuration pairs.
        lag_time : float, optional
            Lag time associated with the analysis.
        update_optimal : bool, optional
            Whether to store the computed TICA parameters for inference,
            by default False.

        Returns
        -------
        eigenvalues : numpy.ndarray
            TICA eigenvalues.
        eigenvectors : numpy.ndarray
            TICA eigenvectors.
        """
        self.eval()

        dataloader = datamodule.train_dataloader()

        # Collect all time-lagged representations
        f_t_list, f_lag_list = [], []
        w_t_list, w_lag_list = [], []

        with torch.no_grad():
            for batch in dataloader:

                # ===== Process batch =====
                if isinstance(self.nn, FeedForward):
                    x_t = batch["data"]
                    x_lag = batch["data_lag"]
                    w_t_batch = batch["weights"]
                    w_lag_batch = batch["weights_lag"]

                elif isinstance(self.nn, BaseGNN):
                    x_t = self._setup_graph_data(batch, key='data_list')
                    x_lag = self._setup_graph_data(batch, key='data_list_lag')
                    w_t_batch = x_t['weight']
                    w_lag_batch = x_lag['weight']

                # ===== Compute representations =====
                f_t_list.append(self.forward_nn(x_t))
                f_lag_list.append(self.forward_nn(x_lag))

                # ===== Append weights =====
                w_t_list.append(w_t_batch)
                w_lag_list.append(w_lag_batch)

        # ===== Concatenate =====
        f_t_all = torch.cat(f_t_list)
        f_lag_all = torch.cat(f_lag_list)
        w_t_all = torch.cat(w_t_list)
        w_lag_all = torch.cat(w_lag_list)

        # Compute TICA
        eigvals, eigvecs = self.tica.compute(
            data=[f_t_all,f_lag_all],
            weights=[w_t_all, w_lag_all],
            save_params=update_optimal
        )

        if update_optimal:
            self._update_tica_params(lag_time)
        
        return eigvals.cpu().numpy(), eigvecs.cpu().numpy()
    
    def _update_tica_params(self, lag_time):
        """Update optimal TICA parameters for inference."""
        self.current_evecs.copy_(self.tica.evecs)
        self.current_means.copy_(self.tica.mean)
        if lag_time is not None:
            self.optimal_lag_time.fill_(lag_time)

    def forward(self, x: torch.Tensor, cell=None) -> torch.Tensor:

        if self.preprocessing is not None:
            x = self._apply_module(self.preprocessing, x, cell=cell)

        # Encode input into latent representation
        x = self.forward_nn(x)

        # In evaluation mode, apply TICA projection to obtain CVs
        if not self.training:
            centered = x - self.current_means
            x = centered @ self.current_evecs[:, :self.out_features]
        
        if self.postprocessing is not None:
            x = self._apply_module(self.postprocessing, x)

        return x
    
    def forward_nn(self, x: torch.Tensor) -> torch.Tensor:
        if not self._override_model:
            if self.norm_in is not None:
                x = self._apply_module(self.norm_in, x)
        x_enc = self._apply_module(self.nn, x)
        return x_enc

    def set_regularization(self, c0_reg=1e-6):
        """
        Set the regularization of the instantaneous correlation matrix.

        Parameters
        ----------
        c0_reg : float, optional
            Regularization added to the diagonal of the instantaneous
            correlation matrix to improve numerical stability during the
            Cholesky decomposition, by default 1e-6.
        """
        self.tica.reg_C_0 = c0_reg
    
    def evaluate_loss(
        self,
        batch,
        batch_idx: int,
        update_state: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the SelfTICA contrastive loss.

        Returns the unregularized loss and TICA eigenvalues as metrics.
        """
        if isinstance(self.nn, FeedForward):
            x_t = batch["data"]
            x_lag = batch["data_lag"]
            w_t = batch["weights"]
            w_lag = batch["weights_lag"]
        elif isinstance(self.nn, BaseGNN):
            x_t = self._setup_graph_data(batch, key="data_list")
            x_lag = self._setup_graph_data(batch, key="data_list_lag")
            w_t = x_t["weight"]
            w_lag = x_lag["weight"]

        z_t = self.forward_nn(x_t)
        z_t_pred = self.predictor(z_t)
        z_lag = self.forward_nn(x_lag)
        loss = self.loss_fn(z_t_pred, z_lag)

        with torch.no_grad():
            loss_noreg = self.loss_fn.noreg(z_t_pred, z_lag)
            eigvals, _ = self.tica.compute(
                data=[z_t, z_lag],
                weights=[w_t, w_lag],
                save_params=update_state,
            )
            if update_state:
                self.current_evecs.copy_(self.tica.evecs)
                self.current_means.copy_(self.tica.mean)

        output = {"loss": loss, "loss_noreg": loss_noreg}
        output.update(
            {f"eigval_{i + 1}": eigval for i, eigval in enumerate(eigvals)}
        )
        return output
