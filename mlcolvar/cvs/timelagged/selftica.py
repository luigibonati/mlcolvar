import torch 
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, BaseGNN, Normalization
from mlcolvar.core.stats import TICA
from mlcolvar.core.loss import ContrastiveLoss
from typing import Union, List

__all__ = ["SelfTICA"]
    

class SelfTICA(BaseCV):
    """Self-supervised time-lagged independent component analysis (Self-TICA).
    
    Is is a self-supervised generalization of Deep-TICA in which uses a encoder
    to learn a latent representation of the input data. TICA is then applied to
    this latent space to extract the slowest modes of the CV.

    **Data**: for training it requires a DictDataset with the keys 'data' (input at time t)
    and 'data_lag' (input at time t+lag), as well as the corresponding 'weights' and
    'weights_lag' which will be used to weight the time correlation functions.
    This can be created with the helper function `create_timelagged_dataset`.

    **Loss** :L2 contrastive loss encourging temporal consistency and decorrelation (ContrastiveLoss)

    References
    ----------
    .. [1] Turri, G., Bonati, L., Zhu, K., Pontil, M., & Novelli, P, "Self-Supervised Evolution 
        Operator Learning for High-Dimensional Dynamical Systems," arXiv preprint arXiv:2505.18671. (2025).
    .. [2] L. Bonati, G. Piccini, and M. Parrinello, “ Deep learning the slow modes for
        rare events sampling,” PNAS USA 118, e2113533118 (2021)

    See also
    --------
    mlcolvar.core.stats.TICA
        Time Lagged Indipendent Component Analysis
    mlcolvar.core.loss.ContrastiveLoss
        Encourging temporal consistency and decorrelation
    mlcolvar.utils.timelagged.create_timelagged_dataset
        Create dataset of time-lagged data.
    """

    DEFAULT_BLOCKS = ["norm_in", "nn", "predictor", "tica"]
    MODEL_BLOCKS = ["nn", "predictor", "tica"]

    def __init__(
        self, 
        model: Union[List[int], FeedForward, BaseGNN],
        n_cvs: int = None,
        regularization: float = 1e-5,
        options: dict = None, 
        **kwargs,
        ):
        """
        Define a Self-TICA CV, composed of a neural network encoder, a predictor and a TICA object.

        By default a module standardizing the inputs is also used. 

        Parameters
        ----------
        encoder_layers : list
            A list of integers specifying the number of neurons in each layer of the encoder network.
        n_cvs : int,
            Number of cvs to optimize, default None (= last layer)
        regularization : float, optional
            L2 regularization strength used in the loss function (default: 1e-5).
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}. 
            Available blocks: ['norm_in', 'encoder', 'predictor', 'tica'].
            Set 'block_name' = None or False to turn off that block.
        """
        super().__init__(model, **kwargs)        

        # =======   LOSS  =======
        self.loss_fn = ContrastiveLoss(reg=regularization, mode="l2")

        # here we need to override the self.out_features attribute
        self.out_features = n_cvs

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
            if self.out_features is not None:
                self.register_buffer('n_out', torch.as_tensor(self.out_features))   

        # initalize predictor
        o = "predictor"
        # ===== infer output dimension =====
        if hasattr(self.nn, "out_features") and isinstance(self.nn.out_features, int):
            out_dim = self.nn.out_features
        elif hasattr(self.nn, "n_out"):
            out_dim = int(self.nn.n_out)
        else:
            raise ValueError("Cannot infer output dimension from model")
        
        self.predictor = FeedForward(
           layers=[out_dim, out_dim],
           **options[o]
        )

        # initialize TICA
        o = "tica"
        self.tica = TICA(self.nn.out_features, n_cvs, **options[o])

        self.register_buffer('current_evecs', torch.eye(self.nn.out_features, n_cvs))
        self.register_buffer('current_means', torch.zeros(self.nn.out_features))
        self.register_buffer('optimal_lag_time', torch.tensor(-1.0))

    def compute_tica(
            self,
            datamodule: lightning.LightningDataModule,
            lag_time: float=None,
            update_optimal: bool = False,
    ):
        """
        Compute TICA features with specified lag time (without updating model parameters).

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader containing time-lagged data pairs
        lag_time : float, optional
            Lag time for analysis (None uses original training lag time)
        update_optimal : bool
            Whether to set these parameters as optimal for interfence
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
        self.current_evecs = self.tica.evecs.clone()
        self.current_means = self.tica.mean.clone()
        if lag_time is not None:
            self.optimal_lag_time = torch.tensor(lag_time)

    def forward(self, x: torch.Tensor, cell=None) -> torch.Tensor:

        if self.preprocessing is not None:
            x = self._apply_module(self.preprocessing, x, cell=cell)

        # Encode input into latent representation
        x = self.forward_nn(x)

        # In evaluation mode, apply TICA projection to obtain CVs
        if not self.training:
            centered = x - self.current_means
            x = centered @ self.current_evecs[:, :self.n_cvs]
        
        if self.postprocessing is not None:
            x = self._apply_module(self.postprocessing, x)

        return x
    
    def forward_nn(self, x: torch.Tensor, predict: bool = False) -> torch.Tensor:
        if not self._override_model:
            if self.norm_in is not None:
                x = self._apply_module(self.norm_in, x)
        # Optionally apply predictor: z → P(z)
        x_enc = self._apply_module(self.nn, x)
        if predict:
            x_enc = self.predictor(x_enc)
        return x_enc

    def set_regularization(self, c0_reg=1e-6):
        """ 
        Add identity matrix multiplied by `c0_reg` to correlation matrix c(0) to avoid instabilities in performin Cholesky and .

        Parameters
        ----------
        c0_reg : float 
            Regularization value for C_0
        """
        self.tica.reg_C_0 = c0_reg
    
    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics
        1) Perform a forward pass through the neural network to obtain the latent representations
        2) Compute the loss between the current and lagged representations
        3) Without gradient tracking
            - Compute the unregularized loss for monitoring
            - Apply the TICA estimator to obtain approximate eigenvalues of the transfer operator
        """
        # =================get data===================
        if isinstance(self.nn, FeedForward):
            x_t = train_batch["data"]
            x_lag = train_batch["data_lag"]
            w_t = train_batch["weights"]
            w_lag = train_batch["weights_lag"]
        elif isinstance(self.nn, BaseGNN):
            x_t = self._setup_graph_data(train_batch, key='data_list')
            x_lag = self._setup_graph_data(train_batch, key='data_list_lag')
            w_t = x_t['weight']
            w_lag = x_lag['weight']
            
        # =================forward====================
        f_t = self.forward_nn(x_t, predict=True)
        f_lag = self.forward_nn(x_lag)
        # ===================loss=====================
        loss = self.loss_fn(f_t, f_lag)
        # ===================tica=====================
        with torch.no_grad():
            loss_noreg = self.loss_fn.noreg(f_t, f_lag)
            f_t_nolin = self.forward_nn(x_t, predict=False)
            eigvals, _ = self.tica.compute(
                data=[f_t_nolin, f_lag], weights=[w_t, w_lag], save_params=True
            )
            self.current_evecs = self.tica.evecs.clone()
            self.current_means = self.tica.mean.clone()
        # ====================log=====================
        name = "train" if self.training else "valid"
        loss_dict = {f"{name}_loss": loss}
        loss_noreg_dict = {f"{name}_loss_noreg": loss_noreg}
        eig_dict = {f"{name}_eigval_{i+1}": eigvals[i] for i in range(len(eigvals))}
        self.log_dict({**loss_dict, **loss_noreg_dict, **eig_dict}, on_step=True, on_epoch=True)
        return loss
        

def test_self_tica():
    #tests
    import numpy as np 
    from mlcolvar.data import DictModule
    from mlcolvar.utils.timelagged import create_timelagged_dataset
    from mlcolvar.tests import data_dir

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

        eigvals, eigvecs = model.compute_tica(datamodule, lag_time=10)
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
    from mlcolvar.core.nn.graph.schnet import SchNetModel
    from mlcolvar.data.graph.utils import create_test_graph_input
    gnn_model = SchNetModel(2, 0.1, [1, 8])
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

    eigvals, eigvecs = model.compute_tica(datamodule, lag_time=10)
    print("TICA eigenvalues:", eigvals)

    example_input_graph_test = create_test_graph_input(output_type='example', n_atoms=4, n_samples=3, n_states=2)
    traced_model = model.to_torchscript(
        file_path=None,
        method="trace",
    )
    assert torch.allclose(model(example_input_graph_test), traced_model(example_input_graph_test), atol=1e-6)

if __name__ == "__main__":
    test_self_tica()
