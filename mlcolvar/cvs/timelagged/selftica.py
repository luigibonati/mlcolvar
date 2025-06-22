import torch 
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.stats import TICA
from mlcolvar.core.loss import RegSpectralLoss

__all__ = ["SelfTICA"]

class SelfTICA(BaseCV, lightning.LightningModule):
    """Self-supervised time-lagged independent component analysis (Deep-TICA).
    
    Is is a self-supervised generalization of Deep-TICA in which uses a encoder
    to learn a latent representation of the input data. TICA is then applied to
    this latent space to extract the slowest modes of the CV.

    **Data**: for training it requires a DictDataset with the keys 'data' (input at time t)
    and 'data_lag' (input at time t+lag), as well as the corresponding 'weights' and
    'weights_lag' which will be used to weight the time correlation functions.
    This can be created with the helper function `create_timelagged_dataset`.

    **Loss** :L2 contrastive loss encourging temporal consistency and decorrelation (RegSpectralLoss)

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
    mlcolvar.core.loss.RegSpectralLoss
        Encourging temporal consistency and decorrelation
    mlcolvar.utils.timelagged.create_timelagged_dataset
        Create dataset of time-lagged data.
    """

    BLOCKS = ['norm_in','encoder','linear','tica']

    def __init__(
        self, 
        encoder_layers: list,
        n_cvs: int = None,
        regularization: float = 1e-5,
        options: dict = None, 
        **kwargs,
        ):
        """
        Define a Self-TICA CV, composed of a neural network encoder, a linear layers and a TICA object.

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
            Available blocks: ['norm_in', 'nn', 'tica'].
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(
            in_features=encoder_layers[0],
            out_features=n_cvs, # TO DO
            **kwargs,
        )

        latent_dim = encoder_layers[-1]

        # =======   LOSS  =======
        self.loss_fn = RegSpectralLoss(reg=regularization)

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======

        # initalize norm_in
        o = "norm_in"
        if (options[o] is not False) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features, **options[o])
        
        # initalize encoder
        o = "encoder"
        self.encoder = FeedForward(encoder_layers, **options[o])

        # initalize linear
        o = "linear"
        self.linear = torch.nn.Linear(latent_dim, latent_dim, bias=False)

        # initialize TICA
        o = "tica"
        self.tica = TICA(latent_dim, n_cvs, **options[o])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_nn(x)
    
    def forward_nn(self, x: torch.Tensor, lagged: bool = False) -> torch.Tensor:
        x_enc = self.encoder(x)
        if lagged:
            x_enc = self.linear(x_enc)
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
        x_t = train_batch["data"]
        x_lag = train_batch["data_lag"]
        w_t = train_batch["weights"]
        w_lag = train_batch["weights_lag"]
        # =================forward====================
        f_t = self.forward_nn(x_t)
        f_lag = self.forward_nn(x_lag, lagged=True)
        # ===================loss=====================
        loss = self.loss_fn(f_t, f_lag)
        # ===================tica=====================
        with torch.no_grad():
            loss_noreg = self.loss_fn.noreg(f_t, f_lag)
            f_lag_nolin = self.forward_nn(x_lag, lagged=False)
            eigvals, _ = self.tica.compute(
                data=[f_t, f_lag_nolin], weights=[w_t, w_lag], save_params=True
            )
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

    # create dataset
    X = np.loadtxt("mlcolvar/tests/data/mb-mcmc.dat")
    X = torch.Tensor(X)
    dataset = create_timelagged_dataset(X, lag_time=1)
    datamodule = DictModule(dataset, batch_size=10000)

    # create cv
    encoder_layers = [2, 10, 5]
    model = SelfTICA(encoder_layers, n_cvs=1)

    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False
    )
    trainer.fit(model, datamodule)

    model.eval()
    with torch.no_grad():
        s = model(X).numpy()
    print(X.shape, "-->", s.shape)


if __name__ == "__main__":
    test_self_tica()
