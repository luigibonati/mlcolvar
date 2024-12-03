import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, BaseGNN, Normalization
from mlcolvar.data import DictModule
from mlcolvar.core.stats import LDA
from mlcolvar.core.loss import ReduceEigenvaluesLoss
from typing import Union, List


__all__ = ["DeepLDA"]


class DeepLDA(BaseCV, lightning.LightningModule):
    """Deep Linear Discriminant Analysis (Deep-LDA) CV.
    Non-linear generalization of LDA in which a feature map is learned by a neural network optimized
    as to maximize the classes separation. The method is described in [1]_.

    **Data**: for training it requires a DictDataset with the keys 'data' and 'labels'.

    **Loss**: maximize LDA eigenvalues (ReduceEigenvaluesLoss)

    References
    ----------
    .. [1] L. Bonati, V. Rizzi, and M. Parrinello, "Data-driven collective variables for enhanced
        sampling", JPCL 11, 2998–3004 (2020).

    See also
    --------
    mlcolvar.core.stats.LDA
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
        layers : list
            Number of neurons per layer
        n_states : int
            Number of states for the training
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['norm_in','nn','lda'] .
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(model=model, **kwargs)
        self.save_hyperparameters(ignore=['model'])

        # =======   LOSS  =======
        # Maximize the sum of all the LDA eigenvalues.
        self.loss_fn = ReduceEigenvaluesLoss(mode="sum")

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # Save n_states
        self.n_states = n_states

        # ======= BLOCKS =======

        # initialize norm_in
        if not self._override_model:
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
                x = self.norm_in(x)
        x = self.nn(x)
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

    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        if isinstance(self.nn, FeedForward):
            # =================get data===================
            x = train_batch["data"]
            labels = train_batch["labels"]
            # =================forward====================
            h = self.forward_nn(x)

        elif isinstance(self.nn, BaseGNN):
            data = self._setup_graph_data(train_batch)
            labels = data['graph_labels'].squeeze()
            h = self.forward_nn(data)

        # ===================lda======================
        eigvals, _ = self.lda.compute(
            h, labels, save_params=True if self.training else False
        )
        # ===================loss=====================
        loss = self.loss_fn(eigvals)
        if self.lorentzian_reg > 0:
            s = self.lda(h)
            lorentzian_reg = self.regularization_lorentzian(s)
            loss += lorentzian_reg

        # ====================log=====================
        name = "train" if self.training else "valid"
        loss_dict = {f"{name}_loss": loss, f"{name}_lorentzian_reg": lorentzian_reg}
        eig_dict = {f"{name}_eigval_{i+1}": eigvals[i] for i in range(len(eigvals))}
        self.log_dict(dict(loss_dict, **eig_dict), on_step=True, on_epoch=True)
        return loss


def test_deeplda(n_states=2):
    from mlcolvar.data import DictDataset

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
        s = model(X).numpy()

    
    # feedforward external
    ff_model = FeedForward(layers=layers)
    print()
    print('EXTERNAL')
    print()
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


    # gnn external
    from mlcolvar.core.nn.graph.schnet import SchNetModel
    from mlcolvar.data.graph.utils import create_test_graph_input
    gnn_model = SchNetModel(2, 0.1, [1, 8])
    print()
    print('GNN')
    print()
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



if __name__ == "__main__":
    test_deeplda(n_states=2)
    test_deeplda(n_states=3)
