#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and functions in mlcolvar.cvs.multitask.multitask.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os
import tempfile

import pytest
import lightning
import torch

from mlcolvar.core.nn import FeedForward
from mlcolvar.core.loss import TDALoss, FisherDiscriminantLoss, AutocorrelationLoss
from mlcolvar.cvs.cv import BaseCV
from mlcolvar.cvs.multitask.multitask import MultiTaskCV
from mlcolvar.cvs.timelagged import DeepTICA
from mlcolvar.cvs.unsupervised import AutoEncoderCV, VariationalAutoEncoderCV
from mlcolvar.data import DictDataset, DictModule


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

N_DESCRIPTORS = 4
N_CVS = 2
N_STATES = 3


# =============================================================================
# TEST UTILITY FUNCTIONS/CLASSES
# =============================================================================


class MockAuxLoss(torch.nn.Module):
    """Mock auxiliary loss for mock testing."""

    def __init__(self, in_features=N_DESCRIPTORS, out_features=N_CVS):
        super().__init__()
        self.task_specific_nn = torch.nn.Linear(in_features, out_features)

        # Store initial parameters to check that training really happens.
        self.initial_nn_weight = self.task_specific_nn.weight.detach().clone()
        self.kwargs = None

    def forward(self, data, data_lag=None, **kwargs):
        self.kwargs = kwargs
        return self.task_specific_nn(data).sum()


class MockCV(BaseCV, lightning.LightningModule):
    """Mock CV for mock testing."""

    DEFAULT_BLOCKS = []
    MODEL_BLOCKS = []

    def __init__(self, in_features=N_DESCRIPTORS, out_features=N_CVS):
        """Constructor."""
        model = FeedForward(layers=[in_features, in_features])
        super().__init__(model=model)
        self.loss_fn = MockAuxLoss(in_features, out_features)

    def training_step(self, train_batch, batch_idx):
        """Training step."""
        return self.loss_fn(**train_batch)


def create_dataset(
    dataset_type,
    weights=False,
    n_samples=40,
    n_descriptors=N_DESCRIPTORS,
    n_labels=N_STATES,
):
    """Create one of three types of datasets with random data for testing.

    dataset_type can be 'unsupervised', 'supervised', or 'time-lagged', which will
    determine the fields of the dataset:

    * 'unsupervised': 'data', ('weights')
    * 'supervised': 'data', 'labels', ('weights')
    * 'time-lagged': 'data', 'data_lag', ('weights', 'weights_lag')

    Weights are added only if ``weights`` is ``True``.

    """
    assert dataset_type in set(["supervised", "unsupervised", "time-lagged"])

    data = {"data": torch.randn((n_samples, n_descriptors))}

    # Add weights.
    if weights:
        data["weights"] = torch.rand(n_samples)

    # Data-type-specific fields.
    if dataset_type == "supervised":
        # With sequential sampling, this make sure that all labels are represented
        # in the validation and training set so that LDA/TDA don't complain.
        labels = torch.arange(n_labels, dtype=torch.get_default_dtype())
        data["labels"] = labels.repeat(n_samples // n_labels + 1)[:n_samples]
    elif dataset_type == "time-lagged":
        data["data_lag"] = torch.randn((n_samples, n_descriptors))
        if weights:
            data["weights_lag"] = torch.rand(n_samples)

    return DictDataset(data)


def create_cv(cv_name, n_descriptors=N_DESCRIPTORS, n_cvs=N_CVS):
    """Return a new CV and its associated dataset type (see create_dataset).

    cv_name can be one of 'ae', 'vae', or 'deeptica'.
    """
    if cv_name == "ae":
        returned = "unsupervised", AutoEncoderCV(
            encoder_layers=[n_descriptors, 10, n_cvs]
        )
    elif cv_name == "vae":
        returned = "unsupervised", VariationalAutoEncoderCV(
            n_cvs=n_cvs, encoder_layers=[n_descriptors, 10]
        )
    elif cv_name == "deeptica":
        returned = "time-lagged", DeepTICA(model=[n_descriptors, 10, n_cvs])
    else:
        raise ValueError("Unrecognized cv_name.")

    # With multiple dataset, Normalization's parameters must be initialized manually.
    # This work because by default mean and range are assigned 0 and 1 value.
    returned[1].norm_in.is_initialized = True

    return returned


def create_loss(loss_name, n_states=N_STATES, n_cvs=N_CVS):
    """Return a loss and its associated dataset type (see create_dataset).

    loss_name can be one of 'lda', 'tda', and 'autocorrelation' or a list of these
    elements.

    """
    if loss_name == "autocorrelation":
        return "time-lagged", AutocorrelationLoss()
    elif loss_name == "lda":
        return "supervised", FisherDiscriminantLoss(n_states=n_states)
    elif loss_name == "tda":
        cv = TDALoss(
            n_states=N_STATES,
            target_centers=torch.linspace(-1, 1, N_STATES)
            .unsqueeze(-1)
            .repeat(1, N_CVS),
            target_sigmas=torch.tensor([0.2] * N_STATES).unsqueeze(-1).repeat(1, N_CVS),
        )
        return "supervised", cv
    else:
        raise ValueError("Unrecognized loss_name")


def create_multitask_cv_and_datasets(
    main_cv_name, weights, auxiliary_loss_names, loss_coefficients
):
    """Return a new MultiTaskCV object and a list of compatible datasets.

    main_cv_name can take all values supported in create_cv().
    weights are passed to create_dataset()
    auxiliary_loss_names can take all values supported in create_loss().
    loss_coefficients are passed directly to MultiTaskCV.__init__().

    Return the MultiTaskCV object and a list of datasets.

    """
    # Instantiate the base CV model and its dataset.
    dataset_type, main_cv = create_cv(main_cv_name)
    datasets = [create_dataset(dataset_type, weights=weights)]

    # Instantiate the auxiliary losses and its datasets.
    aux_losses = []
    for aux_loss_name in auxiliary_loss_names:
        dataset_type, loss = create_loss(aux_loss_name)
        aux_losses.append(loss)
        datasets.append(create_dataset(dataset_type))

    # Create the MultiTask CV.
    multi_cv = MultiTaskCV(main_cv, aux_losses, loss_coefficients)

    return multi_cv, datasets


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    "dataset_types,weights,loss_coefficients",
    [
        (["supervised", "unsupervised"], [False, False], None),
        (["unsupervised", "supervised", "time-lagged"], [True, False, True], None),
        (["time-lagged", "unsupervised", "supervised"], [True, True, True], [2.0, 5.0]),
    ],
)
def test_multitask_loss(dataset_types, weights, loss_coefficients):
    """Auxiliary loss functions are called correctly.

    Test:
    * Auxuliary functions are all called during training.
    * Task specific layers within auxiliary losses are recognized by PyTorch and trained.
    * All the dataset keywords are passed to the losses.
    """
    # Create mock MultitaskCV.
    main_cv = MockCV()
    aux_loss_fns = [MockAuxLoss() for _ in range(len(dataset_types) - 1)]
    datasets = [create_dataset(dt, weights=w) for dt, w in zip(dataset_types, weights)]
    multi_cv = MultiTaskCV(main_cv, aux_loss_fns, loss_coefficients)

    # Do a few steps of training.
    datamodule = DictModule(datasets, shuffle=False, random_split=False)
    trainer = lightning.Trainer(
        max_epochs=2, log_every_n_steps=5, logger=None, enable_checkpointing=False
    )
    trainer.fit(multi_cv, datamodule)

    # Check that all mock loss functions have been called and that
    # the task-specific layers have been regularly trained.
    all_losses = [multi_cv.loss_fn] + [l for l in multi_cv.auxiliary_loss_fns]
    for loss in all_losses:
        assert not torch.allclose(loss.initial_nn_weight, loss.task_specific_nn.weight)

    # Check that all fields were passed.
    for loss, dataset, dataset_type, weighted in zip(
        all_losses, datasets, dataset_types, weights
    ):
        if dataset_type == "supervised":
            expected_kwargs = {"labels"}
        else:
            expected_kwargs = set()
        if weighted:
            expected_kwargs.add("weights")
            if dataset_type == "time-lagged":
                expected_kwargs.add("weights_lag")
        assert set(loss.kwargs.keys()) == expected_kwargs


@pytest.mark.parametrize(
    "main_cv_name,weights",
    [
        ("ae", False),
        ("ae", True),
        ("vae", False),
        ("vae", True),
        ("deeptica", True),  # DeepTICA currently doesn't support unweighted data.
    ],
)
@pytest.mark.parametrize(
    "auxiliary_loss_names,loss_coefficients",
    [
        (["tda"], None),
        (["lda"], None),
        (
            ["autocorrelation"],
            None,
        ),  # This ends up testing DeepTICA + autocorrelation. Doesn't make sense but we can do it.
        (["lda", "autocorrelation"], None),
        (["tda", "autocorrelation"], None),
        (["tda"], [0.5]),  # This adds a coefficient in front of the auxiliary loss
        (["lda", "autocorrelation"], [2.0, 0.2]),
    ],
)
def test_multitask_training(
    main_cv_name, weights, auxiliary_loss_names, loss_coefficients
):
    """Run a full training of a MultiTaskCV.

    Test:
    * The CV is compatible with the PyTorch Lightning's Trainer.
    * Export works correctly.
    """
    # Create the MultiTaskCV and the list of datasets.
    multi_cv, datasets = create_multitask_cv_and_datasets(
        main_cv_name, weights, auxiliary_loss_names, loss_coefficients
    )

    # Train.
    datamodule = DictModule(datasets, shuffle=False, random_split=False)
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False
    )
    trainer.fit(multi_cv, datamodule)

    # Eval.
    multi_cv.eval()
    x = datasets[0]["data"]
    x_hat = multi_cv(x)
    assert x_hat.shape == (x.shape[0], N_CVS)

    # Do round-trip through torchscript.
    # This try-finally clause is a workaround for windows not allowing opening temp files twice.
    try:
        tmp_file = tempfile.NamedTemporaryFile("wb", suffix=".ptc", delete=False)
        tmp_file.close()
        multi_cv.to_torchscript(file_path=tmp_file.name, method="trace")
        multi_cv_loaded = torch.jit.load(tmp_file.name)
    finally:
        os.unlink(tmp_file.name)
    x_hat2 = multi_cv_loaded(x)
    assert torch.allclose(x_hat, x_hat2)
