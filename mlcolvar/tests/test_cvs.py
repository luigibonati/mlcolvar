#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Shared tests for the objects and functions in the mlcolvar.cvs package.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os
import tempfile

import lightning
import pytest
import torch

import mlcolvar.cvs
from mlcolvar.data import DictDataset, DictModule


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

N_STATES = 2
N_DESCRIPTORS = 15
LAYERS = [N_DESCRIPTORS, 5, 5, N_STATES-1]

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def dataset():
    """Dataset with all fields required by all CV types."""
    n_samples = 10

    # Weights should be optional so we don't add them.
    data = {
        "data": torch.randn((n_samples, N_DESCRIPTORS)),
        "data_lag": torch.randn((n_samples, N_DESCRIPTORS)),
        "target": torch.randn(n_samples),
        "weights": torch.rand(n_samples),
        "weights_lag": torch.rand(n_samples),
    }

    # With sequential sampling, this make sure that all labels are represented
    # in the validation and training set so that LDA/TDA don't complain.
    labels = torch.arange(N_STATES, dtype=torch.get_default_dtype())
    data["labels"] = labels.repeat(n_samples // N_STATES + 1)[:n_samples]

    return DictDataset(data)


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize("cv_model", [
    mlcolvar.cvs.DeepLDA(layers=LAYERS, n_states=N_STATES),
    mlcolvar.cvs.DeepTDA(n_states=N_STATES, n_cvs=1, target_centers=[-1., 1.], target_sigmas=[0.1, 0.1], layers=LAYERS),
    mlcolvar.cvs.RegressionCV(layers=LAYERS),
    mlcolvar.cvs.DeepTICA(layers=LAYERS, n_cvs=1),
    mlcolvar.cvs.AutoEncoderCV(encoder_layers=LAYERS),
    mlcolvar.cvs.VariationalAutoEncoderCV(n_cvs=1, encoder_layers=LAYERS[:-1]),
])
def test_resume_from_checkpoint(cv_model, dataset):
    """CVs correctly resume from a checkpoint."""
    datamodule = DictModule(dataset, lengths=[1.0,0.], batch_size=len(dataset))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run a few steps of training in a temporary directory.
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        # Simulate a couple of epochs of training.
        trainer = lightning.Trainer(
            max_epochs=2,
            enable_checkpointing=True,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=tmp_dir_path,
        )
        trainer.fit(cv_model, datamodule)

        # Now load from checkpoint.
        file_name = 'epoch={}-step={}.ckpt'.format(trainer.current_epoch-1, trainer.global_step)
        checkpoint_file_path = os.path.join(tmp_dir_path, 'checkpoints', file_name)
        cv_model2 = cv_model.__class__.load_from_checkpoint(checkpoint_file_path)

    # Check that state is the same.
    x = dataset['data'].to(device)
    cv_model.to(device).eval()
    cv_model2.to(device).eval()
    assert torch.allclose(cv_model(x), cv_model2(x))
