#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in mlcvs.cvs.unsupervised.vae.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os
import tempfile

import pytest
import pytorch_lightning as pl
import torch

from mlcvs.cvs.unsupervised.vae import VariationalAutoEncoderCV
from mlcvs.data import DictionaryDataset, DictionaryDataModule


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('weights', [False, True])
def test_vae_cv_training(weights):
    """Run a full training of a VAECv."""
    # Create VAE CV.
    n_cvs = 2
    in_features = 8
    model = VariationalAutoEncoderCV(
        n_cvs=n_cvs,
        encoder_layers=[in_features, 6, 4],
        options={
            'norm_in': None,
            'encoder': {'activation' : 'relu'},
        }
    )

    # Create input data.
    batch_size = 100
    x = torch.randn(batch_size, in_features)
    data = {'data': x}

    # Create weights.
    if weights is True:
        data['weights'] = torch.rand(batch_size)

    # Train.
    datamodule = DictionaryDataModule(DictionaryDataset(data))
    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False)
    trainer.fit(model, datamodule)

    # Eval.
    model.eval()
    x_hat = model(x)
    assert x_hat.shape == (batch_size, n_cvs)

    # Test export to torchscript.
    # This try-finally clause is a workaround for windows not allowing opening temp files twice.
    try:
        tmp_file = tempfile.NamedTemporaryFile('wb', suffix='.ptc', delete=False)
        tmp_file.close()
        model.to_torchscript(file_path=tmp_file.name, method='trace')
        model_loaded = torch.jit.load(tmp_file.name)
    finally:
        os.unlink(tmp_file.name)
    x_hat2 = model_loaded(x)
    assert torch.allclose(x_hat, x_hat2)
