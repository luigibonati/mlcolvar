#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Variational Autoencoder collective variable.
"""

__all__ = ["VariationalAutoEncoderCV"]

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Any, Optional, Tuple
import torch
import pytorch_lightning as pl
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.loss import ELBOGaussiansLoss


# =============================================================================
# VARIATIONAL AUTOENCODER CV
# =============================================================================

class VariationalAutoEncoderCV(BaseCV, pl.LightningModule):
    """Variational AutoEncoder Collective Variable.

    At training time, the encoder outputs a mean and a variance for each CV
    defining a Gaussian distribution associated to the input. One sample is
    drawn from this Gaussian, and it goes through the decoder. Then the ELBO
    loss is minimized. The ELBO sums the MSE of the reconstruction and the KL
    divergence between the generated Gaussian and a N(0, 1) Gaussian.

    At evaluation time, the encoder's output mean is used as the CV, while the
    variance output and the decoder are ignored.

    For training, it requires a DictionaryDataset with the key ``'data'`` and
    optionally ``'weights'``.
    """
    
    BLOCKS = ['norm_in', 'encoder', 'decoder']
    
    def __init__(
            self,
            n_cvs: int,
            encoder_layers: list,
            decoder_layers: Optional[list] = None,
            options: Optional[dict] = None,
            **kwargs
    ):
        """
        Variational autoencoder constructor.

        Parameters
        ----------
        n_cvs : int
            The dimension of the CV or, equivalently, the dimension of the latent
            space of the autoencoder.
        encoder_layers : list
            Number of neurons per layer of the encoder up to the last hidden layer.
            The size of the output layer is instead specified with ``n_cvs``
        decoder_layers : list, optional
            Number of neurons per layer of the decoder, except for the input layer
            which is specified by ``n_cvs``. If ``None`` (default), it takes automatically
            the reversed architecture of the encoder.
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default ``None``.
            Available blocks are: ``'norm_in'``, ``'encoder'``, and ``'decoder'``.
            Set ``'block_name' = None`` or ``False`` to turn off a block. Encoder
            and decoder cannot be turned off.
        """
        super().__init__(in_features=encoder_layers[0], out_features=n_cvs, **kwargs)

        # =======   LOSS  =======
        #ELBO loss function when latent space and reconstruction distributions are Gaussians.
        self.loss_fn = ELBOGaussiansLoss()

        # ======= OPTIONS ======= 
        # parse and sanitize
        options = self.parse_options(options)

        # The FeedForward implementing the encoder by default needs to have also
        # the nonlinearity (and eventually dropout/batchnorm) also for the output
        # layer since we'll have two separate linear layers for the mean and variance.
        try:
            encoder_opts = options['encoder']
        except KeyError:
            encoder_opts['encoder'] = {'last_layer_activation': True}
        else:  # Make sure we don't overwrite an existing option.
            if 'last_layer_activation' not in encoder_opts:
                encoder_opts['last_layer_activation'] = True

        # if decoder is not given reverse the encoder
        if decoder_layers is None:
            decoder_layers = encoder_layers[::-1]

        # ======= BLOCKS =======

        # initialize norm_in
        o = 'norm_in'
        if (options[o] is not False) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features, **options[o])

        # initialize encoder
        # The encoder outputs two values for each CV representing mean and std.
        o = 'encoder'
        self.encoder = FeedForward(encoder_layers, **options[o])
        self.mean_nn = torch.nn.Linear(in_features=encoder_layers[-1], out_features=n_cvs)
        self.log_var_nn = torch.nn.Linear(in_features=encoder_layers[-1], out_features=n_cvs)

        # initialize encoder
        o = 'decoder'
        self.decoder = FeedForward([n_cvs] + decoder_layers, **options[o])

    def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the value of the CV from preprocessed input.

        Return the mean output (ignoring the variance output) of the encoder
        after (optionally) applying the normalization to the input.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(n_batches, n_descriptors)`` or ``(n_descriptors,)``. The
            input descriptors of the CV after preprocessing.

        Returns
        -------
        cv : torch.Tensor
            Shape ``(n_batches, n_cvs)``. The CVs, i.e., the mean output of the
            encoder (the variance output is discarded).
        """
        if self.norm_in is not None:
            x = self.norm_in(x)
        x = self.encoder(x)

        # Take only the means and ignore the log variances.
        return self.mean_nn(x)

    def encode_decode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a pass of encoding + decoding.

        The function applies the normalizing to the inputs and its reverse on
        the output.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(n_batches, n_descriptors)`` or ``(n_descriptors,)``. The
            input descriptors of the CV after preprocessing.

        Returns
        -------
        mean : torch.Tensor
            Shape ``(n_batches, n_cvs)`` of ``(n_cvs,)``. The mean of the
            Gaussian distribution associated to the input in latent space.
        log_variance : torch.Tensor
            Shape ``(n_batches, n_cvs)`` of ``(n_cvs,)``. The logarithm of the
            variance of the Gaussian distribution associated to the input in
            latent space.
        x_hat : torch.Tensor
            Shape ``(n_batches, n_descriptors)`` or ``(n_descriptors,)``. The
            reconstructed descriptors.
        """
        # Normalize inputs.
        if self.norm_in is not None:
            x = self.norm_in(x)

        # Encode input into a Gaussian distribution.
        x = self.encoder(x)
        mean, log_variance = self.mean_nn(x), self.log_var_nn(x)

        # Sample from the Gaussian distribution in latent space.
        std = torch.exp(log_variance / 2)
        z = torch.distributions.Normal(mean, std).rsample()

        # Decode sample.
        x_hat = self.decoder(z)
        if self.norm_in is not None:
            x_hat = self.norm_in.inverse(x_hat)

        return mean, log_variance, x_hat

    def training_step(self, train_batch, batch_idx):
        """Single training step performed by the PyTorch Lightning Trainer."""
        x = train_batch['data']
        loss_kwargs = {}
        if 'weights' in train_batch:
            loss_kwargs['weights'] = train_batch['weights']

        # Encode/decode.
        mean, log_variance, x_hat = self.encode_decode(x)

        # Reference output (compare with a 'target' key if any, otherwise with input 'data')
        if 'target' in train_batch:
            x_ref = train_batch['target']
        else:
            x_ref = x 

        # Loss function.
        loss = self.loss_fn(x_ref, x_hat, mean, log_variance, **loss_kwargs)

        # Log.
        name = 'train' if self.training else 'valid'       
        self.log(f'{name}_loss', loss, on_epoch=True)

        return loss
