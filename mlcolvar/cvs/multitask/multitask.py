#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Classes and utilities for multi-task collective variable.
"""

__all__ = ["MultiTaskCV"]

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Optional, Sequence

import torch

from mlcolvar.cvs.cv import BaseCV


# =============================================================================
# MULTITASK CV
# =============================================================================


class MultiTaskCV:
    """Multi-task collective variable.

    This class wraps an existing CV object and adds a linear combination of other
    auxiliary loss functions that target different datasets with different information.
    The class works only if the ``main_cv`` does not make use of ``__slots__``.

    Examples
    --------

    A semi-supervised autoencoder mixing ELBO and the Fisher's discriminant loss.

    >>> from mlcolvar.cvs import AutoEncoderCV
    >>> from mlcolvar.core.loss import FisherDiscriminantLoss
    >>> from mlcolvar.data import DictDataset, DictModule

    >>> n_descriptors = 5
    >>> n_labels = 2  # Number of states
    >>> n_cvs = 2

    Initialize the multi-task CV. The Fisher's discriminant loss has half the
    weight of the ELBO loss.

    >>> main_cv = AutoEncoderCV(encoder_layers=[n_descriptors, 10, n_cvs])
    >>> aux_loss_fn = FisherDiscriminantLoss(n_states=n_labels)
    >>> multi_cv = MultiTaskCV(main_cv, auxiliary_loss_fns=[aux_loss_fn], loss_coefficients=[0.5])

    MultiTaskCV now exposes the same API as AutoEncoderCV.

    >>> multi_cv.norm_in.set_custom(mean=torch.tensor(0.0), range=torch.tensor(1.0))

    Create a multi-dataset datamodule for this CV.

    >>> n_samples = 100
    >>> unsupervised_dataset = DictDataset({
    ...     'data': torch.rand(n_samples, n_descriptors),
    ... })
    >>> supervised_dataset = DictDataset({
    ...     'data': torch.rand(n_samples, n_descriptors),
    ...     'labels': torch.tensor([0., 1]).repeat(n_samples//2)
    ... })
    >>> datamodule = DictModule(dataset=[unsupervised_dataset, supervised_dataset])

    # Create a PyTorch Lightning trainer.
    >>> import lightning
    >>> trainer = lightning.Trainer(max_epochs=1, log_every_n_steps=5, logger=None, enable_checkpointing=False)

    """

    def __init__(
        self,
        main_cv: BaseCV,
        auxiliary_loss_fns: Sequence,
        loss_coefficients: Optional[Sequence[float]] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        main_cv : BaseCV
            The main collective variable. The CV will dynamically inherit from
            this object's class and expose all its members.
        auxiliary_loss_fns : list
            A list of auxiliary loss functions.
        loss_coefficients : list-like of floats, optional
            A list of length ``len(auxiliary_loss_fns)`` with the coefficients
            of the linear combination of loss functions. If not provided, all
            auxiliary loss functions are assigned coefficient 1 (the main CV
            has always coefficient 1).

        """
        # This changes dynamically the class of this object to inherit both from
        # MultiTaskCV and main_cv.__class__ so that we can access all members of
        # main_cv and still be able to override some of them.
        self.__class__ = type(
            "MultiTask" + main_cv.__class__.__name__,  # class name
            (self.__class__, main_cv.__class__),  # base classes
            {},  # self.__class__.__dict__
        )

        # Copy all members of main_cv into this object.
        self.__dict__ = main_cv.__dict__

        self.auxiliary_loss_fns = torch.nn.ModuleList(auxiliary_loss_fns)
        self.loss_coefficients = loss_coefficients

    def training_step(self, train_batch, batch_idx):
        stage = "train" if self.training else "valid"

        # Compute main loss (the main CV should already log the first loss).
        loss = super().training_step(train_batch["dataset0"], batch_idx)

        # Compute auxiliary losses one by one.
        for loss_idx, aux_loss_fn in enumerate(self.auxiliary_loss_fns):
            dataset_batch = train_batch["dataset" + str(loss_idx + 1)]

            # Prepare keyword arguments to pass to the auxiliary loss function.
            aux_loss_kwargs = {
                k: v for k, v in dataset_batch.items() if not k.startswith("data")
            }

            # Forward data of this dataset (and eventually the time-lagged one).
            cv = self.forward_cv(dataset_batch["data"])
            try:
                cv_lag = self.forward_cv(dataset_batch["data_lag"])
            except KeyError:  # Not a time-lagged CV.
                aux_loss = aux_loss_fn(cv, **aux_loss_kwargs)
            else:
                aux_loss = aux_loss_fn(cv, cv_lag, **aux_loss_kwargs)

            # Log the auxiliary loss (before the coefficient).
            self.log(f"{stage}_aux_loss_{loss_idx}", aux_loss.to(float), on_epoch=True)

            if self.loss_coefficients is not None:
                aux_loss = self.loss_coefficients[loss_idx] * aux_loss
            loss = loss + aux_loss

        # Log the total loss
        self.log(f"{stage}_total_loss", loss.to(float), on_epoch=True)

        # return loss
        return loss


if __name__ == "__main__":
    import doctest

    doctest.testmod()
