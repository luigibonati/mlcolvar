#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
PyTorch Lightning DataModule object for DictDatasets.
"""

__all__ = ["DictModule"]


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import math
from typing import Sequence, Union, Optional
import warnings

import torch
import numpy as np
import lightning
from torch.utils.data import random_split, Subset
from torch import default_generator, randperm

from mlcolvar.data import DictLoader, DictDataset


# =============================================================================
# DICTIONARY DATAMODULE CLASS
# =============================================================================


class DictModule(lightning.LightningDataModule):
    """Lightning DataModule constructed for :class:`~mlcolvar.data.dataset.DictDataset` .

    The DataModule automatically splits the :class:`~mlcolvar.data.dataset.DictDataset` s
    (using either random or sequential splitting) into training, validation, and (optionally)
    test sets.

    The class can also merge multiple :class:`~mlcolvar.data.dataset.DictDataset`s
    that have different keys (see example below). The datasets can have different sizes but must all have the
    same number of batches.

    Examples
    --------

    >>> x = torch.randn((50, 2))
    >>> dataset = DictDataset({'data': x, 'labels': x.square().sum(dim=1)})
    >>> datamodule = DictModule(dataset, lengths=[0.75,0.2,0.05], batch_size=25)

    >>> # This is usually called by PyTorch Lightning.
    >>> datamodule.setup()
    >>> train_loader = datamodule.train_dataloader()
    >>> for batch in train_loader:
    ...     batch_x = batch['data']
    ...     batch_y = batch['labels']
    ...     print(batch_x.shape, batch_y.shape)
    torch.Size([25, 2]) torch.Size([25])
    torch.Size([13, 2]) torch.Size([13])

    >>> val_loader = datamodule.val_dataloader()
    >>> test_loader = datamodule.test_dataloader()

    You can also iterate over multiple datasets. These can have different keys,
    but they must be of the same length.

    >>> dataset2 = DictDataset({'data2': torch.randn(50, 1), 'weights': torch.arange(50)})
    >>> datamodule = DictModule([dataset, dataset2], lengths=[0.8, 0.2], batch_size=5)
    >>> datamodule.setup()
    >>> train_loader = datamodule.train_dataloader()

    >>> # Print first batch.
    >>> for batch in train_loader:
    ...     print('batch dataset0:', list(batch['dataset0'].keys()))
    ...     print('batch dataset1:', list(batch['dataset1'].keys()))
    ...     break
    batch dataset0: ['data', 'labels']
    batch dataset1: ['data2', 'weights']

    """

    def __init__(
        self,
        dataset: DictDataset,
        lengths: Sequence = (0.8, 0.2),
        batch_size: Union[int, Sequence] = 0,
        random_split: bool = True,
        shuffle: Union[bool, Sequence] = True,
        generator: Optional[torch.Generator] = None,
    ):
        """Create a ``DataModule`` wrapping a :class:`~mlcolvar.data.dataset.DictDataset`.

        For the ``batch_size`` and ``shuffle`` parameters, either a single value
        or a list-type of values (with same size as ``lengths``) can be provided.

        Parameters
        ----------
        dataset : DictDataset or Sequence[DictDataset]
            The dataset or a list of datasets. If a list, the datasets can have
            different keys but they must all have the same number of samples.
        lengths : list-like, optional
            Lengths of the training, validation, and (optionally) test datasets.
            This must be a list of (float) fractions summing to 1. The default is
            ``[0.8,0.2]``.
        batch_size : int or list-like, optional
            Batch size, by default 0 (== ``len(dataset)``).
        random_split: bool, optional
            Whether to randomly split train/valid/test or sequentially, by default ``True``.
        shuffle : int or list-like, optional
            Whether to shuffle the batches in the ``DataLoader``, by default ``True``.
        generator : torch.Generator, optional
            Set random generator for reproducibility, by default ``None``.

        See Also
        --------
        :class:`~mlcolvar.data.dataloader.DictLoader`
            The PyTorch loader built by the data module.

        """
        super().__init__()
        self.dataset = dataset
        self.lengths = lengths
        # Keeping this private for now. Changing it at runtime would
        # require changing dataset_split and the dataloaders.
        self._random_split = random_split

        # save generator if given, otherwise set it to torch.default_generator
        self.generator = generator if generator is not None else default_generator
        if self.generator is not None and not self._random_split:
            warnings.warn(
                "A torch.generator was provided but it is not used with random_split=False"
            )

        # Make sure batch_size and shuffle are lists.
        if isinstance(batch_size, int):
            self.batch_size = [batch_size for _ in lengths]
        else:
            self.batch_size = batch_size
        if isinstance(shuffle, bool):
            self.shuffle = [shuffle for _ in lengths]
        else:
            self.shuffle = shuffle

        # This is initialized in setup().
        self._dataset_split = None

        # dataloaders
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def setup(self, stage: Optional[str] = None):
        if self._dataset_split is None:
            if isinstance(self.dataset, DictDataset):
                self._dataset_split = self._split(self.dataset)
            else:  # List of datasets.
                dataset_split = [self._split(d) for d in self.dataset]
                # Shape (n_datasets, n_loaders) -> (n_loaders, n_datasets)
                self._dataset_split = list(map(list, zip(*dataset_split)))

    def train_dataloader(self):
        """Return training dataloader."""
        self._check_setup()
        if self.train_loader is None:
            self.train_loader = DictLoader(
                self._dataset_split[0],
                batch_size=self.batch_size[0],
                shuffle=self.shuffle[0],
            )
        return self.train_loader

    def val_dataloader(self):
        """Return validation dataloader."""
        self._check_setup()
        if len(self.lengths) < 2:
            raise NotImplementedError(
                "Validation dataset not available, you need to pass two lengths to datamodule."
            )
        if self.valid_loader is None:
            self.valid_loader = DictLoader(
                self._dataset_split[1],
                batch_size=self.batch_size[1],
                shuffle=self.shuffle[1],
            )
        return self.valid_loader

    def test_dataloader(self):
        """Return test dataloader."""
        self._check_setup()
        if len(self.lengths) < 3:
            raise NotImplementedError(
                "Test dataset not available, you need to pass three lengths to datamodule."
            )
        if self.test_loader is None:
            self.test_loader = DictLoader(
                self._dataset_split[2],
                batch_size=self.batch_size[2],
                shuffle=self.shuffle[2],
            )
        return self.test_loader

    def predict_dataloader(self):
        raise NotImplementedError()

    def teardown(self, stage: str):
        pass

    def __repr__(self) -> str:
        string = f"DictModule(dataset -> {self.dataset.__repr__()}"
        string += f",\n\t\t     train_loader -> DictLoader(length={self.lengths[0]}, batch_size={self.batch_size[0]}, shuffle={self.shuffle[0]})"
        if len(self.lengths) >= 2:
            string += f",\n\t\t     valid_loader -> DictLoader(length={self.lengths[1]}, batch_size={self.batch_size[1]}, shuffle={self.shuffle[1]})"
        if len(self.lengths) >= 3:
            string += f",\n\t\t\ttest_loader =DictLoader(length={self.lengths[2]}, batch_size={self.batch_size[2]}, shuffle={self.shuffle[2]})"
        string += f")"
        return string

    def _split(self, dataset):
        """Perform the random or sequential spliting of a single dataset.

        Returns a list of Subset[DictDataset] objects.
        """

        dataset_split = split_dataset(
            dataset, self.lengths, self._random_split, self.generator
        )
        return dataset_split

    def _check_setup(self):
        """Raise an error if setup() has not been called."""
        if self._dataset_split is None:
            raise AttributeError(
                "The datamodule has not been set up yet. To get the dataloaders "
                "outside a Lightning trainer please call .setup() first."
            )


def split_dataset(
    dataset,
    lengths: Sequence,
    random_split: bool,
    generator: Optional[torch.Generator] = default_generator,
) -> list:
    """
    Sequentially or randomly split a dataset into non-overlapping new datasets of given lengths.

    If random_split=True the behavior is the same as torch.utils.data.dataset.random_split.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results.
    """

    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )
    if random_split:
        indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
        return [
            Subset(dataset, indices[offset - length : offset])
            for offset, length in zip(_accumulate(lengths), lengths)
        ]
    else:
        return [
            Subset(dataset, np.arange(offset - length, offset))
            for offset, length in zip(_accumulate(lengths), lengths)
        ]


def sequential_split(dataset, lengths: Sequence) -> list:
    """
    Sequentially split a dataset into non-overlapping new datasets of given lengths.

    The behavior is the same as torch.utils.data.dataset.random_split.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.
    """

    warnings.warn(
        "The function sequential_split is deprecated, use split_dataset(.., .., random_split=False, ..)",
        FutureWarning,
        stacklevel=2,
    )

    return split_dataset(dataset=dataset, lengths=lengths, random_split=False)

# Taken from python 3.5 docs, removed from PyTorch 2.3 onward
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

if __name__ == "__main__":
    import doctest

    doctest.testmod()
