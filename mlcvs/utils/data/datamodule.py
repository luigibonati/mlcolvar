import warnings
import math
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, random_split, Subset
from torch._utils import _accumulate
from . import FastTensorDataLoader

__all__ = ["TensorDataModule"]

class TensorDataModule(pl.LightningDataModule):
    """Lightning DataModule constructed for TensorDataset(s)."""
    def __init__(self, dataset: TensorDataset, lengths=[0.8,0.2,0], batch_size: int or list = 32, random_splits: bool = True, shuffle : bool or list =  False,  generator : torch.Generator = None):
        """Create a DataModule derived from TensorDataset, which returns train/valid/test dataloaders.

        For the batch_size and shuffle parameters either a single value or a list-type of values (with same size as lenghts) can be provided.

        Parameters
        ----------
        dataset : TensorDataset
            Train dataset
        lengths : list, optional
            Lenghts of the training/validation/test datasets , by default [0.8,0.2]
        batch_size : int or list, optional
            Batch size, by default 32
        random_splits: bool, optional
            whether to randomly split train/valid/test or sequentially, by default True
        shuffle : Union[bool,list], optional
            whether to shuffle the batches from the dataloader, by default False
        generator : torch.Generator, optional
            set random generator for reproducibility, by default None
        """
        super().__init__()
        self.dataset = dataset
        self.lengths = lengths
        if isinstance(batch_size,int):
            self.batch_size = [batch_size for _ in lengths ]
        else:
            self.batch_size = batch_size
        if isinstance(shuffle,bool):
            self.shuffle = [shuffle for _ in lengths ]
        else:
            self.shuffle = shuffle
        self.random_splits = random_splits
        self.generator = None

    def setup(self, stage: str):
        if self.random_splits:
            self.dataset_splits = random_split(self.dataset, self.lengths, generator=self.generator)
        else:
            self.dataset_splits = sequential_split(self.dataset, self.lengths)

    def train_dataloader(self):
        """Return training dataloader."""
        return FastTensorDataLoader(self.dataset_splits[0], batch_size=self.batch_size[0],shuffle=self.shuffle[0])

    def val_dataloader(self):
        """Return validation dataloader."""
        return FastTensorDataLoader(self.dataset_splits[1], batch_size=self.batch_size[1],shuffle=self.shuffle[1])

    def test_dataloader(self):
        """Return test dataloader."""
        if len(self.lengths) >= 3:
            return FastTensorDataLoader(self.dataset_splits[2], batch_size=self.batch_size[2],shuffle=self.shuffle[2])
        else: 
            raise ValueError('Test dataset not available, you need to pass three lenghts to datamodule.')  

    def predict_dataloader(self):
        raise NotImplementedError

    def teardown(self, stage: str):
        pass 


def sequential_split(dataset, lengths: list ) -> list:
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
                warnings.warn(f"Length of split at index {i} is 0. "
                                f"This might result in an empty dataset.")

        # Cannot verify that dataset is Sized
        if sum(lengths) != len(dataset):    # type: ignore[arg-type]
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        # LB change: do sequential rather then random splitting
        return [Subset(dataset, np.arange(offset-length,offset)) for offset, length in zip(_accumulate(lengths), lengths)]

def test_TensorDataModule():
    torch.manual_seed(42)
    X = torch.randn((100,2))
    y = X.square()
    dataset = TensorDataset(X,y)

    datamodule = TensorDataModule(dataset,lengths=[0.75,0.2,0.05],batch_size=25)
    datamodule.setup('fit')
    loader = datamodule.train_dataloader()
    for data in loader:
        x_i, y_i = data
        print(x_i.shape, y_i.shape)
    datamodule.val_dataloader()
    datamodule.test_dataloader()

if __name__ == "__main__":
    test_TensorDataModule() 