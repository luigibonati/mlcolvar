import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, random_split
from . import FastTensorDataLoader
from typing import Union
from .dataset import sequential_split

__all__ = ["TensorDataModule"]

class TensorDataModule(pl.LightningDataModule):
    """Lightning DataModule constructed for TensorDataset(s)."""
    def __init__(self, dataset: TensorDataset, lengths=[0.8,0.2,0], batch_size: Union[int,list] = 32, random_splits: bool = True, shuffle : Union[bool,list] =  False,  generator : torch.Generator = None):
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

def test_tensordatamodule():
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
    test_tensordatamodule() 