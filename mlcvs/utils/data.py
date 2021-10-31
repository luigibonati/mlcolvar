"""Dataloader."""

__all__ = ["LabeledDataset"]

from torch.utils.data import Dataset

class LabeledDataset(Dataset):
    """
    Dataset with labels.
    """

    def __init__(self, colvar, labels):
        """
        Create dataset from colvar and labels.

        Parameters
        ----------
        colvar : array-like
            input data 
        labels : array-like
            classes labels
        """

        self.colvar = colvar
        self.labels = labels

    def __len__(self):
        return len(self.colvar)

    def __getitem__(self, idx):
        x = (self.colvar[idx], self.labels[idx])
        return x


