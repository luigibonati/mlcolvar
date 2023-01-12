import torch

# util for computing mean and range over dataset

def compute_mean_range(x: torch.Tensor, print_values=False):
    """
    Compute mean and range of values over dataset (for inputs / outputs standardization)
    """
    Max, _ = torch.max(x, dim=0)
    Min, _ = torch.min(x, dim=0)

    Mean = (Max + Min) / 2.0
    Range = (Max - Min) / 2.0

    if print_values:
        print("Standardization enabled.")
        print("Mean:", Mean.shape, "-->", Mean)
        print("Range:", Range.shape, "-->", Range)
    if (Range < 1e-6).nonzero().sum() > 0:
        print(
            "[Warning] Normalization: the following features have a range of values < 1e-6:",
            (Range < 1e-6).nonzero(),
        )
        Range[Range < 1e-6] = 1.0

    return Mean, Range

def normalize(
    x: torch.Tensor, Mean: torch.Tensor, Range: torch.Tensor
) -> (torch.Tensor):
    """
    Compute standardized inputs/outputs (internal).

    Parameters
    ----------
    x: torch.Tensor
        input/output
    Mean: torch.Tensor
        mean values to be subtracted.
    Range: torch.Tensor
        interval range to be divided by.

    Returns
    -------
    out : torch.Tensor
        standardized inputs/outputs
    """

    # if shape ==

    if x.ndim == 2:
        batch_size = x.size(0)
        x_size = x.size(1)

        Mean_ = Mean.unsqueeze(0).expand(batch_size, x_size)
        Range_ = Range.unsqueeze(0).expand(batch_size, x_size)
    elif x.ndim == 1:
        Mean_ = Mean
        Range_ = Range
    else:
        raise ValueError(
            "Input tensor must of shape (n_features) or (n_batch,n_features)."
        )

    return x.sub(Mean_).div(Range_)

def unnormalize(
    x: torch.Tensor, Mean: torch.Tensor, Range: torch.Tensor
) -> (torch.Tensor):
    """
    Compute standardized inputs/outputs (internal).

    Parameters
    ----------
    x: torch.Tensor
        input/output
    Mean: torch.Tensor
        mean values to be added back.
    Range: torch.Tensor
        interval range to be multiplied back.

    Returns
    -------
    out : torch.Tensor
        standardized inputs/outputs
    """

    # if shape ==

    if x.ndim == 2:
        batch_size = x.size(0)
        x_size = x.size(1)

        Mean_ = Mean.unsqueeze(0).expand(batch_size, x_size)
        Range_ = Range.unsqueeze(0).expand(batch_size, x_size)
    elif x.ndim == 1:
        Mean_ = Mean
        Range_ = Range
    else:
        raise ValueError(
            "Input tensor must of shape (n_features) or (n_batch,n_features)."
        )

    return x.mul(Range_).add(Mean_)