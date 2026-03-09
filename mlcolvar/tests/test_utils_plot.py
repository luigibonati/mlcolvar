import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from mlcolvar.data.dataset import DictDataset
from mlcolvar.utils import plot as plot_utils
from mlcolvar.utils.plot import test_utils_plot


def _make_dataset(with_labels: bool = True) -> DictDataset:
    # Small deterministic dataset used across distribution plotting scenarios.
    data = torch.tensor([[0.1, 1.0, 2.0],
                         [0.2, 1.1, 2.1],
                         [0.3, 1.2, 2.2],
                         [0.4, 1.3, 2.3]])
    kwargs = {"data": data}
    if with_labels:
        kwargs["labels"] = torch.tensor([0, 0, 1, 1])
    return DictDataset(kwargs, feature_names=["a", "b", "c"])


def test_muller_brown():
    # Case 1: potential helpers return vectors with the same shape as input coordinates.
    x = np.linspace(-1.5, 1.5, 10)
    y = np.linspace(-0.5, 2.5, 10)
    mp = plot_utils.muller_brown_potential(x, y)
    mp3 = plot_utils.muller_brown_potential_three_states(x, y)
    assert mp.shape == x.shape
    assert mp3.shape == x.shape

    # Case 2: MFEP loaders return 2D arrays with two columns (x, y).
    mfep = plot_utils.muller_brown_mfep()
    mfep3 = plot_utils.muller_brown_three_states_mfep()
    assert mfep.ndim == 2 and mfep.shape[1] == 2
    assert mfep3.ndim == 2 and mfep3.shape[1] == 2


def test_isolines():
    # Case 1: callable input + contourf path with auto-created axis.
    ax = plot_utils.plot_isolines_2D(
        lambda xx, yy: xx**2 + yy**2,
        mode="contourf",
        num_points=15,
        max_value=0.7,
    )
    assert ax is not None
    plt.close(ax.figure)

    # Tiny module used to test torch.nn.Module branches.
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data = torch.tensor([[1.0, -1.0]])
    model.train()

    # Case 2: module input + provided axis returns None and keeps training mode unchanged.
    fig, ax = plt.subplots()
    out = plot_utils.plot_isolines_2D(
        model,
        mode="contour",
        num_points=8,
        ax=ax,
        allow_grad=False,
    )
    assert out is None
    assert model.training is True
    plt.close(fig)

    # Case 3: allow_grad=True branch with auto-created axis.
    ax2 = plot_utils.plot_isolines_2D(
        model,
        mode="contour",
        num_points=6,
        allow_grad=True,
    )
    assert ax2 is not None
    plt.close(ax2.figure)


def test_metrics():
    # Shared metrics dictionary for all plot_metrics scenarios.
    metrics = {"train_loss_epoch": [3.0, 2.0, 1.5], "valid_loss": [3.1, 2.2, 1.6]}

    # Case 1: auto-created axis with custom labels, styles, and colors.
    ax = plot_utils.plot_metrics(
        metrics,
        labels=["train", "valid"],
        linestyles=["-", "--"],
        colors=["fessa0", "fessa1"],
        yscale="linear",
    )
    assert ax is not None
    assert ax.get_xlabel() == "Epoch"
    assert ax.get_ylabel() == "Loss"
    assert ax.get_title() == "Learning curves"
    plt.close(ax.figure)

    # Case 2: provided axis path, explicit x values, and no title.
    fig, ax = plt.subplots()
    out = plot_utils.plot_metrics(metrics, ax=ax, x=np.array([0, 2, 4]), title=None)
    assert out is None
    assert ax.get_title() == ""
    plt.close(fig)


def test_feature_distribution():
    # Case 1: labeled dataset with auto-created axes and custom titles.
    ds_labeled = _make_dataset(with_labels=True)
    plot_utils.plot_features_distribution(ds_labeled, ["a", "b"], titles=["A", "B"])
    plt.close("all")

    # Case 2: unlabeled dataset with user-provided axes.
    ds_unlabeled = _make_dataset(with_labels=False)
    fig, axs = plt.subplots(1, 2)
    plot_utils.plot_features_distribution(ds_unlabeled, ["a", "b"], axs=axs)
    plt.close(fig)

    # Case 3: invalid `features` type should raise TypeError.
    with pytest.raises(TypeError):
        plot_utils.plot_features_distribution(ds_labeled, {"a": 0})

    # Case 4: feature/axis mismatch should raise ValueError.
    ds = _make_dataset(with_labels=True)
    fig, axs = plt.subplots(1, 1, squeeze=False)
    axs = axs.ravel()
    with pytest.raises(ValueError):
        plot_utils.plot_features_distribution(ds, ["a", "b"], axs=axs)
    plt.close(fig)


if __name__ == "__main__":
    test_utils_plot()