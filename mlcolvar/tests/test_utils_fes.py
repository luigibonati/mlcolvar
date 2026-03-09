import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlcolvar.utils import plot as _plot_utils  # register fessa colormap/colors
from mlcolvar.utils.fes import SKLEARN_IS_INSTALLED, compute_deltaG, compute_fes


def test_fes():
    # Case 1: 1D FES with plotting enabled and block errors shown as fill_between.
    x = np.linspace(0.0, 1.0, 120)
    fig1, ax1 = plt.subplots()
    fes, grid, bounds, err = compute_fes(
        X=x,
        kbt=1.0,
        bandwidth=0.08,
        num_samples=60,
        blocks=2,
        plot=True,
        plot_error_style="fill_between",
        plot_color="C0",
        ax=ax1,
        backend="KDEpy",
    )
    assert fes.shape == (60,)
    assert grid.shape == (60,)
    assert len(bounds) == 2
    assert err is not None and err.shape == (60,)
    assert ax1.get_ylabel() == "FES"
    plt.close(fig1)

    # Case 2: same 1D setup but using the errorbar plotting branch.
    fig2, ax2 = plt.subplots()
    fes2, grid2, _, err2 = compute_fes(
        X=x,
        kbt=1.0,
        bandwidth=0.08,
        num_samples=40,
        blocks=2,
        plot=True,
        plot_error_style="errorbar",
        plot_color="C1",
        ax=ax2,
        backend="KDEpy",
    )
    assert fes2.shape == (40,)
    assert grid2.shape == (40,)
    assert err2 is not None and err2.shape == (40,)
    plt.close(fig2)

    # Case 3: invalid error style should raise in the 1D plotting path when blocks > 1.
    with pytest.raises(ValueError):
        compute_fes(
            X=x,
            kbt=1.0,
            bandwidth=0.08,
            num_samples=30,
            blocks=2,
            plot=True,
            plot_error_style="invalid-style",
            plot_color="C2",
            backend="KDEpy",
        )

    # Case 4: 2D FES plotting path with contourf + colorbar.
    rng = np.random.default_rng(0)
    x2 = rng.normal(loc=0.0, scale=0.5, size=80)
    y2 = rng.normal(loc=1.0, scale=0.4, size=80)
    fig3, ax3 = plt.subplots()
    fes3, grid3, bounds3, err3 = compute_fes(
        X=[x2, y2],
        kbt=1.0,
        bandwidth=0.2,
        num_samples=25,
        blocks=1,
        plot=True,
        plot_levels=10,
        ax=ax3,
        backend="KDEpy",
    )
    assert fes3.shape == (25, 25)
    assert isinstance(grid3, list) and len(grid3) == 2
    assert len(bounds3) == 2
    assert err3 is None
    plt.close(fig3)

    # Case 5: optional sklearn backend path (if available in the current environment).
    if SKLEARN_IS_INSTALLED:
        fes4, grid4, _, _ = compute_fes(
            X=[x2, y2],
            temp=300.0,
            fes_units="kJ/mol",
            bandwidth=0.2,
            num_samples=20,
            blocks=1,
            plot=False,
            backend="sklearn",
        )
        assert fes4.shape == (20, 20)
        assert isinstance(grid4, list) and len(grid4) == 2


def test_delta_g():
    rng = np.random.default_rng(42)

    # Case 1: 1D deltaG with plotting enabled and explicit time axis.
    x_a = rng.normal(loc=-5.0, scale=0.2, size=200)
    x_b = rng.normal(loc=5.0, scale=0.2, size=200)
    x = np.concatenate((x_a, x_b))
    rng.shuffle(x)
    time = np.arange(len(x))
    weights = rng.random(len(x))

    fig1, ax1 = plt.subplots()
    grid, delta_g = compute_deltaG(
        X=x,
        stateA_bounds=[-6, -4],
        stateB_bounds=[4, 6],
        kbt=1.0,
        intervals=10,
        weights=weights,
        reverse=True,
        time=time,
        plot=True,
        plot_color="C0",
        ax=ax1,
    )
    assert grid.shape == (10,)
    assert delta_g.shape == (10,)
    assert np.allclose(delta_g[-1], 0.0, atol=0.8)
    assert ax1.get_xlabel() == "Time"
    assert "$\\Delta$G" in ax1.get_ylabel()
    plt.close(fig1)

    # Case 2: 2D deltaG branch with plot disabled.
    x2_a = rng.normal(loc=-5.0, scale=0.2, size=(200, 2))
    x2_b = rng.normal(loc=5.0, scale=0.2, size=(200, 2))
    x2 = np.concatenate((x2_a, x2_b), axis=0)
    rng.shuffle(x2)
    w2 = rng.random(len(x2))

    grid2, delta_g2 = compute_deltaG(
        X=x2,
        stateA_bounds=[[-6, -4], [-6, -4]],
        stateB_bounds=[[4, 6], [4, 6]],
        kbt=1.0,
        intervals=10,
        weights=w2,
        reverse=False,
        time=None,
        plot=False,
    )
    assert grid2.shape == (10,)
    assert delta_g2.shape == (10,)
    assert np.allclose(delta_g2[-1], 0.0, atol=0.8)
