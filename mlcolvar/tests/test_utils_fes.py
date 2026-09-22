import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlcolvar.utils import plot as _plot_utils  # register fessa colormap/colors
from mlcolvar.utils.fes import (
    SKLEARN_IS_INSTALLED,
    compute_deltaG,
    compute_fes,
)


def test_compute_fes():
    X = np.linspace(1, 11, 100)
    fes, bins, bounds, error_ = compute_fes(
        X=X,
        weights=np.ones_like(X),
        kbt=1,
        bandwidth=0.02,
        num_samples=1000,
        bounds=(0, 10),
        fes_to_zero=25,
        scale_by="range",
        blocks=2,
        backend="KDEpy",
    )

    Y = np.random.rand(2, 100)

    if SKLEARN_IS_INSTALLED:  # TODO: change to use pytest functionalities?
        fes, bins, bounds, error_ = compute_fes(
            X=[Y[0], Y[1]],
            temp=300,
            units="kJ/mol",
            weights=np.ones_like(X),
            bandwidth=0.02,
            num_samples=50,
            bounds=None,
            fes_to_zero=None,
            scale_by="std",
            blocks=2,
            backend="sklearn",
        )

def test_compute_deltaG():
    np.random.seed(42)
    
    # test 1D
    # make two fake states with gaussian distribution, to have deltaG=0
    X_a = np.random.rand(200) - 5
    X_b = np.random.rand(200) + 5
    
    X = np.concatenate((X_a, X_b))
    X = np.random.permutation(X)

    time = np.arange(len(X))
    weights = np.random.rand(len(X))

    # test vanilla
    grid, deltaG = compute_deltaG(X=X,
                                  stateA_bounds=[-6, -4],
                                  stateB_bounds=[4, 6], 
                                  kbt=1,
                                  intervals=10, 
                                  weights=None,
                                  reverse=False,
                                  time=None,
                                  plot=False,
                                  plot_color="fessa6",
                                  ax=None,
                                  )
    assert np.allclose(deltaG[-1], 0, atol=0.5)

    # test keywords
    grid, deltaG = compute_deltaG(X=X,
                                  stateA_bounds=[-6, -4],
                                  stateB_bounds=[4, 6], 
                                  kbt=1,
                                  intervals=10, 
                                  weights=weights,
                                  reverse=True,
                                  time=time,
                                  plot=False,
                                  plot_color="fessa6",
                                  ax=None,
                                  )
    assert np.allclose(deltaG[-1], 0, atol=0.5)

    # test 2D
    # make two fake states with gaussian distribution, to have deltaG=0
    X_a = np.random.rand(200, 2) - 5
    X_b = np.random.rand(200, 2) + 5
    
    X = np.concatenate((X_a, X_b), axis=0)
    X = np.random.permutation(X)

    time = np.arange(len(X))
    weights = np.random.rand(len(X))
    grid, deltaG = compute_deltaG(X=X,
                                  stateA_bounds=[[-6, -4], [-6, -4]],
                                  stateB_bounds=[[4, 6], [4, 6]], 
                                  kbt=1,
                                  intervals=10, 
                                  weights=weights,
                                  reverse=True,
                                  time=time,
                                  plot=False,
                                  plot_color="fessa6",
                                  ax=None,
                                  )
    assert np.allclose(deltaG[-1], 0, atol=0.5)
    
    
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
            units="kJ/mol",
            bandwidth=0.2,
            num_samples=20,
            blocks=1,
            plot=False,
            backend="sklearn",
        )
        assert fes4.shape == (20, 20)
        assert isinstance(grid4, list) and len(grid4) == 2

    # Case 6: bias-derived weights must match the number of samples.
    with pytest.raises(ValueError, match="bias"):
        compute_fes(X=x, kbt=1.0, bias=np.ones(len(x) - 1), backend="KDEpy")


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

    # Case 1b: bias-derived weights should match explicitly supplied weights.
    positive_weights = weights + 0.1
    grid_w, delta_g_w = compute_deltaG(
        X=x,
        stateA_bounds=[-6, -4],
        stateB_bounds=[4, 6],
        kbt=1.0,
        intervals=10,
        weights=positive_weights,
        plot=False,
    )
    grid_b, delta_g_b = compute_deltaG(
        X=x,
        stateA_bounds=[-6, -4],
        stateB_bounds=[4, 6],
        kbt=1.0,
        intervals=10,
        bias=np.log(positive_weights),
        plot=False,
    )
    np.testing.assert_allclose(grid_b, grid_w)
    np.testing.assert_allclose(delta_g_b, delta_g_w)

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
    
def test_funnel_delta_g():
    rng = np.random.default_rng(123)

    # Case 1: 1D funnel deltaG with plotting enabled and explicit time axis.
    x_bound = rng.normal(loc=0.6, scale=0.05, size=200)
    x_unbound = rng.normal(loc=1.6, scale=0.05, size=200)
    x = np.concatenate((x_bound, x_unbound))
    rng.shuffle(x)

    time = np.arange(len(x))
    weights = rng.random(len(x)) + 0.1

    rfunnel = 0.2
    c0 = 1.0
    bat = 0.8
    uat = 1.4
    bounds = (0.4, 1.8)

    bound_region = [bounds[0], bat]
    unbound_region = [uat, bounds[1]]

    fig1, ax1 = plt.subplots()
    grid, delta_g = compute_deltaG(
        X=x,
        stateA_bounds=bound_region,
        stateB_bounds=unbound_region,
        rfunnel=rfunnel,
        c0=c0,
        kbt=1.0,
        intervals=5,
        weights=weights,
        reverse=True,
        time=time,
        plot=True,
        plot_color="C0",
        ax=ax1,
    )

    assert grid.shape == (5,)
    assert delta_g.shape == (5,)
    assert np.all(np.isfinite(delta_g))
    assert ax1.get_xlabel() == "Time"
    assert "$\\Delta G_{funnel}$" in ax1.get_ylabel()
    plt.close(fig1)

    # Case 1b: bias-derived weights should match explicitly supplied weights.
    positive_weights = weights + 0.1

    grid_w, delta_g_w = compute_deltaG(
        X=x,
        stateA_bounds=bound_region,
        stateB_bounds=unbound_region,
        rfunnel=rfunnel,
        c0=c0,
        kbt=1.0,
        intervals=5,
        weights=positive_weights,
        plot=False,
    )

    grid_b, delta_g_b = compute_deltaG(
        X=x,
        stateA_bounds=bound_region,
        stateB_bounds=unbound_region,
        rfunnel=rfunnel,
        c0=c0,
        kbt=1.0,
        intervals=5,
        bias=np.log(positive_weights),
        plot=False,
    )

    np.testing.assert_allclose(grid_b, grid_w)
    np.testing.assert_allclose(delta_g_b, delta_g_w)

    # Case 1c: final value should match the direct population-based formula.
    mask_bound = np.logical_and(x > bound_region[0], x < bound_region[1])
    mask_unbound = np.logical_and(x > unbound_region[0], x < unbound_region[1])

    # Reproduce the same interval construction used internally by compute_deltaG.
    interval_len = len(x) / 5
    interval_bounds = np.arange(0, len(x), interval_len)
    interval_bounds = np.ceil(interval_bounds).astype("int")
    interval_bounds = np.concatenate((interval_bounds, np.array([len(x) - 1])))

    # compute_deltaG uses Python slicing [start:end], so the final index is excluded.
    end = interval_bounds[-1]

    p_bound = 1e-8 + np.sum(positive_weights[:end][mask_bound[:end]])
    p_unbound = 1e-8 + np.sum(positive_weights[:end][mask_unbound[:end]])

    volume_correction = np.pi * rfunnel**2 * c0 / 1.66
    expected_delta_g = -np.log((p_bound / p_unbound) * volume_correction)

    np.testing.assert_allclose(delta_g_w[-1], expected_delta_g)