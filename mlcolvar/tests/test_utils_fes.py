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
    grid, deltaG, error = compute_deltaG(X=X,
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
    assert error is None

    # test keywords and independent trajectories
    trajectories = [X, np.random.permutation(X),
                    np.random.permutation(np.concatenate((X_a[:160], X_b[:180])))]
    weights_list = [weights, np.random.rand(len(trajectories[1])),
                    np.random.rand(len(trajectories[2]))]
    times = [np.arange(len(x)) for x in trajectories]

    grid, deltaG, error = compute_deltaG(X=trajectories,
                                         stateA_bounds=[-6, -4],
                                         stateB_bounds=[4, 6],
                                         kbt=1,
                                         intervals=10,
                                         weights=weights_list,
                                         reverse=True,
                                         time=times,
                                         plot=False,
                                         plot_color="fessa6",
                                         ax=None,
                                         )
    assert np.allclose(deltaG[-1], 0, atol=0.5)
    assert error is not None
    assert error.shape == deltaG.shape
    assert np.all(np.isfinite(error))
    assert np.all(error >= 0)

    # test 2D
    # make two fake states with gaussian distribution, to have deltaG=0
    X_a = np.random.rand(200, 2) - 5
    X_b = np.random.rand(200, 2) + 5

    X = np.concatenate((X_a, X_b), axis=0)
    X = np.random.permutation(X)

    trajectories = [X, np.random.permutation(X),
                    np.random.permutation(np.concatenate((X_a[:160], X_b[:180])))]
    weights_list = [np.random.rand(len(x)) for x in trajectories]
    times = [np.arange(len(x)) for x in trajectories]

    grid, deltaG, error = compute_deltaG(X=trajectories,
                                         stateA_bounds=[[-6, -4], [-6, -4]],
                                         stateB_bounds=[[4, 6], [4, 6]],
                                         kbt=1,
                                         intervals=10,
                                         weights=weights_list,
                                         reverse=True,
                                         time=times,
                                         plot=False,
                                         plot_color="fessa6",
                                         ax=None,
                                         )
    assert np.allclose(deltaG[-1], 0, atol=0.5)
    assert error is not None
    assert error.shape == deltaG.shape


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

    trajectories = [x, rng.permutation(x), rng.permutation(x[:350])]
    times = [np.arange(len(item)) for item in trajectories]
    weights = [rng.random(len(item)) + 0.1 for item in trajectories]

    fig1, ax1 = plt.subplots()
    grid, delta_g, error = compute_deltaG(
        X=trajectories,
        stateA_bounds=[-6, -4],
        stateB_bounds=[4, 6],
        kbt=1.0,
        intervals=10,
        weights=weights,
        reverse=True,
        time=times,
        plot=True,
        plot_color="C0",
        ax=ax1,
    )
    assert grid.shape == (10,)
    assert delta_g.shape == (10,)
    assert error is not None
    assert error.shape == (10,)
    assert np.all(np.isfinite(error))
    assert np.all(error >= 0)
    assert np.allclose(delta_g[-1], 0.0, atol=0.8)
    assert ax1.get_xlabel() == "Time"
    assert r"$\Delta G$" in ax1.get_ylabel()
    plt.close(fig1)

    # Case 1b: bias-derived weights should match explicitly supplied weights.
    positive_weights = [w + 0.1 for w in weights]

    grid_w, delta_g_w, error_w = compute_deltaG(
        X=trajectories,
        stateA_bounds=[-6, -4],
        stateB_bounds=[4, 6],
        kbt=1.0,
        intervals=10,
        weights=positive_weights,
        plot=False,
    )

    grid_b, delta_g_b, error_b = compute_deltaG(
        X=trajectories,
        stateA_bounds=[-6, -4],
        stateB_bounds=[4, 6],
        kbt=1.0,
        intervals=10,
        bias=[np.log(w) for w in positive_weights],
        plot=False,
    )

    np.testing.assert_allclose(grid_b, grid_w)
    np.testing.assert_allclose(delta_g_b, delta_g_w)
    np.testing.assert_allclose(error_b, error_w)

    # Case 2: 2D deltaG branch with independent trajectory uncertainty.
    x2_a = rng.normal(loc=-5.0, scale=0.2, size=(200, 2))
    x2_b = rng.normal(loc=5.0, scale=0.2, size=(200, 2))
    x2 = np.concatenate((x2_a, x2_b), axis=0)
    rng.shuffle(x2)
    trajectories2 = [x2, rng.permutation(x2), rng.permutation(x2[:320])]
    w2 = [rng.random(len(item)) + 0.1 for item in trajectories2]

    grid2, delta_g2, error2 = compute_deltaG(
        X=trajectories2,
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
    assert error2 is not None
    assert error2.shape == (10,)
    assert np.all(np.isfinite(error2))
    assert np.all(error2 >= 0)
    assert np.allclose(delta_g2[-1], 0.0, atol=0.8)

    # Case 3: a single trajectory does not provide a between-run error.
    grid3, delta_g3, error3 = compute_deltaG(
        X=x,
        stateA_bounds=[-6, -4],
        stateB_bounds=[4, 6],
        kbt=1.0,
        intervals=10,
        weights=weights[0],
        plot=False,
    )

    assert grid3.shape == (10,)
    assert delta_g3.shape == (10,)
    assert error3 is None


def test_funnel_delta_g():
    rng = np.random.default_rng(123)

    x_bound = rng.normal(loc=0.6, scale=0.05, size=200)
    x_unbound = rng.normal(loc=1.6, scale=0.05, size=200)
    x = np.concatenate((x_bound, x_unbound))
    rng.shuffle(x)

    trajectories = [x, rng.permutation(x), rng.permutation(x[:360])]
    times = [np.arange(len(item)) for item in trajectories]
    weights = [rng.random(len(item)) + 0.1 for item in trajectories]

    rfunnel = 0.2
    c0 = 1.0
    bat = 0.8
    uat = 1.4
    bounds = (0.4, 1.8)

    bound_region = [bounds[0], bat]
    unbound_region = [uat, bounds[1]]

    # Case 1: funnel deltaG with independent trajectory uncertainty and plotting.
    fig1, ax1 = plt.subplots()
    grid, delta_g, error = compute_deltaG(
        X=trajectories,
        stateA_bounds=bound_region,
        stateB_bounds=unbound_region,
        rfunnel=rfunnel,
        c0=c0,
        kbt=1.0,
        intervals=5,
        weights=weights,
        reverse=True,
        time=times,
        plot=True,
        plot_color="C0",
        ax=ax1,
    )

    assert grid.shape == (5,)
    assert delta_g.shape == (5,)
    assert error is not None
    assert error.shape == (5,)
    assert np.all(np.isfinite(delta_g))
    assert np.all(np.isfinite(error))
    assert np.all(error >= 0)
    assert ax1.get_xlabel() == "Time"
    assert r"$\Delta G_{funnel}$" in ax1.get_ylabel()
    plt.close(fig1)

    # Case 1b: bias-derived weights should match explicit weights.
    positive_weights = [w + 0.1 for w in weights]

    grid_w, delta_g_w, error_w = compute_deltaG(
        X=trajectories,
        stateA_bounds=bound_region,
        stateB_bounds=unbound_region,
        rfunnel=rfunnel,
        c0=c0,
        kbt=1.0,
        intervals=5,
        weights=positive_weights,
        plot=False,
    )

    grid_b, delta_g_b, error_b = compute_deltaG(
        X=trajectories,
        stateA_bounds=bound_region,
        stateB_bounds=unbound_region,
        rfunnel=rfunnel,
        c0=c0,
        kbt=1.0,
        intervals=5,
        bias=[np.log(w) for w in positive_weights],
        plot=False,
    )

    np.testing.assert_allclose(grid_b, grid_w)
    np.testing.assert_allclose(delta_g_b, delta_g_w)
    np.testing.assert_allclose(error_b, error_w)

    # Case 1c: final value should match the mean of independent estimates.
    volume_correction = np.pi * rfunnel**2 * c0 / 1.66
    delta_g_trajectories = []

    for x_traj, w_traj in zip(trajectories, positive_weights):
        mask_bound = np.logical_and(
            x_traj > bound_region[0],
            x_traj < bound_region[1],
        )
        mask_unbound = np.logical_and(
            x_traj > unbound_region[0],
            x_traj < unbound_region[1],
        )

        p_bound = 1e-8 + np.sum(w_traj[mask_bound])
        p_unbound = 1e-8 + np.sum(w_traj[mask_unbound])

        delta_g_trajectories.append(
            -np.log((p_bound / p_unbound) * volume_correction)
        )

    np.testing.assert_allclose(
        delta_g_w[-1],
        np.mean(delta_g_trajectories),
    )
    np.testing.assert_allclose(
        error_w[-1],
        np.std(delta_g_trajectories, ddof=1) / np.sqrt(len(trajectories)),
    )

    # Case 2: a single trajectory does not provide a between-run error.
    grid2, delta_g2, error2 = compute_deltaG(
        X=x,
        stateA_bounds=bound_region,
        stateB_bounds=unbound_region,
        rfunnel=rfunnel,
        c0=c0,
        kbt=1.0,
        intervals=5,
        weights=weights[0],
        plot=False,
    )

    assert grid2.shape == (5,)
    assert delta_g2.shape == (5,)
    assert error2 is None


def test_fes_independent_trajectories():
    if not SKLEARN_IS_INSTALLED:
        pytest.skip("scikit-learn is needed to verify the reference KDE")

    from sklearn.neighbors import KernelDensity

    rng = np.random.default_rng(21)
    trajectories = [rng.normal(loc=-0.6, scale=0.6, size=80),
                    rng.normal(loc=0.0, scale=0.8, size=120),
                    rng.normal(loc=0.6, scale=0.5, size=150)]

    fes, grid, bounds, error = compute_fes(
        trajectories=trajectories,
        kbt=1.0,
        bandwidth=0.25,
        num_samples=40,
        bounds=(-3, 3),
        fes_to_zero=20,
        backend="sklearn",
    )

    # Compute a reference KDE for each independent simulation on one grid.
    fes_trajectories = []
    for traj in trajectories:
        kde = KernelDensity(bandwidth=0.25, kernel="gaussian")
        kde.fit(traj.reshape(-1, 1))
        fes_trajectories.append(-kde.score_samples(grid.reshape(-1, 1)))

    fes_trajectories = np.asarray(fes_trajectories)
    mean = np.mean(fes_trajectories, axis=0)
    mean -= mean[20]
    sem = np.std(fes_trajectories, axis=0, ddof=1) / np.sqrt(len(trajectories))

    assert fes.shape == (40,)
    assert error.shape == (40,)
    np.testing.assert_allclose(fes, mean)
    np.testing.assert_allclose(error, sem)

    # Bias and explicit weights should give the same multi-trajectory result.
    weights = [rng.random(len(traj)) + 0.1 for traj in trajectories]
    fes_w, _, _, error_w = compute_fes(
        trajectories=trajectories,
        weights=weights,
        kbt=1.0,
        bandwidth=0.25,
        num_samples=40,
        bounds=(-3, 3),
        backend="sklearn",
    )
    fes_b, _, _, error_b = compute_fes(
        trajectories=trajectories,
        bias=[np.log(w) for w in weights],
        kbt=1.0,
        bandwidth=0.25,
        num_samples=40,
        bounds=(-3, 3),
        backend="sklearn",
    )
    np.testing.assert_allclose(fes_b, fes_w)
    np.testing.assert_allclose(error_b, error_w)

    # A single trajectory should not report between-run uncertainty.
    _, _, _, error_single = compute_fes(
        trajectories=[trajectories[0]],
        kbt=1.0,
        bandwidth=0.25,
        backend="sklearn",
    )
    assert error_single is None


def test_fes_independent_trajectories_2d():
    if not SKLEARN_IS_INSTALLED:
        pytest.skip("scikit-learn is required for this test")

    rng = np.random.default_rng(22)
    trajectories = [rng.normal(size=(100, 2)),
                    rng.normal(loc=0.5, size=(160, 2))]
    fes, grid, bounds, error = compute_fes(
        trajectories=trajectories,
        kbt=1.0,
        bandwidth=0.3,
        num_samples=20,
        backend="sklearn",
    )

    assert fes.shape == (20, 20)
    assert isinstance(grid, list) and len(grid) == 2
    assert len(bounds) == 2
    assert error.shape == (20, 20)
    assert np.all(np.isfinite(error))


def test_fes_independent_trajectories_invalid_input():
    if not SKLEARN_IS_INSTALLED:
        pytest.skip("scikit-learn is required for this test")

    x = np.linspace(0, 1, 60)
    with pytest.raises(ValueError, match="either X or trajectories"):
        compute_fes(X=x, trajectories=[x, x], kbt=1.0, backend="sklearn")

    with pytest.raises(ValueError, match="blocks and trajectories"):
        compute_fes(trajectories=[x, x], blocks=2, kbt=1.0, backend="sklearn")

    with pytest.raises(ValueError, match="same CV dimension"):
        compute_fes(trajectories=[x, np.ones((40, 2))], kbt=1.0, backend="sklearn")

    with pytest.raises(ValueError, match="match their trajectory lengths"):
        compute_fes(trajectories=[x, x], weights=[np.ones(60), np.ones(59)],
                    kbt=1.0, backend="sklearn")


def test_delta_g_independent_trajectories_mean_sem():
    rng = np.random.default_rng(23)
    trajectories = []
    for n_a, n_b in [(60, 40), (50, 50), (30, 70)]:
        x = np.concatenate((np.full(n_a, -5.0), np.full(n_b, 5.0)))
        trajectories.append(rng.permutation(x))

    grid, mean, error = compute_deltaG(
        X=trajectories,
        stateA_bounds=[-6, -4],
        stateB_bounds=[4, 6],
        kbt=1.0,
        intervals=5,
    )
    per_traj = np.asarray([
        np.log((n_a + 1e-8) / (n_b + 1e-8))
        for n_a, n_b in [(60, 40), (50, 50), (30, 70)]
    ])

    assert grid.shape == (5,)
    assert mean.shape == (5,)
    assert error.shape == (5,)
    np.testing.assert_allclose(mean[-1], np.mean(per_traj))
    np.testing.assert_allclose(error[-1], np.std(per_traj, ddof=1) / np.sqrt(3))
    np.testing.assert_allclose(grid[-1], 1.0)


def test_delta_g_independent_trajectories_missing_basin():
    x1 = np.concatenate((np.full(20, -5.0), np.full(80, 5.0)))
    x2 = np.concatenate((np.full(20, 5.0), np.full(80, -5.0)))
    grid, mean, error = compute_deltaG(
        X=[x1, x2],
        stateA_bounds=[-6, -4],
        stateB_bounds=[4, 6],
        kbt=1.0,
        intervals=5,
    )

    assert np.isnan(mean[0])
    assert np.isnan(error[0])
    assert np.all(np.isfinite(mean[-1:]))
    assert np.all(np.isfinite(error[-1:]))
