# __all__ = ["compute_fes"]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings

# optional packages
# pandas
try:
    import pandas as pd

    PANDAS_IS_INSTALLED = True
except ImportError:
    PANDAS_IS_INSTALLED = False
# check whether KDEpy and scikit-learn are installed (used for FES)
try:
    from KDEpy import FFTKDE
    from KDEpy.utils import cartesian

    KDEPY_IS_INSTALLED = True
except ImportError:
    KDEPY_IS_INSTALLED = False
try:
    import sklearn

    SKLEARN_IS_INSTALLED = True
except ImportError:
    SKLEARN_IS_INSTALLED = False
# set default backend based on installation
if KDEPY_IS_INSTALLED:
    kdelib = "KDEpy"
elif SKLEARN_IS_INSTALLED:
    kdelib = "sklearn"
else:
    kdelib = None


def compute_fes(
    X,
    temp=None,
    fes_units="kJ/mol",
    kbt=None,
    num_samples=200,
    bounds=None,
    bandwidth=0.01,
    kernel="gaussian",
    weights=None,
    scale_by=None,
    blocks=1,
    fes_to_zero=None,
    plot=False,
    plot_color = "fessa6",
    plot_max_fes = None,
    plot_error_style = "fill_between",
    plot_levels = None,
    ax=None,
    backend=None,
    eps=None,
):
    """Compute the Free Energy Surface using a kernel density estimation (KDE) along the given variables. See notes below.

        Parameters
    ----------
    X : array-like or list of arrays
        Input data of shape (n_samples, n_features), or list of 1D arrays (one per dimension).
    temp : float, optional
        Temperature (in Kelvin). Required if `kbt` is not provided.
    fes_units : str, optional
        Units of the FES if using `temp`, by default "kJ/mol".
    kbt : float, optional
        Thermal energy in the same units as the FES. Required if `temp` is not provided.
    num_samples : int, optional
        Number of grid points per dimension for FES evaluation, by default 200.
    bounds : list of tuples, optional
        List of (min, max) for each dimension. If None, computed from data range.
    bandwidth : float, optional
        Bandwidth for the kernel density estimation, by default 0.01.
    kernel : str, optional
        Kernel type for the KDE. Supported values depend on the backend, by default "gaussian".
    weights : array-like, optional
        Weights associated with the data points, shape (n_samples,).
    scale_by : str or list, optional
        Standardize each variable before KDE. Use "std" to scale by standard deviation, "range" for range normalization, or provide a list of scaling factors.
    blocks : int, optional
        Number of data blocks to use for uncertainty estimation. Default is 1 (no error estimate).
    fes_to_zero : int or tuple, optional
        Index (or multi-index) in the grid where FES is shifted to zero. If None, minimum value is subtracted.
    plot : bool, optional
        Whether to plot the FES (only for 1D or 2D).
    plot_color : str, optional
        Color for 1D FES plot, default "fessa6".
    plot_max_fes : float, optional
        Maximum FES value to plot. Higher values will be masked.
    plot_error_style : str, optional
        Style for plotting 1D uncertainty: "fill_between", "errorbar", or None.
    plot_levels : list or int, optional
        Contour levels for 2D FES plots. Passed to `matplotlib.contourf`.
    ax : matplotlib.axes.Axes, optional
        Axis object to plot into. If None, a new figure is created.
    backend : str, optional
        Backend to use for KDE: "KDEpy" or "sklearn". If None, use the best available.
    eps : float, optional
        Regularization added to the KDE estimate before taking the log to avoid log(0). If None, auto-tuned.

    Returns
    -------

    fes : ndarray
        Free energy surface evaluated on the grid. Shape is (num_samples,) in 1D and (num_samples, num_samples) in 2D.
    grid : ndarray or list of ndarrays
        Grid coordinates corresponding to the FES values. A single array in 1D, list of arrays in 2D.
    bounds : list of tuples
        Bounds used for each variable, after optional rescaling.
    error : ndarray or None
        Standard error estimate from block averaging (only if `blocks` > 1). Same shape as `fes`.

    Notes
    -----
    - The KDE can be computed using either KDEpy (recommended for speed in 2D) or scikit-learn. Use `backend="KDEpy"` or `backend="sklearn"` to specify.
    - Either `temp` or `kbt` must be provided (not both). If using `temp`, ensure `fes_units` is set correctly.
    - If `scale_by` is used, input variables are rescaled before KDE. This means that if using ``scale_by='range'`` a bandwidth of 0.01 corresponds to a 1/100 of the range of values assumed by the variable.
    - If `blocks > 1`, the function performs block averaging to estimate uncertainty [1].
    - The FES is computed on a regular grid; the grid ordering is consistent with `numpy.meshgrid(..., indexing='xy')`, so no manual transpose is needed for plotting.

    References
    ----------
    [1] Invernizzi, Piaggi, and Parrinello, Phys. Rev. X 10, 041034 (2020)




    """
    # check backend for KDE
    if kdelib is None:
        raise NotImplementedError(
            "The function compute_fes requires either KDEpy (fast) or scikit-learn (slow) to work."
        )
    if kdelib == "sklearn":
        warnings.warn(
            "Warning: KDEpy is not installed, falling back to scikit-learn for kernel density estimation (very slow for d>1)."
        )

    if backend == "sklearn":
        from sklearn.neighbors import KernelDensity
    elif backend is None:
        backend = kdelib

    # check temperature / units
    if kbt is not None:
        if temp is not None:
            raise ValueError("Only one of kbt and temp can be specified.")
     
        fes_units = None
    else: 
        if temp is None:
            raise ValueError("One of kbt and temp must be specified.")
    
        if fes_units == "kJ/mol":
                kb = 0.00831441
        elif fes_units == "kcal/mol":
            kb = 0.0019872041
        elif fes_units == "eV":
            kb = 8.6173324e-5
        else:
            raise ValueError(
                "fes_units must be one of 'kJ/mol', 'kcal/mol', 'eV'."
            )
        kbt = kb * temp
        
    # dataset
    if PANDAS_IS_INSTALLED:
        if type(X) == pd.DataFrame:
            X = X.values
    if type(X) == list:
        X = np.vstack(X).T
    elif type(X) == torch.Tensor:
        X = X.numpy()
    if X.ndim == 1:
        X = X.reshape([-1, 1])

    # div
    nsamples = X.shape[0]
    dim = X.shape[1]

    # weights
    if weights is None:
        weights = np.ones(nsamples)
    else:
        assert weights.ndim == 1
        assert weights.shape[0] == nsamples                                                

    # rescale
    if scale_by is not None:
        if scale_by == "std":
            scale = X.std(axis=0)
        elif scale_by == "range":
            scale = X.max(axis=0) - X.min(axis=0)
        elif type(scale_by) == list:
            scale = np.asarray(scale_by)
        X = np.copy(X)
        X /= scale

    # eval points
    offset = 1e-3
    if dim == 1:
        if bounds is None:
            bounds = (X[:, 0].min() - offset, X[:, 0].max() + offset)
        grid = np.linspace(bounds[0], bounds[1], num_samples)
        positions = grid.reshape([-1, 1])
    else:
        if bounds is None:
            bounds = [
                (X[:, i].min() - offset, X[:, i].max() + offset) for i in range(dim)
            ]
        grid_list = [np.linspace(b[0], b[1], num_samples) for b in bounds]
        grid = list(np.meshgrid(*grid_list))
        positions = np.vstack([g.ravel() for g in grid]).T

    # divide in blocks
    X_b = np.array_split(X, blocks)
    w_b = np.array_split(weights, blocks)

    fes_blocks, W_blocks = [], []  # values per block
    # block average
    for i in range(blocks):
        # data of each block
        X_i = X_b[i]
        w_i = w_b[i]

        # fit
        if backend == "KDEpy":
            kde = FFTKDE(bw=bandwidth, kernel=kernel)
            kde.fit(X_i, weights=w_i)

            if dim == 1:
                pos = [grid]
            else:
                pos = grid_list

            # pdf --> fes
            if eps is not None:
                fes_i = (
                    -kbt
                    * np.log(kde.evaluate(cartesian(pos)) + eps)
                    .reshape([num_samples for i in range(dim)],order="F")
                )
            else:
                # automatically adjust eps to avoid nans
                eps_values = [0] +np.logspace(-15,-6,10).tolist()
                for e in eps_values:    
                    fes_i = (
                        -kbt
                        * np.log(kde.evaluate(cartesian(pos)) + e)
                        .reshape([num_samples for i in range(dim)],order="F")

                    )
                    if not np.isnan(fes_i).any():
                        if e>0:
                            print(f"Adjusting regularization (eps) to {e:1.1e} to avoid NaNs.")
                        break

        elif backend == "sklearn":
            kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
            kde.fit(X_i, sample_weight=w_i)

            # logpdf --> fes
            fes_i = -kbt * kde.score_samples(positions).reshape(
                [num_samples for i in range(dim)]
            )

        # result for each block
        fes_blocks.append(fes_i)
        W_blocks.append(np.sum(w_i))

    fes_blocks = np.asarray(fes_blocks)
    W_blocks = np.asarray(W_blocks)

    # compute avg and std
    if blocks > 1:
        # weighted average
        fes = (np.nansum(fes_blocks.T * W_blocks, axis=-1) / np.nansum(W_blocks)).T
        # weighted std
        dev = fes_blocks - fes
        blocks_eff = (np.sum(W_blocks)) ** 2 / (np.sum(W_blocks**2))
        variance = (
            blocks_eff / (blocks_eff - 1)
            * (np.nansum((dev**2).T * W_blocks, axis=-1))
            / np.nansum(W_blocks)
        ).T
        error = np.sqrt(variance / blocks_eff)
    else:
        fes = fes_blocks[0]
        error = None
    
    if fes_to_zero is not None:
        fes -= fes[fes_to_zero]
    else:
        fes -= np.nanmin(fes)

    # rescale back
    if scale_by is not None:
        if dim == 1:
            bounds = (bounds[0] * scale[0], bounds[1] * scale[0])
            grid *= scale
        else:
            bounds = [
                (bounds[i][0] * scale[i], bounds[i][1] * scale[i]) for i in range(dim)
            ]
            for i in range(dim):
                grid[i] *= scale[i]

    # Plot
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        if dim == 1:
            fes2 = np.copy(fes)
            if plot_max_fes is not None:
                fes2[fes2 > plot_max_fes] = np.nan
            if blocks > 1:
                if plot_error_style is not None:
                    if plot_error_style == "errorbar":  
                        ax.errorbar(grid, fes2, error,color=plot_color,alpha=0.5)
                    elif plot_error_style == "fill_between":
                        ax.plot(grid, fes2, color=plot_color)
                        ax.fill_between(grid, fes2 - error, fes2 + error,alpha=0.5, color=plot_color)
                    else:
                        raise ValueError(
                            "plot_error_style must be 'errorbar' or 'fill_between' (None to disable plotting errors)"
                        )
            else:
                ax.plot(grid, fes2, color=plot_color)
            ax.set_ylabel(f"FES [{fes_units}]" if fes_units is not None else "FES")
        elif dim == 2:
            fes2 = np.copy(fes)
            if plot_max_fes is not None:
                fes2[fes2 > plot_max_fes] = np.nan
            extent = [item for sublist in bounds for item in sublist]
            pp = ax.contourf(fes2, cmap="fessa", levels=plot_levels, extent=extent)  # ,vmax=max_fes)
            cbar = plt.colorbar(pp, ax=ax)
            cbar.set_label(f"FES [{fes_units}]" if fes_units is not None else "FES")

    return fes, grid, bounds, error


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
            fes_units="kJ/mol",
            weights=np.ones_like(X),
            bandwidth=0.02,
            num_samples=50,
            bounds=None,
            fes_to_zero=None,
            scale_by="std",
            blocks=2,
            backend="sklearn",
        )
