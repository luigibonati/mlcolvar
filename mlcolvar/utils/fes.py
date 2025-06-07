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
    X : array-like, list of arrays
        data
    temp : float, optional
        temperature, by default None
    fes_units : string, optional
        units of the FES, by default "kJ/mol"
    kbt : float, optional
        temperature in energy units (fes_units), by default None
    num_samples : int, optional
        number of points used along each direction, by default 100
    bounds : list of lists, optional
        limit the calculation of the FES to a region, by default equal to the range of values
    bandwidth : float, optional
        Bandwidth use for the kernels, by default 0.1.
    kernel: string, optional
        Kernel type, by default 'gaussian'.
    weights : array, optional
        array of samples weights, by default None
    scale_by : string, optional
        Standardize each variable by its standard deviation (`std`) or by its range of values, by default None
    blocks : int, optional
        number of blocks to be used, by default 1
    fes_to_zero: int, optional
        index of the array where to shift the FES to zero (if none the minimum is set to zero)
    plot : bool, optional
        Plot the results (only available for 1D and 2D FES), by default False
    plot_max_fes : float, optional
        Maximum value of the FES to be plotted, by default None
    plot_error_style: string, optional
        Style of the error bars in the plot (either "errorbar" or "fill_between"), by default "fill_between"
    ax : matplotlib axis, optional
        Axis where to plot, default create new figure
    backend: string, optional
        Specify the backend for KDE ("KDEpy" or "sklearn")
    eps: float, optional
        Add an epsilon in the argument of the log

    Returns
    -------
    fes: np.array
        free energy
    grid: np.array or list of arrays (>1D)
        grid points for plotting
    bounds: list
        bounds along each variable
    std: np.array
        (weighted) standard deviation along the blocks (only if blocks>1)

    Notes
    -----
    - This function has two backends to calculate the kernel density estimates, one using KDEpy (default) and the other using sklearn.neighbors.KernelDensity function to construct the estimate of the FES (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html). You can specify all the kernels supported.

    - One between temp and kbt must be specified. In the former case, the free energy units needs to be specified.

    - Note that if ``scale_by`` is not none, the bandwidth is not rescaled as well. So if using ``scale_by='range'`` a bandwidth of 0.01 corresponds to a 1/100 of the range of values assumed by the variable.

    - If blocks>1 a (weighted) block analyis is performed [1], which allows also to obtain an estimate of the uncertainty.

    [1] Invernizzi, Piaggi and Parrinello, PRX, 2020,10,4

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
                    .reshape([num_samples for i in range(dim)])
                    .T
                )
            else:
                # automatically adjust eps to avoid nans
                eps_values = [0] +np.logspace(-15,-6,10).tolist()
                for e in eps_values:    
                    fes_i = (
                        -kbt
                        * np.log(kde.evaluate(cartesian(pos)) + e)
                        .reshape([num_samples for i in range(dim)])
                        .T
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

        if fes_to_zero is not None:
            fes_i -= fes_i[fes_to_zero]
        else:
            fes_i -= np.nanmin(fes_i)
        # result for each block
        fes_blocks.append(fes_i)
        W_blocks.append(np.sum(w_i))

    fes_blocks = np.asarray(fes_blocks)
    W_blocks = np.asarray(W_blocks)

    # compute avg and std
    if blocks > 1:
        # weighted average
        fes = np.nansum(fes_blocks.T * W_blocks, axis=-1) / np.nansum(W_blocks)
        # weighted std
        dev = fes_blocks - fes
        blocks_eff = (np.sum(W_blocks)) ** 2 / (np.sum(W_blocks**2))
        variance = (
            blocks_eff / (blocks_eff - 1)
            * (np.nansum((dev**2).T * W_blocks, axis=-1))
            / np.nansum(W_blocks)
        )
        error = np.sqrt(variance / blocks_eff)
    else:
        fes = fes_blocks[0]
        error = None

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
