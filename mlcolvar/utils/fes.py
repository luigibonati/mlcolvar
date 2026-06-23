import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import torch
import warnings
from typing import List, Union

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
    X: np.ndarray,
    temp: float = None,
    units: str = "kJ/mol",
    kbt: float = None,
    num_samples: int = 200,
    bounds: List[float] = None,
    bandwidth: float = 0.01,
    kernel: str = "gaussian",
    weights: np.ndarray = None,
    bias: np.ndarray = None,
    scale_by: Union[str, List[float]] = None,
    blocks: int = 1,
    fes_to_zero: bool = None,
    plot: bool =False,
    plot_color: str = "fessa6",
    plot_max_fes: float = None,
    plot_error_style: str = "fill_between",
    plot_levels: Union[int, List[float]] = None,
    ax: matplotlib.axes = None,
    backend: str = None,
    eps: float = None,
):
    """Compute the Free Energy Surface using a kernel density estimation (KDE) along the given variables. See notes below.

        Parameters
    ----------
    X : array-like or list of arrays
        Input data of shape (n_samples, n_features), or list of 1D arrays (one per dimension).
    temp : float, optional
        Temperature (in Kelvin). Required if `kbt` is not provided.
    units : str, optional
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
    bias: array-like, optional
        Bias values to be used to compute the weights as exp(bias / kbt), shape (n_samples,)
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
    - Either `temp` or `kbt` must be provided (not both). If using `temp`, ensure `units` is set correctly.
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
    kbt, temp, units = _check_kbt_units(kbt, temp, units)
        
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
    if weights is not None and bias is not None:
        raise ValueError("The weights and bias keywords cannot be defined together, use only one of them!")
    if weights is None:
        if bias is None:
            weights = np.ones(nsamples)
        else:
            bias = np.asarray(bias)
            n_bias = bias.shape[0] if bias.ndim > 0 else 1
            if bias.ndim != 1 or n_bias != nsamples:
                raise ValueError(f"Input data and bias must have the same number of entries! "
                                 f"Found {nsamples} and {n_bias}.")
            weights = np.exp(bias / kbt)
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
            ax.set_ylabel(f"FES [{units}]" if units is not None else "FES")
        elif dim == 2:
            fes2 = np.copy(fes)
            if plot_max_fes is not None:
                fes2[fes2 > plot_max_fes] = np.nan
            extent = [item for sublist in bounds for item in sublist]
            pp = ax.contourf(fes2, cmap="fessa", levels=plot_levels, extent=extent)  # ,vmax=max_fes)
            cbar = plt.colorbar(pp, ax=ax)
            cbar.set_label(f"FES [{units}]" if units is not None else "FES")

    return fes, grid, bounds, error

def compute_deltaG(X: np.ndarray,
                   stateA_bounds: Union[List[float], List[List[float]], np.ndarray] = None,
                   stateB_bounds: Union[List[float], List[List[float]], np.ndarray] = None,
                   temp=None,
                   units="kJ/mol",
                   kbt: float = None,
                   intervals: int = 10,
                   weights: np.ndarray = None,
                   bias: np.ndarray = None,
                   reverse: bool = False,
                   time: np.ndarray = None,
                   plot: bool = False,
                   plot_color: str = "fessa6",
                   ax: matplotlib.axes = None,
                   eps: float = 1e-8,
                   mode: str = "state",
                   rfunnel: float = None,
                   bat: float = None,
                   uat: float = None,
                   bandwidth: float = 0.01,
                   num_samples: int = 200,
                   bounds: List[float] = None,
                   kernel: str = "gaussian",
                   backend: str = None,
                  ):
    """Compute free-energy differences.

    This function supports two modes:

    1. mode="state"
       Compute the difference in free energy between two states A and B
       from reweighted populations.

    2. mode="funnel"
       Compute funnel-corrected binding free energy from a 1D FES/KDE estimate.

    Parameters
    ----------
    X : np.ndarray
        Input CV data.

    stateA_bounds : List[float] or List[List[float]], optional
        Bounds of state A. Required when mode="state".

    stateB_bounds : List[float] or List[List[float]], optional
        Bounds of state B. Required when mode="state".

    temp : float, optional
        Temperature in Kelvin. Required if `kbt` is not provided.

    units : str, optional
        Units of the FES if using `temp`, by default "kJ/mol".

    kbt : float, optional
        Thermal energy in the same units as the FES.
        Required if `temp` is not provided.

    intervals : int, optional
        Number of intervals on which deltaG is progressively computed,
        by default 10.

    weights : np.ndarray, optional
        Weights associated with the data points.

    bias : np.ndarray, optional
        Bias values used to compute weights as exp(bias / kbt).

    reverse : bool, optional
        Whether to reverse the data before computing deltaG.

    time : np.ndarray, optional
        Time reference for the input data.

    plot : bool, optional
        Whether to plot deltaG.

    plot_color : str, optional
        Color for the deltaG plot.

    ax : matplotlib.axes, optional
        Axis object to plot into.

    eps : float, optional
        Small regularization term to prevent logarithm from having zero argument.

    mode : {"state", "states", "funnel"}, optional
        Calculation mode.

    rfunnel : float, optional
        Funnel radius in nm. Required when mode="funnel".

    bat : float, optional
        Bound cutoff. The bound region is z < bat.
        Required when mode="funnel".

    uat : float, optional
        Unbound cutoff. The unbound plateau is z > uat.
        Required when mode="funnel".

    bandwidth : float, optional
        KDE bandwidth used in compute_fes when mode="funnel".

    num_samples : int, optional
        Number of grid points used in compute_fes when mode="funnel".

    bounds : List[float], optional
        Bounds used in compute_fes when mode="funnel".

    kernel : str, optional
        KDE kernel used in compute_fes when mode="funnel".

    backend : str, optional
        Backend used in compute_fes when mode="funnel".

    Returns
    -------
    grid : np.ndarray
        Interval or time grid.

    deltaG : np.ndarray
        DeltaG values computed up to each interval.
    """

    mode = mode.lower()
    if mode not in ["state", "states", "funnel"]:
        raise ValueError("mode must be either 'state', 'states', or 'funnel'.")

    # Ensure numpy array
    X = np.asarray(X)

    # Funnel mode only supports 1D data
    if mode == "funnel":
        X = X.reshape(-1)

    n_samples = len(X)

    # Check input consistency
    if weights is not None and bias is not None:
        raise ValueError(
            "The weights and bias keywords cannot be defined together, use only one of them!"
        )

    if weights is not None and n_samples != len(weights):
        raise ValueError(
            f"Input data and weights must have the same number of entries! "
            f"Found {n_samples} and {len(weights)}."
        )

    if bias is not None and n_samples != len(bias):
        raise ValueError(
            f"Input data and bias must have the same number of entries! "
            f"Found {n_samples} and {len(bias)}."
        )

    if time is not None and n_samples != len(time):
        raise ValueError(
            f"Input data and time must have the same number of entries! "
            f"Found {n_samples} and {len(time)}."
        )

    # Check temperature / units
    kbt, temp, units = _check_kbt_units(kbt, temp, units)

    # Prepare weights
    if bias is not None:
        bias = np.asarray(bias)

    if weights is None:
        if bias is None:
            weights = np.ones(n_samples)
        else:
            weights = np.exp(bias / kbt)
    else:
        weights = np.asarray(weights)

    if time is not None:
        time = np.asarray(time)

    # Reverse data if requested
    if reverse:
        X = np.flip(X, axis=0)
        weights = np.flip(weights, axis=0)
        if time is not None:
            time = np.flip(time, axis=0)

    deltaG = []

    # ------------------------------------------------------------------
    # Mode 1: original state-based deltaG
    # ------------------------------------------------------------------
    if mode in ["state", "states"]:

        if stateA_bounds is None or stateB_bounds is None:
            raise ValueError(
                "stateA_bounds and stateB_bounds must be provided when mode='state'."
            )

        stateA_bounds = np.array(stateA_bounds)
        stateB_bounds = np.array(stateB_bounds)

        n_dim = 1
        if X.ndim > 1 and X.shape[-1] == 2:
            n_dim = 2
            if stateA_bounds.shape[-1] != n_dim or stateB_bounds.shape[-1] != n_dim:
                raise ValueError("Input data are 2D, state bounds must be 2D as well!")

        # Define masks for states A and B
        if n_dim == 1:
            X_1d = X.reshape(-1)
            mask_A = np.logical_and(X_1d > stateA_bounds[0], X_1d < stateA_bounds[1])
            mask_B = np.logical_and(X_1d > stateB_bounds[0], X_1d < stateB_bounds[1])
        else:
            mask_A = np.logical_and(
                np.logical_and(
                    X[:, 0] > stateA_bounds[0, 0],
                    X[:, 0] < stateA_bounds[0, 1],
                ),
                np.logical_and(
                    X[:, 1] > stateA_bounds[1, 0],
                    X[:, 1] < stateA_bounds[1, 1],
                ),
            )

            mask_B = np.logical_and(
                np.logical_and(
                    X[:, 0] > stateB_bounds[0, 0],
                    X[:, 0] < stateB_bounds[0, 1],
                ),
                np.logical_and(
                    X[:, 1] > stateB_bounds[1, 0],
                    X[:, 1] < stateB_bounds[1, 1],
                ),
            )

        # Build intervals
        interval_len = n_samples / intervals
        interval_bounds = np.arange(0, n_samples, interval_len)
        interval_bounds = np.ceil(interval_bounds).astype("int")
        interval_bounds = np.concatenate(
            (interval_bounds, np.array([n_samples - 1]))
        )

        # Progressive accumulation
        tot_A = eps
        tot_B = eps

        for i in range(intervals):
            start = interval_bounds[i]
            end = interval_bounds[i + 1]

            aux_A = weights[start:end][mask_A[start:end]]
            aux_B = weights[start:end][mask_B[start:end]]

            tot_A += np.sum(aux_A)
            tot_B += np.sum(aux_B)

            G_A = -kbt * np.log(tot_A)
            G_B = -kbt * np.log(tot_B)

            deltaG.append(G_B - G_A)

        deltaG = np.array(deltaG)

        # Switch to time if needed
        if time is not None:
            interval_bounds = time[interval_bounds]

        grid = interval_bounds[1:]

        ylabel = (
            f"$\\Delta$G [{units}]"
            if units is not None
            else "$\\Delta$G"
        )

    # ------------------------------------------------------------------
    # Mode 2: funnel-corrected deltaG
    # ------------------------------------------------------------------
    elif mode == "funnel":

        if rfunnel is None or bat is None or uat is None:
            raise ValueError(
                "rfunnel, bat, and uat must be provided when mode='funnel'."
            )

        if rfunnel <= 0:
            raise ValueError("rfunnel must be positive.")

        if bat >= uat:
            raise ValueError("bat should be smaller than uat.")

        # Original funnel prefactor:
        # pref = pi * (1 / 1.66) * rfunnel^2
        prefactor_funnel = np.log(np.pi * rfunnel**2 / 1.66)

        # Build cumulative intervals
        interval_len = n_samples / intervals
        interval_bounds = np.arange(0, n_samples, interval_len)
        interval_bounds = np.ceil(interval_bounds).astype(int)
        interval_bounds = np.concatenate(
            (interval_bounds, np.array([n_samples]))
        )

        for i in range(intervals):
            end = interval_bounds[i + 1]

            X_i = X[:end]
            w_i = weights[:end]

            fes, grid_fes, used_bounds, error = compute_fes(
                X=X_i,
                kbt=kbt,
                num_samples=num_samples,
                bounds=bounds,
                bandwidth=bandwidth,
                kernel=kernel,
                weights=w_i,
                scale_by=None,
                blocks=1,
                fes_to_zero=None,
                plot=False,
                backend=backend,
                eps=eps,
            )

            grid_fes = np.asarray(grid_fes)
            fes = np.asarray(fes)

            mask_bound = grid_fes < bat
            mask_unbound = grid_fes > uat

            if not np.any(mask_bound):
                raise ValueError(
                    "No grid points found in the bound region. Check bat or bounds."
                )

            if not np.any(mask_unbound):
                raise ValueError(
                    "No grid points found in the unbound region. Check uat or bounds."
                )

            # Same as original script:
            # shift2 = average FES in the unbound plateau
            shift_unbound = np.average(fes[mask_unbound])

            # Grid spacing along z
            dz = (grid_fes[-1] - grid_fes[0]) / (len(grid_fes) - 1)
            extra_factor = np.log(dz)

            # Shifted FES
            fes_shifted = fes - shift_unbound

            # Bound free energy integral
            G_bound = -kbt * np.logaddexp.reduce(
                -fes_shifted[mask_bound] / kbt
            )

            # Original formula:
            # deltafunnel = -(G_bound - kbt * (prefactor_funnel + extra_factor))
            delta_funnel = -(
                G_bound - kbt * (prefactor_funnel + extra_factor)
            )

            deltaG.append(delta_funnel)

        deltaG = np.asarray(deltaG)

        if time is not None:
            grid = time[interval_bounds[1:] - 1]
        else:
            grid = interval_bounds[1:]

        ylabel = (
            f"$\\Delta G_{{funnel}}$ [{units}]"
            if units is not None
            else "$\\Delta G_{funnel}$"
        )

    # Plot if needed
    if plot:
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(grid, deltaG, color=plot_color)
        ax.set_xlabel("Time" if time is not None else "Frame")
        ax.set_ylabel(ylabel)

    return grid, deltaG


def _check_kbt_units(kbt, temp, units):
    "Helper function to handle inputs to specify free energy units in free energy utils"
    if kbt is not None:
        if temp is not None:
            raise ValueError("Only one of kbt and temp can be specified.")
     
        units = None
    else: 
        if temp is None:
            raise ValueError("One of kbt and temp must be specified.")
    
        if units == "kJ/mol":
                kb = 0.00831441
        elif units == "kcal/mol":
            kb = 0.0019872041
        elif units == "eV":
            kb = 8.6173324e-5
        else:
            raise ValueError(
                "units must be one of 'kJ/mol', 'kcal/mol', 'eV'."
            )
        kbt = kb * temp

    return kbt, temp, units


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
