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
    X: np.ndarray = None,
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
    trajectories: List[np.ndarray] = None,
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
    trajectories : list of arrays, optional
        Independent trajectories, each of shape (n_frames,) or (n_frames, n_features).
        Each trajectory is analyzed independently on the same KDE grid.
        The mean and standard error are computed across trajectories.
        Cannot be combined with blocks > 1.
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
        Standard error estimate from block averaging or independent trajectories. Same shape as `fes`.

    Notes
    -----
    - The KDE can be computed using either KDEpy (recommended for speed in 2D) or scikit-learn. Use `backend="KDEpy"` or `backend="sklearn"` to specify.
    - Either `temp` or `kbt` must be provided (not both). If using `temp`, ensure `units` is set correctly.
    - If `scale_by` is used, input variables are rescaled before KDE. This means that if using ``scale_by='range'`` a bandwidth of 0.01 corresponds to a 1/100 of the range of values assumed by the variable.
    - If `blocks > 1`, the function performs block averaging to estimate uncertainty [1].
    - If `trajectories` is provided, the function computes the mean and standard error across independent trajectories.
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

    # independent trajectories
    multi_traj = trajectories is not None
    if multi_traj:
        if X is not None:
            raise ValueError("Use either X or trajectories, not both.")
        if blocks != 1:
            raise ValueError("blocks and trajectories cannot be used together.")
        if len(trajectories) == 0:
            raise ValueError("trajectories must not be empty.")

        traj_arrays = []
        for item in trajectories:
            if PANDAS_IS_INSTALLED and isinstance(item, pd.DataFrame):
                item = item.values
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            item = np.asarray(item)
            if item.ndim == 1:
                item = item[:, None]
            if item.ndim != 2 or item.shape[1] not in (1, 2) or len(item) == 0:
                raise ValueError("Each trajectory must have shape (n,), (n, 1), or (n, 2).")
            traj_arrays.append(item)

        if len({arr.shape[1] for arr in traj_arrays}) != 1:
            raise ValueError("All trajectories must have the same CV dimension.")

        traj_lengths = [len(a) for a in traj_arrays]
        traj_breaks = np.cumsum(traj_lengths)[:-1]
        X = np.concatenate(traj_arrays, axis=0)

        if weights is not None and bias is not None:
            raise ValueError("The weights and bias keywords cannot be defined together, use only one of them!")

        if weights is not None or bias is not None:
            name = "weights" if weights is not None else "bias"
            values = weights if weights is not None else bias

            if not isinstance(values, (list, tuple)) or len(values) != len(traj_lengths):
                raise ValueError(f"{name} must contain one array per trajectory.")

            converted = [np.asarray(v) for v in values]
            if any(v.ndim != 1 or len(v) != n for v, n in zip(converted, traj_lengths)):
                raise ValueError(f"{name} arrays must match their trajectory lengths.")

            if weights is not None:
                weights = np.concatenate(converted)
            else:
                bias = np.concatenate(converted)

    elif X is None:
        raise ValueError("Either X or trajectories must be provided.")

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

    # divide in blocks or independent trajectories
    if multi_traj:
        X_b = np.split(X, traj_breaks)
        w_b = np.split(weights, traj_breaks)
    else:
        X_b = np.array_split(X, blocks)
        w_b = np.array_split(weights, blocks)

    n_estimates = len(X_b)

    fes_blocks, W_blocks = [], []  # values per block
    # block average
    for i in range(n_estimates):
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
        W_blocks.append(1.0 if multi_traj else np.sum(w_i))

    fes_blocks = np.asarray(fes_blocks)
    W_blocks = np.asarray(W_blocks)

    # compute avg and std
    if n_estimates > 1:
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
            if n_estimates > 1:
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

def compute_deltaG(X: Union[np.ndarray, List[np.ndarray]],
                   stateA_bounds: Union[List[float], List[List[float]], np.ndarray],
                   stateB_bounds: Union[List[float], List[List[float]], np.ndarray],
                   temp=None,
                   units="kJ/mol",
                   kbt: float = None,
                   intervals: int = 10,
                   weights=None,
                   bias=None,
                   reverse: bool = False,
                   time=None,
                   plot: bool = False,
                   plot_color: str = "fessa6",
                   plot_error_style: str = "fill_between",
                   ax: matplotlib.axes = None,
                   eps: float = 1e-8,
                   rfunnel: float = None,
                   c0: float = 1.0):
    """Compute progressive free-energy differences from independent trajectories.

    ``X`` can be one 1D/2D NumPy array or a list of independent trajectory
    arrays (which may have different lengths). ``weights``, ``bias`` and
    ``time`` follow the same structure as ``X``. Separate trajectories are
    averaged *after* calculating their own free-energy differences; input
    samples are never split into artificial blocks.

    ``intervals`` controls the number of progressive estimates. For multiple
    trajectories without ``time``, the grid is fractional trajectory progress
    from 1/intervals to 1. With time arrays, times are converted to elapsed
    time relative to the first frame, and all trajectories are compared on the
    common time window up to the shortest trajectory duration.

    With two or more trajectories, ``error`` is the standard error of the
    independent deltaG estimates (sample std / sqrt(n_trajectories)). With
    one trajectory, ``error`` is None. For funnel correction, state A denotes
    the bound region and state B the unbound region.

    Returns
    -------
    grid : np.ndarray
        Frame index (single trajectory without time), fractional progress
        (multiple trajectories without time), or time coordinate.
    deltaG : np.ndarray
        Progressive deltaG, averaged over trajectories when applicable.
    error : np.ndarray or None
        Standard error across independent trajectories, or None.
    """
    if intervals < 1:
        raise ValueError("intervals must be at least 1.")
    if eps < 0:
        raise ValueError("eps must be nonnegative.")
    if weights is not None and bias is not None:
        raise ValueError("Specify either weights or bias, not both.")
    if rfunnel is not None and rfunnel <= 0:
        raise ValueError("rfunnel must be positive.")
    if rfunnel is not None and c0 <= 0:
        raise ValueError("c0 must be positive.")

    # A list of arrays means independent simulations. A list of scalar CV
    # values, like [0.1, 0.2], still denotes a single 1D trajectory.
    multiple = (isinstance(X, (list, tuple)) and len(X) > 0
                and all(np.asarray(item).ndim >= 1 for item in X))
    trajectories = [np.asarray(item) for item in X] if multiple else [np.asarray(X)]
    if not trajectories or any(len(item) == 0 for item in trajectories):
        raise ValueError("Each trajectory must contain at least one frame.")

    n_traj = len(trajectories)

    def _series(arg, name):
        if arg is None:
            return [None] * n_traj
        if multiple:
            if not isinstance(arg, (list, tuple)) or len(arg) != n_traj:
                raise ValueError(f"{name} must contain one array per trajectory.")
            return [None if item is None else np.asarray(item) for item in arg]
        return [np.asarray(arg)]

    weight_list = _series(weights, "weights")
    bias_list = _series(bias, "bias")
    time_list = _series(time, "time")
    if multiple and time is not None and any(t is None for t in time_list):
        raise ValueError("All independent trajectories must provide time arrays.")
    if multiple and bias is not None and any(b is None for b in bias_list):
        raise ValueError("Bias is missing for an input trajectory.")
    if multiple and weights is not None and any(w is None for w in weight_list):
        raise ValueError("Weights are missing for an input trajectory.")

    kbt, temp, units = _check_kbt_units(kbt, temp, units)
    stateA_bounds = np.asarray(stateA_bounds)
    stateB_bounds = np.asarray(stateB_bounds)
    use_funnel = rfunnel is not None

    records = []
    for x, w, b, t in zip(trajectories, weight_list, bias_list, time_list):
        if x.ndim == 1:
            dim = 1
        elif x.ndim == 2 and x.shape[1] in (1, 2):
            dim = x.shape[1]
        else:
            raise ValueError("Each trajectory must have shape (n,), (n, 1), or (n, 2).")
        if use_funnel and dim != 1:
            raise ValueError("Funnel correction only supports 1D data.")

        n = len(x)
        for name, arg in (("weights", w), ("bias", b), ("time", t)):
            if arg is not None and (arg.ndim != 1 or len(arg) != n):
                raise ValueError(f"{name} must be a 1D array of length {n}.")

        if w is None:
            w = np.ones(n) if b is None else np.exp(b / kbt)
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("Weights must be finite and nonnegative.")
        if dim == 1:
            if stateA_bounds.shape != (2,) or stateB_bounds.shape != (2,):
                raise ValueError("1D state bounds must each have shape (2,).")
            xx = x.reshape(-1)
            mask_a = (xx > stateA_bounds[0]) & (xx < stateA_bounds[1])
            mask_b = (xx > stateB_bounds[0]) & (xx < stateB_bounds[1])
        else:
            if stateA_bounds.shape != (2, 2) or stateB_bounds.shape != (2, 2):
                raise ValueError("2D state bounds must each have shape (2, 2).")
            mask_a = np.all((x > stateA_bounds[:, 0]) &
                            (x < stateA_bounds[:, 1]), axis=1)
            mask_b = np.all((x > stateB_bounds[:, 0]) &
                            (x < stateB_bounds[:, 1]), axis=1)

        if reverse:
            w, mask_a, mask_b = w[::-1], mask_a[::-1], mask_b[::-1]
            if t is not None:
                # Use increasing elapsed time for the reversed trajectory.
                t = (t[-1] - t[::-1])
        elif t is not None:
            t = t - t[0]

        if t is not None and (not np.all(np.isfinite(t)) or
                              np.any(np.diff(t) <= 0)):
            raise ValueError("Trajectory time arrays must be strictly increasing.")
        records.append((w, mask_a, mask_b, t))

    if n_traj > 1 and time is not None:
        # A true common elapsed-time axis, without interpolating deltaG values.
        max_time = min(t[-1] for _, _, _, t in records)
        if max_time <= 0:
            raise ValueError("Each trajectory needs at least two time points.")
        grid = np.linspace(max_time / intervals, max_time, intervals)
    elif n_traj > 1:
        grid = np.arange(1, intervals + 1, dtype=float) / intervals
    else:
        n = len(trajectories[0])
        if intervals > n:
            raise ValueError("intervals cannot exceed the number of frames.")
        indices = np.ceil(np.arange(1, intervals + 1) * n / intervals).astype(int) - 1
        grid = records[0][3][indices] if time is not None else indices

    correction = (np.pi * rfunnel**2 * c0 / 1.66) if use_funnel else None
    estimates = []
    for w, mask_a, mask_b, t in records:
        n = len(w)
        if n_traj > 1 and time is not None:
            # Include frames at or before each common physical time point.
            ends = np.searchsorted(t, grid, side="right")
        elif n_traj > 1:
            ends = np.ceil(grid * n).astype(int)
        else:
            ends = indices + 1
    
        cum_a = np.concatenate(([0.0], np.cumsum(np.where(mask_a, w, 0.0))))
        cum_b = np.concatenate(([0.0], np.cumsum(np.where(mask_b, w, 0.0))))
        pa = cum_a[ends]
        pb = cum_b[ends]
        # Do not treat a basin that has never been visited as a physical
        # estimate: eps prevents log(0), but cannot create missing samples.
        with np.errstate(divide="ignore", invalid="ignore"):
            dg = kbt * np.log((pa + eps) / (pb + eps))
        if use_funnel:
            dg = -kbt * np.log((pa + eps) / (pb + eps) * correction)
        else:
            dg = kbt * np.log((pa + eps) / (pb + eps))
        dg = np.where((pa > 0) & (pb > 0), dg, np.nan)
        estimates.append(dg)

    estimates = np.stack(estimates)
    # Only report mean/SEM where all runs supplied valid estimates, so the
    # uncertainty always refers to the same n independent trajectories.
    valid = np.all(np.isfinite(estimates), axis=0)
    deltaG = np.full(intervals, np.nan)
    error = np.full(intervals, np.nan) if n_traj > 1 else None
    if np.any(valid):
        deltaG[valid] = np.mean(estimates[:, valid], axis=0)
        if error is not None:
            error[valid] = np.std(estimates[:, valid], axis=0, ddof=1) / np.sqrt(n_traj)

    if plot:
        if ax is None:
            _, ax = plt.subplots()
        if error is not None and plot_error_style is not None:
            if plot_error_style == "fill_between":
                ax.plot(grid, deltaG, color=plot_color)
                ax.fill_between(grid, deltaG - error, deltaG + error,
                                color=plot_color, alpha=0.3)
            elif plot_error_style == "errorbar":
                ax.errorbar(grid, deltaG, yerr=error,
                            color=plot_color, alpha=0.5)
            else:
                raise ValueError("plot_error_style must be 'errorbar', 'fill_between', or None.")
        else:
            ax.plot(grid, deltaG, color=plot_color)
        xlabel = ("Time" if time is not None else
                  "Trajectory fraction" if n_traj > 1 else "Frame")
        ax.set_xlabel(xlabel)
        label = r"$\Delta G_{funnel}$" if use_funnel else r"$\Delta G$"
        ax.set_ylabel(f"{label} [{units}]" if units is not None else label)
    return grid, deltaG, error


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
