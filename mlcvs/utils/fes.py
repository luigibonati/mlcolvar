__all__ = ["compute_fes"]

from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor

def compute_fes(X, temp=300, num_samples=100, bounds=None, bandwidth=0.01, kernel='gaussian', weights=None, scale_by = None, blocks = 1, plot=False, plot_max_fes=None, ax = None):
    """Compute the Free Energy Surface along the given variables.

    Parameters
    ----------
    X : array-like, list of arrays
        data
    temp : float, optional
        temperature, by default 300
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
    plot : bool, optional
        Plot the results (only available for 1D and 2D FES), by default False
    plot_max_fes : float, optional
        Maximum value of the FES to be plotted, by default None
    ax : matplotlib axis, optional
        Axis where to plot, default create new figure
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
    - This function uses sklearn.neighbors.KernelDensity function to construct the estimate of the FES (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html). You can specify all the kernels supported.

    - Note that if ``scale_by`` is not none, the bandwidth is not rescaled as well. So if using ``scale_by='range'`` a bandwidth of 0.01 corresponds to a 1/100 of the range of values assumed by the variable.

    - If blocks>1 a (weighted) block analyis is performed [1], which allows also to obtain an estimate of the uncertainty.

    [1] Invernizzi, Piaggi and Parrinello, PRX, 2020,10,4

    """
    # temperature
    kb = 0.00831441
    kbt = kb * temp

    # dataset
    if type(X) == list:
        X = np.vstack(X).T
    elif type(X) == pd.DataFrame:
        X = X.values
    elif type(X) == Tensor:
        X = X.numpy()
    
    if X.ndim == 1:
        X = X.reshape([-1,1])  

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
        if scale_by == 'std':
            scale = X.std(axis=0)
        elif scale_by == 'range':
            scale = X.max(axis=0)-X.min(axis=0)
        elif type(scale_by) == list:
            scale = np.asarray(scale_by)
        X = np.copy(X)
        X /= scale

    # eval points
    if dim == 1:
        if bounds is None:
            bounds = (X[:,0].min(), X[:,0].max())
        grid = np.linspace(bounds[0], bounds[1], num_samples)
        positions = grid.reshape([-1,1])
    else:
        if bounds is None:
            bounds = [(X[:,i].min(), X[:,i].max()) for i in range(dim)]
        grid = np.meshgrid(*[np.linspace(b[0], b[1], num_samples) for b in bounds])
        positions = np.vstack([g.ravel() for g in grid]).T

    # divide in blocks
    X_b = np.array_split(X,blocks)
    w_b = np.array_split(weights,blocks)

    O_i,W_i = [], [] #values per block
    # block average
    for i in range(blocks):
        # data of each block
        X_i = X_b[i]
        w_i = w_b[i]

        # fit
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(X_i, sample_weight=w_i)

        # logpdf --> fes
        ##positions = torch.tensor(positions, dtype=torch.float32)
        fes = - kbt * kde.score_samples(positions).reshape([num_samples for i in range(dim)])
        fes -= fes.min()

        # result for each block
        O_i.append(fes)
        W_i.append(np.sum(w_i))

    O_i = np.asarray(O_i)
    W_i = np.asarray(W_i)

    # compute avg and std 
    if blocks > 1: 
        # weighted average
        O = np.dot(O_i.T,W_i)/np.sum(W_i)
        # weighted std
        dev = O_i - O
        blocks_eff = (np.sum(W_i))**2/(np.sum(W_i**2))
        variance = blocks_eff/(blocks_eff - 1) * (np.dot((dev**2).T,W_i) )/(np.sum(W_i))
        error = np.sqrt(variance/blocks_eff)
    else:
        O = O_i[0]
        error = None

    # rescale back
    if scale_by is not None:
        if dim == 1:
            bounds = (bounds[0]*scale[0], bounds[1]*scale[0])
            grid *= scale
        else:
            bounds = [(bounds[i][0]*scale[i], bounds[i][1]*scale[i]) for i in range(dim)]
            for i in range(dim):
                grid[i] *= scale[i]

    # Plot
    if plot:
        if ax is None:
            fig,ax = plt.subplots()
        if dim == 1:
            fes2 = np.copy(O)
            if plot_max_fes is not None:
                fes2[fes2>plot_max_fes] = np.nan
            if blocks > 1:
                ax.errorbar(grid,fes2,error)
            else:
                ax.plot(grid,fes2)  
            ax.set_ylabel('FES [kJ/mol]')
        elif dim == 2: 
            fes2 = np.copy(O)
            if plot_max_fes is not None:
                fes2[fes2>plot_max_fes] = np.nan
            extent = [item for sublist in bounds for item in sublist] 
            pp = ax.contourf(fes2,cmap='fessa',extent=extent)#,vmax=max_fes)
            cbar = plt.colorbar(pp,ax=ax)
            cbar.set_label('FES [kJ/mol]')

    return fes,grid,bounds,error