import numpy as np
import torch
from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
import mlcolvar.utils.plot

__all__ = [ "sensitivity_analysis", "plot_sensitivity" ]

def sensitivity_analysis(
    model,
    dataset,
    std=None,
    feature_names=None,
    metric="mean_abs_val",
    per_class=False,
    plot_mode="violin",
    ax=None,
):
    """Perform a sensitivity analysis using the partial derivatives method. This allows us to measure which input features the model is most sensitive to (i.e., which quantities produce significant changes in the output).

    To do this, the partial derivatives of the model with respect to each input :math:`x_i` are computed over a set of `N` points of a :math:`$$\{\mathbf{x}^{(j)}\}_{j=1} ^N$$` dataset.
    These values, in the case where the dataset is not standardized, are multiplied by the standard deviation of the features over the dataset.

    Then, an average sensitivity value :math:`s_i` is computed, either as the mean absolute value (metric=`MAV`):
    .. math:: s_i = \frac{1}{N} \sum_j \left|{\frac{\partial s}{\partial x_i}(\mathbf{x}^{(j)})}\right| \sigma_i

    or as the root mean square (metric=`RMS`):
    .. math:: s_i = \sqrt{\frac{1}{N} \sum_j \left({\frac{\partial s}{\partial x_i}(\mathbf{x}^{(j)})}\  \sigma_i\right)^2 }

    In alternative, one can also compute simply average, without taking the absolute values (metric=`mean`).
    
    In all the above cases, the sensitivity values are normalized such that they sum to 1.

    In case in which a labeled dataset these quantities can be computed also on the subset of the data belonging to each class.

    See also
    --------
    mlcolvar.utils.fes.plot_sensitivity
        Plot the sensitivity analysis results

    Parameters
    ----------
    model : mlcolvar.cvs.BaseCV
        collective variable model
    dataset : mlcovar.data.DictDataset
        dataset on which to compute the sensitivity analysis.
    std : array_like, optional
        standard deviation of the features, by default it will be computed from the dataset
    feature_names : array-like, optional
        array-like with input features names, by default they will be taken from the dataset if available
    metric : str, optional
        sensitivity measure ('mean_abs_val'|'MAV','root_mean_square'|'RMS','mean'), by default 'mean_abs_val'
    per_class : bool, optional
        if the dataset has labels, compute also the sensitivity per class, by default False
    plot_mode : str, optional
        how to visualize the results ('violin','barh','scatter'), by default 'violin'
    ax : matplotlib.axis, optional
        ax where to plot the results, by default it will be initialized

    Returns
    -------
    results: dictionary
        results of the sensitivity analysis, containing 'feature_names', the 'sensitivity' and the 'gradients' per samples, ordered according to the sensitivity.
    """

    # get dataset
    X = dataset["data"]
    n_inputs = X.shape[1]

    # get feature names
    if feature_names is None:
        if dataset.feature_names is not None:
            feature_names = dataset.feature_names
        else:
            feature_names = np.asarray([str(i + 1) for i in range(n_inputs)])

    # get standard deviation
    if std is None:
        std = dataset.get_stats()["data"]["std"].detach().numpy()
    else: 
        std = np.asarray(std)

    # compute cv
    X.requires_grad = True
    s = model(X)

    # get gradients
    grad_output = torch.ones_like(s)
    grad = torch.autograd.grad(s, X, grad_outputs=grad_output)[0].detach().cpu().numpy()
    if metric != "mean":
        grad = np.abs(grad)

    # multiply grad_xi by std_xi
    grad = grad * std

    # normalize such that the average of the abs sums to 1
    grad /= np.abs(grad).mean(axis=0).sum()

    # get metrics
    def _compute_score(grad, metric):
        if (metric == "mean_abs_val") | (metric == "MEAN_ABS") | (metric == "MAV") | (metric == 'mean'):
            score = grad.mean(axis=0)
        elif (metric == "root_mean_square") | (metric == "rms") | (metric == "RMS"):
            score = np.sqrt((grad**2).mean(axis=0))
        else:
            raise NotImplementedError(
                "only `mean_abs_value` (MAV) or `root_mean_square` (RMS), or `mean` metrics are allowed"
            )
        return score

    score = _compute_score(grad, metric)

    # sort features based on (absolute) sensitivity
    index = np.abs(score).argsort()
    feature_names = np.asarray(feature_names)[index]
    score = score[index]
    grad = grad[:, index]

    # store into results
    out = {}
    out["feature_names"] = feature_names
    out["sensitivity"] = {"Dataset": score}
    out["gradients"] = {"Dataset": grad}

    # per class statistics
    if per_class:
        try:
            labels = dataset["labels"].numpy().astype(int)
        except KeyError:
            raise KeyError(
                "Per class analyis requested but no labels found in the given dataset."
            )

        unique_labels = np.unique(labels)
        for i, l in enumerate(unique_labels):
            mask = np.argwhere(labels == l)[:, 0]
            grad_l = grad[mask, :]
            score_l = _compute_score(grad_l, metric)
            out["sensitivity"][f"State {l}"] = score_l
            out["gradients"][f"State {l}"] = grad_l

    # plot
    if plot_mode is not None:
        plot_sensitivity(out, mode=plot_mode, ax=ax)

    return out

def plot_sensitivity(results, mode="violin", per_class=None, max_features = 100, ax=None):
    """Plot results of the sensitivity analysis. They can be plotted in three modes:
    * Violin plot ('violin'), showing the density of per-sample sensitivities besides the mean value
    * Scatter ('scatter'), plotting the mean and standard deviation of the gradients
    * Horizontal bar plot ('barh') only displaying the mean of the gradients

    Parameters
    ----------
    results : dictionary
        sensitity results calculated by sensitivity_analysis
    mode : string, optional
        ('violin','barh','scatter'), by default 'violin'
    per_class : bool, optional
        plot per-class statistics if available, by default plot them if available
    max_features : int, optional
        plot at most max_features, by default 100
    ax : matplotlib axis, optional
        ax where to plot the results, by default it will be initialized

    See also
    --------
    mlcolvar.utils.explain.sensitivity_analysis
        Perform a sensitivity analysis

    Returns
    -------
    ax
        return the generated matplotlib axis if not passed
    """

    # retrieve info from results dictionary
    feature_names = results["feature_names"]
    n_inputs = len(feature_names)
    if max_features < n_inputs:
        print(f'Plotting only the first {max_features} features out of {n_inputs}.')
        feature_names = feature_names[-max_features:]
        n_inputs = max_features
    
    in_num = np.arange(n_inputs)
    n_results = len(results["sensitivity"].keys())

    # check whether to plot per-class statistics
    if per_class is None:
        per_class = True if n_results > 1 else False
    else:
        if not type(per_class) == bool:
            raise TypeError("per_class should be either `True`, `False` or `None`.")
        if per_class & (n_results == 1):
            raise KeyError(
                "Per class analyis requested but no labels found in the results dictionary. You need to call `sensitivity_analysis` with `per_class`=True. "
            )

    # initialize ax
    if ax is None:
        fig = plt.figure(figsize=(5, 0.25 * n_inputs))
        ax = fig.add_subplot(111)
        ax.set_title("Sensitivity Analysis")

    # define utils functions
    def _set_violin_attributes(violin_parts, color, alpha=0.5, label=None, zorder=None):
        for pc in violin_parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(alpha)
            if zorder is not None:
                pc.set_zorder(zorder)
        if label is not None:
            patch_label = (mpatches.Patch(color=color, alpha=alpha), label)
            return patch_label

    patch_labels = []

    patterns = ["", "", "/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]

    for i, label in enumerate(results["sensitivity"].keys()):
        score = results["sensitivity"][label][-max_features:]
        grad = results["gradients"][label][:,-max_features:]

        color = "fessa0" if "Dataset" in label else f"fessa{7-i}"

        if mode == "barh":
            alpha = 0.6 if "Dataset" in label else 0.4
            height = 0.8 if "Dataset" in label else 0.4
            ax.barh(
                in_num,
                score,
                height=height,
                color=color,
                edgecolor="k",
                hatch=patterns[i],
                alpha=alpha,
                label=label,
            )
        elif mode == "violin":
            widths = 0 if (("Dataset" in label) & per_class) else 0.5
            zorder = 1 if "Dataset" in label else 0
            showmeans = True if "Dataset" in label else False
            violin_parts = ax.violinplot(
                grad,
                positions=in_num,
                vert=False,
                showmeans=showmeans,
                showextrema=False,
                widths=widths,
            )
            patch_label = _set_violin_attributes(
                violin_parts, color, alpha=0.5, label=label, zorder=zorder
            )
            patch_labels.append(patch_label)
            if "Dataset" in label:
                ax.scatter(y=in_num, x=score, c="dimgrey", s=10, zorder=2)
        elif mode == "scatter":
            fmt = "D" if "Dataset" in label else "."
            ax.errorbar(
                y=in_num,
                x=score,
                xerr=grad.std(axis=0),
                color=color,
                fmt=fmt,
                alpha=0.5,
                label=label,
            )
        else:
            raise NotImplementedError(
                'plot mode can be only "barh","violin","scatter".'
            )

        if not per_class:
            break

    # >> legend
    ax.set_xlabel("Sensitivity")
    ax.set_yticks(in_num)
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_ylabel("Input features")

    if mode == "violin":
        ax.legend(*zip(*patch_labels), loc="lower right", frameon=False)
    else:
        ax.legend(loc="lower right", frameon=False)
    if np.min(results["sensitivity"]["Dataset"])>=0: 
        ax.set_xlim(0, None)
    else:
        ax.axvline(0,color='grey')
    ax.set_ylim(-1, in_num[-1] + 1)

def test_sensitivity_analysis():
    from mlcolvar.data import DictDataset
    from mlcolvar.cvs import DeepLDA

    n_states = 2
    in_features, out_features = 2, n_states - 1
    layers = [in_features, 5, 5, out_features]

    # create dataset
    samples = 10
    X = torch.randn((samples * n_states, 2))

    # create labels
    y = torch.zeros(X.shape[0])
    for i in range(1, n_states):
        y[samples * i :] += 1

    dataset = DictDataset({"data": X, "labels": y})

    # define CV
    opts = {
        "nn": {"activation": "shifted_softplus"},
    }
    model = DeepLDA(layers, n_states, options=opts)

    # feature importances
    for per_class in [True, False, None]:
        for names in [None, ["x", "y"], np.asarray(["x", "y"])]:
            results = sensitivity_analysis(
                model, dataset, feature_names=names, per_class=per_class, plot_mode=None
            )
