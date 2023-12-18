import numpy as np
import torch

from mlcolvar.utils.plot import plot_sensitivity

def sensitivity_analysis(model, dataset, feature_names = None, metric='mean_abs_val', per_class=False, plot_mode='violin', ax=None):
    """Perform a sensitivity analysis to measure which input features the model is most sensitive to (i.e., which quantities produce significant changes in the output).
    
    To do this, the partial derivatives of the model with respect to each input :math:`x_i` are computed over a set of `N` points of a :math:`$$\{\mathbf{x}^{(j)}\}_{j=1} ^N$$` dataset. 
    These values, in the case where the dataset is not standardized, are multiplied by the standard deviation of the features over the dataset.

    Then, an average sensitivity value :math:`s_i` is computed, either as the mean absolute value (metric=`MAV`):
    .. math:: s_i = \frac{1}{N} \sum_j \left|{\frac{\partial s}{\partial x_i}(\mathbf{x}^{(j)})}\right| \sigma_i

    or as the root mean square (metric=`RMS`):
    .. math:: s_i = \sqrt{\frac{1}{N} \sum_j \left({\frac{\partial s}{\partial x_i}(\mathbf{x}^{(j)})}\  \sigma_i\right)^2 }

    The sensitivity values are normalized such that they sum to 1.

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
    feature_names : _type_, optional
        array-like with input features names, by default it takes them from the dataset if available
    metric : str, optional
        sensitivity measure ('mean_abs_val'|'MAV','root_mean_square'|'RMS')', by default 'mean_abs_val'
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
    X = dataset['data']
    std = dataset.get_stats()['data']['std'].detach().numpy()
    n_inputs = X.shape[1]

    # get feature names
    if feature_names is None:
        if dataset.feature_names is not None:
            feature_names = dataset.feature_names
        else:
            feature_names = np.asarray([str(i+1) for i in range(n_inputs)])

    # compute cv
    X.requires_grad=True
    s = model(X) 

    # get gradients
    grad_output = torch.ones_like(s)
    grad = torch.autograd.grad(s, X, grad_outputs=grad_output)[0].detach().cpu().numpy()
    grad = np.abs(grad)

    # multiply grad_xi by std_xi
    grad = grad*std

    # normalize such that the averages sums to 1
    grad /= grad.mean(axis=0).sum()

    # get metrics
    def _compute_score(grad, metric):
        if (metric == 'mean_abs_val') | (metric == 'MEAN_ABS') | (metric == 'MAV'):
            score = grad.mean(axis=0)
        elif (metric == 'root_mean_square') | (metric == 'rms') | (metric == 'RMS'):
            score = np.sqrt((grad**2).mean(axis=0))
        else:
            raise NotImplementedError('only mean_abs_value (MAV) or root_mean_square (RMS) metrics are allowed')
        return score

    score = _compute_score(grad, metric)

    # sort features based on score
    index = score.argsort()
    feature_names = np.asarray(feature_names)[index]
    score = score[index]
    grad = grad[:,index]

    # store into results
    out = {}
    out['feature_names'] = feature_names
    out['sensitivity'] = { 'Dataset' : score }
    out['gradients'] = { 'Dataset' : grad }
    
    # per class statistics 
    if per_class:
        try: 
            labels = dataset['labels'].numpy().astype(int)
        except KeyError:
            raise KeyError('Per class analyis requested but no labels found in the given dataset.')

        unique_labels = np.unique(labels)
        for i,l in enumerate(unique_labels):
            mask = np.argwhere(labels==l)[:,0]
            grad_l = grad[mask,:]
            score_l = _compute_score(grad_l, metric)
            out['sensitivity'][f'State {l}'] =  score_l
            out['gradients'][f'State {l}'] =  grad_l

    # plot
    if plot_mode is not None:
        plot_sensitivity(out,mode=plot_mode,ax=ax)
    
    return out

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
    for per_class in [True,False,None]:
        for names in [None,['x','y'],np.asarray(['x','y'])]:
            results =  sensitivity_analysis(model,dataset,feature_names=names,per_class=per_class,plot_mode=None)
