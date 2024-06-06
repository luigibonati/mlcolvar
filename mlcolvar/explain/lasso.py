import numpy as np
import torch

import matplotlib 
import matplotlib.pyplot as plt
import mlcolvar.utils.plot

try:
    import sklearn
except ImportError:
    print('The lasso module requires scikit-learn as additional dependency.')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import balanced_accuracy_score

__all__ = [ "lasso_classification", "plot_lasso_classification", "lasso_regression", "plot_lasso_regression" ]

class SparsityScoring:
    """Scorer function used as metric in lasso_classification. 
    The score balances the accuracy of the predictions and the number of features n:
    .. math:: \text{score} = - (1-\text{accuracy})*100 - \left|n-\text{min\_features}\right| 

    this implies that a feature is retained only if increases the accuracy by 1%.
    """

    def __init__(self, min_features=0 ):
        """Initialize scorer with (optional) number of features

        Parameters
        ----------
        min_features : int, optional
            minimum number of features, by default 0
        """
        self.min_features = min_features

    def __call__(self, estimator, X, y, **kwargs):
        y_pred = estimator.predict(X)
        # Accuracy
        #acc = accuracy_score(y, y_pred)
        acc = balanced_accuracy_score(y, y_pred)
        # Number of features
        coefficents = estimator.coef_
        num = np.count_nonzero(coefficents)

        # criterion
        loss = (1-acc)*100 + np.abs(num-self.min_features)
        return -1*loss
    
    def accuracy_from_score(self, score, num_features):
        loss = -1*score
        err = loss - np.abs(num_features - self.min_features)
        acc = 1 - err/100
        return acc

def lasso_classification(dataset,
                         min_features = 0,
                         Cs = 40,
                         scale_inputs = True,
                         feature_names = None,
                         print_info = True,
                         plot = True
):
    """Perform sparse classification via LASSO on a given DictDataset (requires keys: "data" and "labels").
    The (inverse) regularization strength C is automatically chosen based on cross-validation on a set of values (Cs),
    see sklearn.linear_model.LogisticRegressionCV. The scoring function used is `SparsityScoring`, balancing the accuracy and the number of features.

    In the two-classes case a single classifier is built, otherwise a one-vs-rest classifier is constructed, composed by N different estimators are trained to classify each state from the others.
    
    Parameters
    ----------
    dataset : DictDataset
        dataset with 'data' and 'labels'
    min_features : int, optional
        minimum number of features, by default 0
    Cs : int or array-like, optional
        Each of the values in Cs describes the inverse of regularization strength. If Cs is as an int, then a grid of Cs values are chosen in a logarithmic scale between 1e-4 and 1e4. Like in support vector machines, smaller values specify stronger regularization., by default 40
    scale_inputs : bool, optional
        whether to standardize inputs based on mean and std.dev., by default True
    feature_names : list, optional
        names of the input features, if not given they are taken from `dataset.feature_names`, by default None
    print_info : bool, optional
        whether to print results, by default True
    plot : bool, optional
        whether to plot results, by default True

    See also
    --------
    mlcolvar.explain.lasso.SparsityScoring
        Scoring function used in LASSO classification

    Returns
    -------
    classifier: 
        optimized estimator
    feats: 
        dictionary with the names of the non-zero features, per label
    coeffs: 
        dictionary with the coefficients of the non-zero features, per label
    """

    # Convert dataset to numpy
    with torch.no_grad():
        raw_descriptors = dataset['data'].numpy()
        labels = dataset['labels'].numpy().astype(int)
    if feature_names is None:
        if dataset.feature_names is None:
            raise ValueError('Feature names not found (either in the dataset or as argument to the function).')
        feature_names = dataset.feature_names
    
    # Scaling inputs
    if scale_inputs:
        scaler = StandardScaler(with_mean=True, with_std=True)
        descriptors = scaler.fit_transform(raw_descriptors)
    else:
        descriptors = raw_descriptors

    # Define cross-validation for LASSO, using
    #   a custom scoring function based on accuracy and number of features
    scorer = SparsityScoring(min_features=min_features) 

    _classifier = LogisticRegressionCV(Cs=Cs, 
                                    solver='liblinear', 
                                    multi_class='ovr', 
                                    fit_intercept=False, 
                                    penalty='l1', 
                                    max_iter = 500,
                                    scoring=scorer)

    # Fit classifier
    feature_selector = SelectFromModel(_classifier)
    feature_selector.fit(descriptors, labels)

    classifier = feature_selector.estimator_

    # Get selected features and coefficients 
    feats = {}
    coeffs = {}

    for i,key in enumerate(classifier.coefs_paths_.keys()):

        index = np.abs(classifier.coef_).argsort()[i][::-1]

        sorted_feature_names = feature_names[index]
        sorted_coeffs = classifier.coef_[i,index]

        idx = np.argwhere ( np.abs(sorted_coeffs)>1e-5 )[:,0]
        selected_feature_names = sorted_feature_names[idx]
        selected_coeffs = sorted_coeffs[idx] 
        feats[key] = selected_feature_names
        coeffs[key] = selected_coeffs

        # display summary
        if print_info:
            #score = classifier.score(descriptors,labels)
            C_idx = np.argwhere(np.abs(classifier.Cs_ - classifier.C_[i]) < 1e-8)[0,0]
            score = classifier.scores_[key].mean(axis=0)[C_idx]
            accuracy = classifier.scoring.accuracy_from_score(score, len(selected_coeffs))

            print(f'======= LASSO results ({key}) ========')
            print(f'- Regularization : {classifier.C_[i]:.8f}')
            print(f'- Score          : {score:.2f}')
            print(f'- Accuracy       : {accuracy*100:.2f}%')
            print(f'- # features     : {len(selected_coeffs)}\n')
            print(f'Features: ')
            for j,(f,c) in enumerate(zip(selected_feature_names, selected_coeffs)):
                print(f'({i+1}) {f:13s}: {c:.6f}')
            print('==================================\n')

    # plot results
    if plot:
        plot_lasso_classification(classifier, feats, coeffs)

    return classifier, feats, coeffs

def plot_lasso_classification(classifier, feats = None, coeffs = None, draw_labels='auto', axs = None):
    """Plot results of the LASSO classification."""
    
    # check that the tested regularization values are more than 1 
    if len(classifier.Cs_) == 1:
        print('Plotting is not available, as the regressor has been optimized with a single regularization value.')
        return 
    
    # get number of classifiers (1 if n_states = 2, otherwise equal to n_states)
    n_models = len(classifier.C_)

    # define figure
    if axs is None:
        init_axs = True
        _, axs = plt.subplots(3, n_models, figsize=(6*n_models, 9), sharex=True)
        plt.suptitle('LASSO CLASSIFICATION')
    else:
        init_axs = False

    for i,key in enumerate(classifier.scores_.keys()):

        # (1) COEFFICIENTS PATH
        ax = axs[0] if n_models == 1 else axs[0][i]
        c = 'black'
        ax.plot(classifier.Cs_, np.mean(classifier.coefs_paths_[key], axis=0),color=c,alpha=0.6)

        ax.set_xscale('log')
        ax.set_title(f'Coefficients path ({key})')
        ax.set_xmargin(0)
        ax.set_ylabel('Coefficients',color=c)
        values = np.asarray( [np.mean(c, axis=0) for c in classifier.coefs_paths_.values() ]) 
        ax.set_ylim(values.min(),values.max())

        if draw_labels == 'auto':
            draw_labels = False
            if feats is not None and coeffs is not None:
                draw_labels = True if len(feats[key])<= 3 else False

        if draw_labels:
            if feats is None or coeffs is None:
                raise ValueError('To draw the names of the features one need to pass both the selected features (`feats`) an coefficients (`coeffs`).')
                
            for name, coef in zip(feats[key],coeffs[key]):
                ax.text(classifier.C_[i], coef, name, 
                        fontsize='small', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.6))

        # (2) ACCURACY AND SCORE
        ax = axs[1] if n_models == 1 else axs[1][i]
        c = 'fessa6'
        score_path = classifier.scores_[key].mean(axis=0)

        ax.plot(classifier.Cs_, score_path, color=c,alpha=0.9)
        ax.set_title('Score')
        if i == 0:
            ax.set_ylabel('Score',color=c)
        ax.set_ylim(-50,5)

        ax2 = ax.twinx()
        c = 'fessa5'
        num_features_path = np.mean(np.count_nonzero(classifier.coefs_paths_[key], axis = -1), axis= 0)

        ax2.plot(classifier.Cs_,  classifier.scoring.accuracy_from_score(score_path, num_features_path), color=c, linestyle='-.',alpha=0.9)
        if i == n_models-1:
            ax2.set_ylabel('Accuracy',color=c)
        ax2.set_ylim(0.5,1.05)

        # (3) NUMBER OF FEATURES 
        ax = axs[2] if n_models == 1 else axs[2][i]
        c = 'fessa0'
        ax.plot(classifier.Cs_, num_features_path, color=c)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        ax.set_title('Number of features')
        ax.set_xlabel('Regularization strength'+r'$^{-1}$')
        ax.set_ylabel('# features',color=c)
        ax.set_ylim(0,None)

        # selected regularization value
        axs_i = axs if n_models == 1 else axs[:,i]
        for ax in axs_i:
            ax.axvline(classifier.C_[i],color='gray',linestyle='dotted')
            ax.set_xmargin(0)

    if init_axs:
        matplotlib.pyplot.tight_layout()


def lasso_regression(dataset,
                        alphas = None,
                        scale_inputs = True,
                        print_info = True,
                        plot = True,
):
    """Perform sparse regression via LASSO on a given DictDataset (requires keys: "data" and "target").
    The regularization strength alpha is automatically chosen based on cross-validation on a set of values (alphas),
    see sklearn.linear_model.LassoCV. The scoring function used is the MSE loss + alpha * L1 regularization.

    Parameters
    ----------
    dataset : DictDataset
        dataset with 'data' and 'target'
    alphas : int or array-like, optional
        List of alphas where to compute the models. If None alphas are set automatically.
    scale_inputs : bool, optional
        whether to standardize inputs based on mean and std.dev., by default True
    feature_names : list, optional
        names of the input features, if not given they are taken from `dataset.feature_names`, by default None
    print_info : bool, optional
        whether to print results, by default True
    plot : bool, optional
        whether to plot results, by default True

    Returns
    -------
    regressor: 
        optimized estimator
    feats: 
        names of the non-zero features
    coeffs: 
        coefficients of the non-zero features
    """
    # Convert dataset to numpy
    with torch.no_grad():
        raw_descriptors = dataset['data'].numpy()
        target = dataset['target'].numpy()
        feature_names = dataset.feature_names

    # Scaling inputs
    if scale_inputs:
        scaler = StandardScaler(with_mean=True, with_std=True)
        descriptors = scaler.fit_transform(raw_descriptors)
    else:
        descriptors = raw_descriptors

    # Define Cross-validation & fit
    _regressor = LassoCV(alphas=alphas)

    feature_selector = SelectFromModel(_regressor)
    feature_selector.fit(descriptors, np.squeeze(target))

    regressor = feature_selector.estimator_

    # Save coefficients path
    _, coefs_paths, dual_gaps = regressor.path(descriptors, target, alphas=regressor.alphas_)

    regressor.coefs_paths_ = coefs_paths

    # Get selected features and coefficients 
    index = np.abs(regressor.coef_).argsort()[::-1]

    sorted_feature_names = feature_names[index]
    sorted_coeffs = regressor.coef_[index]

    idx = np.argwhere ( np.abs(sorted_coeffs)>1e-5 )[:,0]
    selected_feature_names = sorted_feature_names[idx]
    selected_coeffs = sorted_coeffs[idx] 

    # display summary
    if print_info:
        score = regressor.score(descriptors,target)

        print(f'========= LASSO results ==========')
        print(f'- Regularization : {regressor.alpha_:.8f}')
        print(f'- Score          : {score:.8f}')
        print(f'- # features     : {len(selected_coeffs)}')
        print('\n======= Relevant features =======')
        for i,(f,c) in enumerate(zip(selected_feature_names, selected_coeffs)):
            print(f'({i+1}) {f:13s}: {c:.6f}')
        print('=================================')

    # plot results
    if plot:
        _ = plot_lasso_regression(regressor, selected_feature_names, selected_coeffs)

    return regressor, selected_feature_names, selected_coeffs

def plot_lasso_regression(regressor, feats = None, coeffs = None, draw_labels='auto', axs = None):
    """Plot the results of the LASSO regression."""

    # check that the tested regularization values are more than 1 
    if len(regressor.alphas_) == 1:
        print('Plotting is not available, as the regressor has been optimized with a single regularization value.')
        return 

    # define figure
    if axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
        plt.suptitle('LASSO REGRESSION')
        init_axs = True
    else:
        init_axs = False

    # (1) COEFFICIENTS PATH
    ax = axs[0]
    c = 'black'
    alphas = regressor.alphas_
    coefs_paths = regressor.coefs_paths_.T
    axs[0].plot(alphas, coefs_paths,color=c,alpha=0.6)

    ax.set_xscale('log')
    ax.set_title('Coefficients path')
    ax.set_xmargin(0)
    ax.set_ylabel('Coefficients',color=c)

    if draw_labels == 'auto':
        draw_labels = False
        if feats is not None and coeffs is not None:
            draw_labels = True if len(feats)<= 3 else False

    if draw_labels:
        if feats is None or coeffs is None:
            raise ValueError('To draw the labels one need to pass both the selected features (`feats`) an coefficients (`coeffs`).')
        for name, coef in zip(feats,coeffs):
            ax.text(regressor.alpha_, coef, name, 
                    fontsize='small', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.6))

    # (2) ACCURACY AND SCORE
    ax = axs[1]
    c = 'fessa6'
    mse_path = regressor.mse_path_.mean(axis=1)

    ax.plot(alphas, mse_path, color=c,alpha=0.9)
    ax.set_title('Score')
    ax.set_ylabel('MSE',color=c)

    # (3) NUMBER OF FEATURES 

    ax = axs[2]
    c = 'fessa0'
    num_features_path = np.count_nonzero(coefs_paths, axis = 1)
    
    ax.plot(alphas, num_features_path, color=c)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    ax.set_title('Number of features')
    ax.set_xlabel('Regularization strength')
    ax.set_ylabel('# features',color=c)

    # selected regularization value
    for ax in axs:
        ax.axvline(regressor.alpha_,color='gray',linestyle='dotted')

    if init_axs:
        matplotlib.pyplot.tight_layout()



