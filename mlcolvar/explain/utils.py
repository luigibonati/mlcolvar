import matplotlib.pyplot as plt
import numpy as np

__all__ = [ "plot_features_distribution" ]

def plot_features_distribution(dataset, features, titles=None, axs=None):
    """Plot distribution of the given features.

    Parameters
    ----------
    dataset : DictDataset
        dataset 
    features : list
        list of features names
    titles : list,optional
        list titles to be displayed, by default None 
    axs : optional
        matplotlib axs, by default None

    """

    if isinstance(features,dict):
        raise TypeError('features should be a list of feature names, not a dictionary')
    
    n_feat = len(features)

    if axs is None:
        if n_feat <=5 :
            fig, axs = plt.subplots(1,n_feat,figsize=(3*n_feat+1,3))
        else:
            fig, axs = plt.subplots(n_feat, 1, figsize=(3, 3*n_feat))
        
        plt.suptitle('Features distribution')
        init_ax = True
    else:
        if n_feat != len(axs):
            raise ValueError(f'Number of features ({len(features)}) != number of axis ({len(axs)})')
        init_ax = False
        
    axs[0].set_ylabel('Distribution')
    
    if "labels" in dataset.keys:
        labels = sorted(dataset['labels'].unique().numpy())
        for l in labels:
            id_l = np.argwhere(dataset['labels'] == l)[0,:]
            data_label = dataset['data'][id_l,:]
        
            for i,feat in enumerate(features):  
                ax = axs[i]
                id = np.argwhere(dataset.feature_names == feat)[0]
                x = data_label[:,id].numpy()
                ax.hist(x,bins=50,label=f"State {int(l)}",histtype='step')
                ax.set_yticks([])
                ax.set_xlabel(feat)
                if i == 0: 
                    if titles is not None:
                        ax.legend(title=titles[i], loc='upper center', framealpha=0.8, edgecolor='white')
                    else:
                        ax.legend(loc='upper center', framealpha=0.8, edgecolor='white')
                else:
                    if titles is not None:
                        ax.legend([],[],title=titles[i],loc='upper center', framealpha=0.8, edgecolor='white')
    else:
        for i,feat in enumerate(features):  
            ax = axs[i]
            id = np.argwhere(dataset.feature_names == feat)[0]
            data = dataset['data']
            x = data[:,id].numpy()
            ax.hist(x,bins=100,)
            ax.set_yticks([])
            ax.legend([],[],title=feat,loc='upper center',frameon=False)