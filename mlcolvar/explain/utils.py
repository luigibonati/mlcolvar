import matplotlib
import matplotlib.pyplot as plt
import numpy as np

__all__ = [ "plot_features_distribution" ]

def plot_features_distribution(dataset, features, axs=None):
    """Plot distribution of the given features.

    Parameters
    ----------
    dataset : DictDataset
        dataset 
    features : list
        list of features names
    axs : optional
        matplotlib axs, by default None

    """

    if isinstance(features,dict):
        raise TypeError('features should be a list of feature names, not a dictionary')
    
    n_feat = len(features)

    if axs is None:
        fig, axs = plt.subplots(n_feat, 1, figsize=(3.5, 2*n_feat))
        plt.suptitle('Features distribution')
        init_ax = True
    else:
        if n_feat != len(axs):
            raise ValueError(f'Number of features ({len(features)}) != number of axis ({len(axs)})')
        init_ax = False
        
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
                if i == 0: 
                    ax.legend(title=feat, loc='upper center',frameon=False)
                else:
                    ax.legend([],[],title=feat,loc='upper center',frameon=False)
    else:
        for i,feat in enumerate(features):  
            ax = axs[i]
            id = np.argwhere(dataset.feature_names == feat)[0]
            data = dataset['data']
            x = data[:,id].numpy()
            ax.hist(x,bins=100,)
            ax.set_yticks([])
            ax.legend([],[],title=feat,loc='upper center',frameon=False)

    if init_ax:
        plt.tight_layout()