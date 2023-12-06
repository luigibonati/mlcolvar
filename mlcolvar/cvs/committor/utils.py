import torch
import pandas as pd
import numpy as np

def compute_committor_weights(
    dataframe, 
    dataset, 
    beta : float, 
    mixing : bool = False, 
    mixing_csi : float = None
    ):
    """Utils to compute the appropriate weights for the training set for the learning of committor function.
    Compute the weights of training data based on the bias and the iteration they belong to

    Parameters
    ----------
    dataframe : pd.Dataframe
        Pandas dataframe with the training data, including descriptors, labels and bias columns.
        This should be created using the mlcolvar.utils.io.create_dataset_from_files function
    dataset : _type_
        Dataset to be updated. This should be created using the mlcolvar.utils.io.create_dataset_from_files function
    beta : float
        Inverse temperature in the right energy units
    mixing : bool
        Switch for mixing of coefficients, by default False. TODO deprecate
    mixing_csi : float
        Mixing coefficient, by default None. TODO deprecate

    Returns
    -------
    Dataframe
        Updated Pandas dataframe with the 'weights' column
    Dataset
        Updated dataset with the 'weights' entry
    """
    # Check if we have the bias column and sanitize it
    if 'bias' in dataframe.columns:
        dataframe = dataframe.fillna({'bias': 0})
    else:
        dataframe['bias'] = 0

    # compute weights
    dataframe['weights'] = np.exp(beta * dataframe['bias'])

    # group data from same iteration under the same label, keep 0 and 1 safe because unbiased!
    dataframe.loc[np.logical_and(dataframe['labels'] % 2 == 1 , dataframe['labels'] > 1), 'labels'] = dataframe.loc[np.logical_and(dataframe['labels'] % 2 == 1 , dataframe['labels'] > 1), 'labels'] - 1
    dataframe.loc[dataframe['labels'] > 1, 'labels'] = dataframe.loc[dataframe['labels'] > 1, 'labels'] / 2 + 1

    # get the reweight averages
    for i in np.unique(dataframe['labels'].values):
        # compute average of exp(beta*V) on this simualtions
        coeff = 1 / np.mean(dataframe.loc[dataframe['labels'] == i, 'weights'].values)

        # we apply weight mixing between iterations: more weight to last iterations   
        if mixing:
            max_iter = np.max(np.unique(dataframe['labels'].values))
            if i>1:
                coeff = coeff * ( mixing_csi**(max_iter - i) * (1 - mixing_csi))
            else:
                coeff = coeff * (mixing_csi**(max_iter - 1))
            
        # update the weights
        dataframe.loc[dataframe['labels'] == i, 'weights'] = coeff * dataframe.loc[dataframe['labels'] == i, 'weights']
    
    # update labels add weights to torch dataset
    dataset['labels'] = torch.Tensor(dataframe['labels'].values)
    dataset['weights'] = torch.Tensor(dataframe['weights'].values)
    
    return dataframe, dataset

def initialize_committor_masses(atoms_map : list, n_dims : int = 3):
    """Initialize the masses tensor with the right shape for committor learning

    Parameters
    ----------
    atoms_map : list[int, float]
        List of atoms in the system and the corresponing masses. Each entry should be [atom_type, atomic_mass]
    n_dims : int
        Number of dimensions of the system, by default 3.

    Returns
    -------
    atomic_masses
        Atomic masses tensor readdy to be used for committor learning.
    """
    # atomic masses of the atoms --> size N_atoms * n_dims

    # put number of atoms for each type and the corresponding atomic mass
    atoms_map = np.array(atoms_map)

    atomic_masses = []
    for i in range(len(atoms_map)):
        # each mass has to be repeated for each dimension 
        for n in range( int(atoms_map[i, 0] * n_dims) ):
            atomic_masses.append(atoms_map[i, 1])

    # make it a tensor
    atomic_masses = torch.Tensor(atomic_masses)
    # atomic_masses = atomic_masses.to(device)
    return atomic_masses
    