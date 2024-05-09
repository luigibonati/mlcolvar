import torch
import numpy as np
from typing import List

__all__ = ["compute_committor_weights", "initialize_committor_masses"]

def compute_committor_weights(dataset, 
                              bias: torch.Tensor, 
                              data_groups: List[int], 
                              beta: float):
    """Utils to update a DictDataset object with the appropriate weights and labels for the training set for the learning of committor function.

    Parameters
    ----------
    dataset : 
        Labeled dataset containig data from different simulations, the labels must identify each of them. 
        For example, it can be created using `mlcolvar.utils.io.create_dataset_from_files(filenames=[file1, ..., fileN], ... , create_labels=True)`
    bias : torch.Tensor
        Bias values for the data in the dataset, usually it should be the committor-based bias
    data_groups : List[int]
        Indices specyfing the iteration each labeled data group belongs to. 
        Unbiased simulations in A and B used for the boundary conditions must have indices 0 and 1.
    beta : float
        Inverse temperature in the right energy units

    Returns
    -------
        Updated dataset with weights and updated labels
    """

    if bias.isnan().any():
        raise(ValueError('Found Nan(s) in bias tensor. Check before proceeding! If no bias was applied replace Nan with zero!'))

    # TODO sign if not from committor bias
    weights = torch.exp(beta * bias)
    new_labels = torch.zeros_like(dataset['labels'])

    data_groups = torch.Tensor(data_groups)

    # correct data labels according to iteration
    for j,index in enumerate(data_groups):
        new_labels[torch.nonzero(dataset['labels'] == j, as_tuple=True)] = index

    for i in np.unique(data_groups):
        # compute average of exp(beta*V) on this simualtions
        coeff = 1 / torch.mean(weights[torch.nonzero(new_labels == i, as_tuple=True)])
        
        # update the weights
        weights[torch.nonzero(new_labels == i, as_tuple=True)] = coeff * weights[torch.nonzero(new_labels == i, as_tuple=True)]
    
    # update dataset
    dataset['weights'] = weights
    dataset['labels'] = new_labels

    return dataset

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
    