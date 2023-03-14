import torch
from typing import Union
from warnings import warn

__all__ = ["Statistics"]

class Statistics(object):
    """
    Calculate statistics (running mean and std.dev based on Welford's algorithm, as well as min and max).
    If used with an iterable (such as a dataloader) provides the running estimates. 
    To get the dictionary with the results use the .to_dict() method.
    """
    def __init__(self, X : torch.Tensor = None):
        self.count = 0

        self.properties = ['Mean','Std','Min','Max']

        # initialize properties and temp var M2
        for prop in self.properties:
            setattr(self,prop,None)
        setattr(self,'M2',None)
    
        self.__call__(X)

    def __call__(self,x):
        self.update(x)

    def update(self,x):
        if x is None:
            return 
        
        # get batch size
        ndim = x.ndim
        batch_size = 1
        if ndim == 0:
            x = x.reshape(1,1)
        elif ndim == 1:
            x = x.unsqueeze(1)
        
        batch_size = x.shape[0]
        nfeatures = x.shape[1]
        
        new_count = self.count + batch_size

        # Initialize
        if self.Mean is None:
            for prop in ['Mean','M2','Std']:
                setattr(self,prop, torch.zeros(nfeatures))

        # compute sample mean
        sample_mean = torch.mean(x, dim=0)
        sample_m2 = torch.sum((x - sample_mean) ** 2, dim=0)

        # update stats
        delta = sample_mean - self.Mean 
        self.Mean += delta * batch_size / new_count
        corr = batch_size * self.count / new_count
        self.M2 += sample_m2 + delta**2 * corr
        self.count = new_count
        self.Std = torch.sqrt(self.M2 / self.count)

        # compute min/max
        sample_min = torch.min(x, dim=0).values
        sample_max = torch.max(x, dim=0).values

        if self.Min is None:
            self.Min = sample_min
            self.Max = sample_max
        else:
            self.Min = torch.min( torch.stack((sample_min,self.Min)), dim=0).values
            self.Max = torch.max( torch.stack((sample_max,self.Max)), dim=0).values

    def to_dict(self) -> dict:
        return {prop: getattr(self,prop) for prop in self.properties}
    
    def __repr__(self):
        repr = "<Statistics>  " 
        for prop in self.properties:
            repr+= f"{prop}: {getattr(self,prop).numpy()} "
        return repr
  

def batch_reshape(t: torch.Tensor, size : torch.Size) -> (torch.Tensor):
    """Return value reshaped according to size. 
    In case of batch expand unsqueeze and expand along the first dimension.
    For single inputs just pass:

    Parameters
    ----------
        Mean and range 

    """
    if len(size) == 1:
        return t
    if len(size) == 2:
        batch_size = size[0]
        x_size = size[1]
        t = t.unsqueeze(0).expand(batch_size, x_size)
    else:
        raise ValueError(
            f"Input tensor must of shape (n_features) or (n_batch,n_features), not {size} (len={len(size)})."
        )
    return t


def compute_distances_components_matrices(pos : torch.Tensor,
                                     n_atoms : int,
                                     PBC : bool,
                                     real_cell : Union[float, list],
                                     scaled_coords : bool,
                                    ) -> torch.Tensor:
    """Compute the matrices of all the atomic pairwise distances along the cell dimensions from batches of atomic coordinates.
    The three matrices (xyz) are symmetric, of size (n_atoms,n_atoms) and i,j-th element gives the distance between atoms i and j along that component. 

    Parameters
    ----------
    pos : torch.Tensor
        Positions of the atoms, they can be given with shapes:
        - Shape: (n_batch (optional), n_atoms * 3), i.e [ [x1,y1,z1, x2,y2,z2, .... xn,yn,zn] ]
        - Shape: (n_batch (optional), n_atoms, 3),  i.e [ [ [x1,y1,z1], [x2,y2,z2], .... [xn,yn,zn] ] ]
    n_atoms : int
        Number of atoms 
    PBC : bool
        Switch for Periodic Boundary Conditions use
    real_cell : Union[float, list]
        Dimensions of the real cell, orthorombic-like cells only
    scaled_coords : bool
        Switch for coordinates scaled on cell's vectors use

    Returns
    -------
    torch.Tensor
        Components of all the atomic pairwise distances along the cell dimensions, index map: (batch_idx, atom_i_idx, atom_j_idx, component_idx)
    """
    # ======================= CHECKS =======================
    # check if we have batch dimension in positions tensor
    if len(pos.shape)==2:
        pos = pos.unsqueeze(0)
    batch_size = pos.shape[0]

    if pos.shape[-2] != n_atoms:
        raise ValueError(f"The given positions tensor has the wrong number of atoms. Expected {n_atoms} found {pos.shape[-2]}")
    if pos.shape[-1] != 3:
        raise ValueError(f"The given position tensor has a wrong number of spatial coordinates. Expected 3 found {pos.shape[-1]}")
     
    # Convert cell to tensor and shape it to have 3 dims
    if isinstance(real_cell, float) or isinstance(real_cell, int):
        real_cell = torch.Tensor([real_cell])
    elif isinstance(real_cell, list):    
        real_cell = torch.Tensor(real_cell)

    if real_cell.shape[0] != 1 and real_cell.shape[0] != 3:
        raise ValueError(f"Cell must have either shape (1) or (3). Found {cell.shape} ")

    if isinstance(real_cell, torch.Tensor):
        # TODO assert size makes sense if you directly pass a tensor
        if len(real_cell) != 3:
            real_cell = torch.tile(real_cell, (3,))

    # Set which cell to be used for PBC
    if scaled_coords:
        cell = torch.Tensor([1., 1., 1.])
    else:
        cell = real_cell

    # ======================= COMPUTE =======================
    pos = torch.reshape(pos, (batch_size, n_atoms, 3)) # this preserves the order when the pos are passed as a list
    pos = torch.transpose(pos, 1, 2)
    pos = pos.reshape((batch_size, 3, n_atoms))

    # expand tiling the coordinates to a tensor of shape (n_batch, 3, n_atoms, n_atoms)
    pos_expanded = torch.tile(pos,(1, 1, n_atoms)).reshape(batch_size, 3, n_atoms, n_atoms)

    # compute the distances with transpose trick
    # This works only with orthorombic cells 
    dist_components = pos_expanded - torch.transpose(pos_expanded, -2, -1)  # transpose over the atom index dimensions

    # get PBC shifts
    if PBC:
        shifts = torch.zeros_like(dist_components)
        # avoid loop if cell is cubic
        if cell[0]==cell[1] and cell[1]==cell[2]:
            shifts = torch.div(dist_components, cell[0]/2, rounding_mode='trunc')*cell[0]/2
        else: 
            # loop over dimensions of the cell
            for d in range(3):
                shifts[:, d, :, :] = torch.div(dist_components[:, d, :, :], cell[d]/2, rounding_mode='trunc')*cell[d]/2
            
        # apply shifts
        dist_components = dist_components - shifts
    return dist_components, real_cell, scaled_coords


def compute_distances_matrix(pos : torch.Tensor,
                             n_atoms : int,
                             PBC : bool,
                             real_cell : Union[float, list],
                             scaled_coords : bool,
                            ) -> torch.Tensor:
    """Compute the pairwise distances matrix from batches of atomic coordinates. 
    The matrix is symmetric, of size (n_atoms,n_atoms) and i,j-th element gives the distance between atoms i and j. 

    Parameters
    ----------
    pos : torch.Tensor
        Positions of the atoms, they can be given with shapes:
        - Shape: (n_batch (optional), n_atoms * 3), i.e [ [x1,y1,z1, x2,y2,z2, .... xn,yn,zn] ]
        - Shape: (n_batch (optional), n_atoms, 3),  i.e [ [ [x1,y1,z1], [x2,y2,z2], .... [xn,yn,zn] ] ]
    n_atoms : int
        Number of atoms
    PBC : bool
        Switch for Periodic Boundary Conditions use
    real_cell : Union[float, list]
        Dimensions of the real cell, orthorombic-like cells only
    scaled_coords : bool
        Switch for coordinates scaled on cell's vectors use

    Returns
    -------
    torch.Tensor
        Matrix of the pairwise distances along the cell dimensions, index map: (batch_idx, atom_i_idx, atom_j_idx, component_idx)
    """
    # compute distances components, keep only first element of the output tuple
    dist_components = compute_distances_components_matrices(pos=pos, n_atoms=n_atoms, PBC=PBC, real_cell=real_cell, scaled_coords=scaled_coords)[0]
    
    # all the checks on the shape are already in the components function
    batch_size = dist_components.shape[0]

    # mask out diagonal --> to keep the derivatives safe
    mask_diag = ~torch.eye(n_atoms, dtype=bool)
    mask_diag = torch.tile(mask_diag, (batch_size, 1, 1))
    
    # if we used scaled coords we need to get back to real distances
    if scaled_coords:
        dist_components = torch.einsum('bijk,i->bijk', dist_components, real_cell)

    # sum squared components and get final distance
    dist = torch.sum( torch.pow(dist_components, 2), 1 )
    dist[mask_diag] = torch.sqrt( dist[mask_diag]) 
    return dist


def apply_cutoff(x : torch.Tensor,
                 cutoff : float,
                 mode : str = 'continuous',
                 switching_function = None) -> torch.Tensor:
    """Apply a cutoff to a quantity.
    Returns 1 below the cutoff and 0 above 

    Parameters
    ----------
    x : torch.Tensor
        Quantity on which the cutoff has to be applied
    cutoff : float
        Value of the cutoff. In case of distances it must be given in the real units
    mode : str, optional
        Application mode for the cutoff, either 'continuous'or 'discontinuous', by default 'continuous'
        This can be either:
        - 'continuous': applies a switching function and gives stable derivatives accordingly
        - 'discontinuous': sets to one what is below the cutoff and to zero what is above. The derivatives may be problematic 
    switching_function : function, optional
        Switching function to be applied if in continuous mode, by default None.
        This can be either a user-defined and torch-based function or a method of class SwitchingFuncitons

    Returns
    -------
    torch.Tensor
        Cutoffed quantity
    """
    if mode == 'continuous' and switching_function is None:
        warn('switching_function is required to use continuous mode! Set This can be either a user-defined and torch-based function or a method of class switching_functions/SwitchingFunctions')
    
    batch_size = x.shape[0]
    mask_diag = ~torch.eye(x.shape[-1], dtype=bool)
    mask_diag = torch.tile(mask_diag, (batch_size, 1, 1))

    if mode == 'continuous':
        x[mask_diag] = switching_function( x[mask_diag] )

    if mode == 'discontinuous':  
        mask_cutoff = torch.ge(x, cutoff)      
        x[mask_cutoff] = x[mask_cutoff] * 0
        mask = torch.logical_and(~mask_cutoff, mask_diag)
        x[mask] = x[mask] ** 0
    return x


# ================================================================================================
# ======================================== TEST FUNCTIONS ========================================
# ================================================================================================

def test_statistics():
    # create fake data
    X = torch.arange(0,100)
    X = torch.stack([X+0.,X+100.,X-1000.],dim=1)
    y = X.square().sum(1)
    print('X',X.shape)
    print('y',y.shape)

    # compute stats
    
    stats = Statistics()
    stats(X)
    print(stats)
    stats.to_dict()

    # create dataloader
    from mlcvs.data import FastDictionaryLoader
    loader = FastDictionaryLoader({'data':X,'target':y},batch_size=20)

    # compute statistics of a single key of loader
    key = 'data'
    stats = Statistics()
    for batch in loader:
        stats.update(batch[key])
    print(stats)

    # compute stats of all keys in dataloader

    # init a statistics object for each key
    stats = {}
    for batch in loader:
        for key in loader.keys:
            #initialize
            if key not in stats:
                stats[key] = Statistics(batch[key])
            # or accumulate
            else:
                stats[key].update(batch[key])
        
    for key in loader.keys:
        print(key,stats[key])


def test_distances_and_cutoff():
    from mlcvs.core.transform.switching_functions import SwitchingFunctions
    pos = torch.Tensor([ [ [0., 0., 0.],
                           [1., 1., 1.] ],
                         [ [0., 0., 0.],
                           [1., 1., 1.] ] ]
                      )
    real_cell = torch.Tensor([1., 2, 1.])
    
    # TEST no scaled coords
    out = compute_distances_matrix(pos=pos,
                                   n_atoms=2,
                                   PBC=True,
                                   real_cell=real_cell,
                                   scaled_coords=False)
    
    cutoff = 1.8
    switching_function=SwitchingFunctions('Fermi', cutoff, options={'q':0.01})
    out2 = apply_cutoff(out, cutoff, mode='continuous', switching_function=switching_function)
    
    def silly_switch(x):
        return torch.pow(x, 2)
    switching_function = silly_switch
    out2 = apply_cutoff(out, cutoff, mode='continuous', switching_function=switching_function)
    out2 = apply_cutoff(out, cutoff, mode='discontinuous')

    # TEST scaled coords
    pos = torch.einsum('bij,j->bij', pos, 1/real_cell)
    out = compute_distances_matrix(pos=pos,
                                   n_atoms=2,
                                   PBC=True,
                                   real_cell=real_cell,
                                   scaled_coords=True)
    cutoff = 1.8
    switching_function=SwitchingFunctions('Fermi', cutoff, options={'q':0.01})
    out2 = apply_cutoff(out, cutoff, mode='continuous', switching_function=switching_function)
    out2 = apply_cutoff(out, cutoff, mode='discontinuous')


if __name__ == "__main__":
    test_distances_and_cutoff()
    test_statistics()