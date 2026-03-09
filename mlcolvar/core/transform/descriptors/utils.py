import torch
from typing import Union, List, Tuple

def sanitize_positions_shape(pos: torch.Tensor,
                             n_atoms: int):
    """Sanitize positions tensor to have [batch, atoms, dims=3] shape

    Parameters
    ----------
    pos : torch.Tensor
        Positions of the atoms, they can be given with shapes:
        - Shape: (n_batch (optional), n_atoms * 3), i.e [ [x1,y1,z1, x2,y2,z2, .... xn,yn,zn] ]
        - Shape: (n_batch (optional), n_atoms, 3),  i.e [ [ [x1,y1,z1], [x2,y2,z2], .... [xn,yn,zn] ] ]
    n_atoms : int
        Number of atoms 
    """
    # check if we have batch dimension in positions tensor
    
    if len(pos.shape)==3:
        # check that index 0: batch, 1: atom, 2: coords
        if pos.shape[1] != n_atoms:
            raise ValueError(f"The given positions tensor has the wrong format, probably the wrong number of atoms. Expected {n_atoms} found {pos.shape[1]}")
        if pos.shape[2] != 3:
            raise ValueError(f"The given position tensor has the wrong format, probably the wrong number of spatial coordinates. Expected 3 found {pos.shape[2]}")
            
    if len(pos.shape)==2:
        # check that index 0: atoms, 1: coords
        if pos.shape[0]==n_atoms and pos.shape[1] == 3:
            pos = pos.unsqueeze(0) # add batch dimension
        # check that is not 0: batch, 1: atom*coords
        elif not pos.shape[1] == int(n_atoms * 3):
            raise ValueError(f"The given positions tensor has the wrong format, found {pos.shape}, expected either {[n_atoms, 3]} or {-1, n_atoms*3}")

    if len(pos.shape)==1:
        # check that index 0: atoms*coord
        if len(pos) != n_atoms*3:
            raise ValueError(f"The given positions tensor has the wrong format. It should be at least of shape {int(n_atoms*3)}, found {pos.shape[0]}")
        # else:
        #     pos = pos.unsqueeze(0) # add batch dimension
    
    pos = torch.reshape(pos, (-1, n_atoms, 3))

    batch_size = pos.shape[0]
    return pos, batch_size

def sanitize_cell_shape(cell: Union[float, torch.Tensor, list]):
    # Convert cell to tensor and shape it to have 3 dims
    if isinstance(cell, float) or isinstance(cell, int):
        cell = torch.Tensor([cell])
    elif isinstance(cell, list):    
        cell = torch.Tensor(cell)
    elif isinstance(cell, torch.Tensor) and cell.shape == torch.Size([]):
        cell = cell.unsqueeze(0)
    
    # Extra check to ensure that cell is now a tensor and has the right shape
    if not isinstance(cell, torch.Tensor):
        raise ValueError(f"Cell should be a torch.Tensor after parsing, found {type(cell)}")

    # Single float for cubic cell, or 3 floats for orthorombic cell
    if cell.ndim == 1:
        if cell.shape[0] not in (1, 3):
            raise ValueError(f"Cell must have shape (1), (3), (BatchSize,1) or (BatchSize,3). Found {cell.shape}.")
        if cell.shape[0] == 1:
            cell = torch.tile(cell, (3,))
    # Batch of single floats for cubic cells, or batch of 3 floats for orthorombic cells
    elif cell.ndim == 2:
        if cell.shape[1] not in (1, 3):
            raise ValueError(f"Cell must have shape (1), (3), (BatchSize,1) or (BatchSize,3). Found {cell.shape}.")
        if cell.shape[1] == 1:
            cell = torch.tile(cell, (1, 3))
    else:
        raise ValueError(f"Cell must have shape (1), (3), (BatchSize,1) or (BatchSize,3). Found {cell.shape}.")
    
    return cell

def _resolve_descriptor_cell(runtime_cell: Union[float, torch.Tensor, list, None],
                            default_cell: Union[torch.Tensor, None],
                            require_cell: bool = False,
                        ) -> Union[torch.Tensor, None]:
    """Resolve descriptor cell coming from init-time default and/or runtime input.

    Rules:
    - Runtime cell and init-time default cell cannot be provided together.
    - If `require_cell=True`, one of them must be provided.
    """
    if runtime_cell is not None and default_cell is not None:
        raise ValueError(
            "`cell` was provided at initialization and cannot be passed again at runtime."
        )

    cell = default_cell if runtime_cell is None else runtime_cell
    if require_cell and cell is None:
        raise ValueError(
            "No `cell` was provided at initialization or runtime, but it is required for PBC calculations."
        )
    return cell

def resolve_cell(
    cell: Union[float, torch.Tensor, list, None],
    *,
    PBC: bool,
    scaled_coords: bool,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Resolve and validate cell information for distance-based descriptors."""
    if cell is None:
        if PBC or scaled_coords:
            raise ValueError(
                "A `cell` must be provided when `PBC=True` or `scaled_coords=True`."
            )
        return torch.ones((batch_size, 3), device=device)

    cell = sanitize_cell_shape(cell).to(device)
    if cell.ndim == 1:
        return cell
    if cell.shape[0] not in (1, batch_size):
        raise ValueError(
            f"Batch cell size mismatch: got {cell.shape[0]} cells for batch size {batch_size}."
        )
    return cell

def _apply_pbc_distances(dist_components, pbc_cell):
    # Normalize cell to batched shape [B,3] so single and batched cell inputs share the same logic.
    batch_size = dist_components.shape[0]
    if pbc_cell.ndim == 1:
        pbc_cell = pbc_cell.unsqueeze(0)
    if pbc_cell.shape[0] not in (1, batch_size):
        raise ValueError(
            f"Batch cell size mismatch: got {pbc_cell.shape[0]} cells for batch size {batch_size}."
        )

    # Mixed cubic/non-cubic batches are not supported.
    is_cubic = (pbc_cell[:, 0] == pbc_cell[:, 1]) & (pbc_cell[:, 1] == pbc_cell[:, 2])
    if not torch.all(is_cubic) and not torch.all(~is_cubic):
        raise ValueError("Mixed cubic and non-cubic cells in the same batch are not supported.")

    shifts = torch.zeros_like(dist_components)
    
    # For cubic cells we can apply the minimum image convention in one step
    if torch.all(is_cubic):
        c = pbc_cell[:, 0]
        while c.ndim < dist_components.ndim:
            c = c.unsqueeze(-1)
        shifts = torch.div(dist_components, c / 2, rounding_mode='trunc')
        shifts = torch.div(shifts + 1 * torch.sign(shifts), 2, rounding_mode='trunc') * c
    # For non-cubic cells we need to apply the minimum image convention separately for each dimension
    else:
        # Loop over cell dimensions. Works for both matrix and pairwise distance tensors.
        for d in range(3):
            c = pbc_cell[:, d]
            while c.ndim < dist_components[:, d, :].ndim:
                c = c.unsqueeze(-1)
            shifts[:, d, :] = torch.div(dist_components[:, d, :], c / 2, rounding_mode='trunc')
            shifts[:, d, :] = torch.div(shifts[:, d, :] + 1 * torch.sign(shifts[:, d, :]), 2, rounding_mode='trunc') * c / 2

    # apply shifts
    dist_components = dist_components - shifts
    return dist_components

def compute_distances_matrix(pos: torch.Tensor,
                             n_atoms: int,
                             PBC: bool,
                             cell: Union[float, list, None] = None,
                             vector: bool = False,
                             scaled_coords: bool = False,
                            ) -> torch.Tensor:
    """Compute the pairwise distances matrix from batches of atomic coordinates. 
    The matrix is symmetric, of size (n_atoms,n_atoms) and i,j-th element gives the distance between atoms i and j. 
    Optionally can return the vector distances.

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
    cell : Union[float, list]
        Dimensions of the real cell, orthorombic-like cells only, by default False
    vector : bool, optional
        Switch to return vector distances
    scaled_coords : bool, optional
        Switch for coordinates scaled on cell's vectors use

    Returns
    -------
    torch.Tensor
        Matrix of the scalar pairwise distances, index map: (batch_idx, atom_i_idx, atom_j_idx)
        Enabling `vector=True` can return the vector components of the distances, index map: (batch_idx, atom_i_idx, atom_j_idx, component_idx)
    """
    # compute distances components, keep only first element of the output tuple
    # ======================= CHECKS =======================
    pos, batch_size = sanitize_positions_shape(pos, n_atoms)
    _device = pos.device
    cell = resolve_cell(
        cell,
        PBC=PBC,
        scaled_coords=scaled_coords,
        device=_device,
        batch_size=batch_size,
    )

    # Set which cell to be used for PBC
    if scaled_coords:
        pbc_cell = torch.ones_like(cell, device=_device)
    else:
        pbc_cell = cell
    
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
        dist_components = _apply_pbc_distances(dist_components=dist_components, pbc_cell=pbc_cell)

    # if we used scaled coords we need to get back to real distances
    if scaled_coords:
        if cell.ndim == 1:
            dist_components = torch.einsum('bijk,i->bijk', dist_components, cell)
        else:
            dist_components = dist_components * cell[:, :, None, None]

    if vector: 
        return dist_components
    else:
        # mask out diagonal --> to keep the derivatives safe
        mask_diag = ~torch.eye(n_atoms, dtype=bool, device=_device)
        mask_diag = torch.tile(mask_diag, (batch_size, 1, 1))

        # sum squared components and get final distance
        dist = torch.sum( torch.pow(dist_components, 2), 1 )
        dist[mask_diag] = torch.sqrt( dist[mask_diag]) 
        return dist


def compute_distances_pairs(pos: torch.Tensor,
                             n_atoms: int,
                             PBC: bool,
                             cell: Union[float, list, None] = None,
                             slicing_pairs: List[Tuple[int, int]] = None,
                             vector: bool = False,
                             scaled_coords: bool = False,
                            ) -> torch.Tensor:
    """Compute the pairwise distances for a list of atom pairs from batches of atomic coordinates. 
    Optionally can return the vector distances.

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
    cell : Union[float, list]
        Dimensions of the real cell, orthorombic-like cells only, by default False
    slicing_pairs : list[tuple[int, int]]
        List of the indeces of the pairs for which to compute the distances
    vector : bool, optional
        Switch to return vector distances
    scaled_coords : bool, optional
        Switch for coordinates scaled on cell's vectors use

    Returns
    -------
    torch.Tensor
        Pairwise distances for the selected atom pairs
        Enabling `vector=True` can return the vector components of the distances
    """
    # ======================= CHECKS =======================
    pos, batch_size = sanitize_positions_shape(pos, n_atoms)
    _device = pos.device
    cell = resolve_cell(
        cell,
        PBC=PBC,
        scaled_coords=scaled_coords,
        device=_device,
        batch_size=batch_size,
    )

    if slicing_pairs is None:
        raise ValueError("`slicing_pairs` must be provided.")

    # Convert slicing_pairs to tensor on device if needed
    slicing_pairs = torch.as_tensor(slicing_pairs, dtype=torch.long, device=_device)

    # Set which cell to be used for PBC
    if scaled_coords:
        pbc_cell = torch.ones_like(cell, device=_device)
    else:
        pbc_cell = cell
    
    # ======================= COMPUTE =======================
    pos = torch.reshape(pos, (batch_size, n_atoms, 3)) # this preserves the order when the pos are passed as a list
    pos = torch.transpose(pos, 1, 2)
    pos = pos.reshape((batch_size, 3, n_atoms))

    # Initialize tensor to hold distances
    if vector:
        distances = torch.zeros((batch_size, len(slicing_pairs), 3), device=_device)
    else:
        distances = torch.zeros((batch_size, len(slicing_pairs)), device=_device)

    # we create two tensors for starting and ending positions
    pos_a = pos[:, :, slicing_pairs[:, 0]]
    pos_b = pos[:, :, slicing_pairs[:, 1]]

    # compute the distance components for all the pairs
    dist_components = pos_b - pos_a
    
    # get PBC shifts
    if PBC:
        dist_components = _apply_pbc_distances(dist_components=dist_components, pbc_cell=pbc_cell)

    # if we used scaled coords we need to get back to real distances
    if scaled_coords:
        if cell.ndim == 1:
            dist_components = torch.einsum('bij,i->bij', dist_components, cell)
        else:
            dist_components = dist_components * cell[:, :, None]

    if vector:
        distances = dist_components
    else:
        distances = torch.sqrt(torch.sum(dist_components ** 2, dim=1))
    
    return distances

def apply_cutoff(x: torch.Tensor,
                 cutoff: float,
                 mode: str = 'continuous',
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
    x_clone = torch.clone(x)
    if mode == 'continuous' and switching_function is None:
        raise ValueError('switching_function is required to use continuous mode! Set This can be either a user-defined and torch-based function or a method of class switching_functions/SwitchingFunctions')
    
    batch_size = x.shape[0]
    _device = x.device
    if x.shape[-1] == x.shape[-2]:
        mask_diag = ~torch.eye(x.shape[-1], dtype=bool, device=_device)
        mask_diag = torch.tile(mask_diag, (batch_size, 1, 1))
    else:
        mask_diag = torch.ones_like(x_clone, dtype=torch.bool) 

    if mode == 'continuous':
        x_clone[mask_diag] = switching_function( x_clone[mask_diag] )

    if mode == 'discontinuous':  
        mask_cutoff = torch.ge(x_clone, cutoff)      
        x_clone[mask_cutoff] = x_clone[mask_cutoff] * 0
        mask = torch.logical_and(~mask_cutoff, mask_diag)
        x_clone[mask] = x_clone[mask] ** 0
    return x_clone


def compute_adjacency_matrix(pos: torch.Tensor,
                             mode: str,
                             cutoff: float, 
                             n_atoms: int,
                             PBC: bool,
                             cell: Union[float, list, None] = None,
                             scaled_coords: bool = False,
                             switching_function = None) -> torch.Tensor:
    """Initialize an adjacency matrix object.

    Parameters
    ----------
    pos : torch.Tensor
        Positions of the atoms, they can be given with shapes:
        - Shape: (n_batch (optional), n_atoms * 3), i.e [ [x1,y1,z1, x2,y2,z2, .... xn,yn,zn] ]
        - Shape: (n_batch (optional), n_atoms, 3),  i.e [ [ [x1,y1,z1], [x2,y2,z2], .... [xn,yn,zn] ] ]
    mode : str
        Mode for cutoff application, either:
        - 'continuous': applies a switching function to the distances which can be specified with switching_function keyword, has stable derivatives
        - 'discontinuous': set at zero everything above the cutoff and one below, derivatives may be be incorrect
    cutoff : float
        Cutoff for the adjacency criterion 
    n_atoms : int
        Number of atoms in the system
    PBC : bool
        Switch for Periodic Boundary Conditions use
    cell : Union[float, list]
        Dimensions of the real cell, orthorombic-like cells only
    scaled_coords : bool
        Switch for coordinates scaled on cell's vectors use, by default False
    switching_function : _type_, optional
        Switching function to be applied for the cutoff, can be either initialized as a switching_functions/SwitchingFunctions class or a simple function, by default None

    Returns
    -------
    torch.Tensor
        Adjacency matrix of all the n_atoms according to cutoff
    """
    dist = compute_distances_matrix(pos=pos,
                                    n_atoms=n_atoms,
                                    PBC=PBC,
                                    cell=cell,
                                    scaled_coords=scaled_coords)
    adj_matrix = apply_cutoff(x=dist, 
                              cutoff=cutoff, 
                              mode=mode, 
                              switching_function = switching_function)
    return adj_matrix                          


def test_applycutoff():
    from mlcolvar.core.transform.tools.switching_functions import SwitchingFunctions
    
    n_atoms=2
    pos = torch.Tensor([ [ [0., 0., 0.],
                           [1., 1., 1.] ],
                         [ [0., 0., 0.],
                           [1., 1., 1.] ] ]
                      )
    cell = torch.Tensor([1., 2, 1.])
    cutoff = 1.8

    # TEST no scaled coords
    out = compute_distances_matrix(pos=pos, n_atoms=n_atoms, PBC=True, cell=cell, scaled_coords=False)
    switching_function=SwitchingFunctions(in_features=n_atoms**2, name='Fermi', cutoff=cutoff, options={'q':0.01})
    apply_cutoff(x=out, cutoff=cutoff, mode='continuous', switching_function=switching_function)
    
    def silly_switch(x):
        return torch.pow(x, 2)
    switching_function = silly_switch
    apply_cutoff(x=out, cutoff=cutoff, mode='continuous', switching_function=switching_function)
    apply_cutoff(x=out, cutoff=cutoff, mode='discontinuous')

    # TEST scaled coords
    pos = torch.einsum('bij,j->bij', pos, 1/cell)
    out = compute_distances_matrix(pos=pos, n_atoms=2, PBC=True, cell=cell, scaled_coords=True)
    switching_function=SwitchingFunctions(in_features=n_atoms**2, name='Fermi', cutoff=cutoff, options={'q':0.01})
    apply_cutoff(x=out, cutoff=cutoff, mode='continuous', switching_function=switching_function)
    apply_cutoff(x=out, cutoff=cutoff, mode='discontinuous')


def test_adjacency_matrix():
    from mlcolvar.core.transform.tools.switching_functions import SwitchingFunctions
    
    n_atoms=2
    pos = torch.Tensor([ [ [0., 0., 0.],
                           [1., 1., 1.] ],
                         [ [0., 0., 0.],
                           [1., 1.1, 1.] ] ]
                      )
    
    cell = torch.Tensor([1., 2., 1.])
    cutoff = 1.8
    switching_function=SwitchingFunctions(in_features=n_atoms*3, name='Fermi', cutoff=cutoff, options={'q' : 0.01})
  
    compute_adjacency_matrix(pos=pos, mode='continuous', cutoff=cutoff,  n_atoms=n_atoms, PBC=True, cell=cell, scaled_coords=False, switching_function=switching_function)
