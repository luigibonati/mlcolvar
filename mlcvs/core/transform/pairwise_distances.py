import torch
import pytorch_lightning as pl

from typing import Union
from warnings import warn

from mlcvs.core.transform import Transform


class PairwiseDistances(Transform):
    '''
    TODO 
    '''

    MODES = ['adjacency_matrix_continuous', 
             'adjacency_matrix_discontinuous',  
             'components_distances_matrix', 
             'distances_matrix', 
             'distances_array_no_cutoff',
             'radius_graph_continuous',
             'radius_graph_discontinuous']

    # TODO maybe add a switch for reduced coordinates?
    # better to convert everything to real coordinatesa at the beginning to have safe cutoff application
    def __init__(self, 
                n_atoms: int,
                mode: str,
                reduced_coords: bool,
                cutoff : float = None, 
                PBC : bool = None, 
                cell : torch.Tensor = None):
        
        super().__init__()

        # parse args 
        self.n_atoms = n_atoms
        self.set_mode(mode)
        self.cutoff = cutoff
        self.reduced_coords = reduced_coords
        # TODO montagne di assert

        if PBC is not None:
            self.PBC = PBC
        else:
            self.PBC = False

        if PBC and cell is None:
            raise ValueError("To apply PBC, cell has to be provided!")
        if reduced_coords and cell is None:
            raise ValueError("To use reduced_coords, cell has to be provided!")
        
        # TODO assert cell size makes sense

        # Convert cell to tensor and shape it to have 3 dims
        if isinstance(cell, float) or isinstance(cell, int):
            cell = torch.Tensor([cell])
        elif isinstance(cell, list):    
            cell = torch.Tensor(cell)

        if cell.shape[0] != 1 and cell.shape[0] != 3:
            raise ValueError(f"Cell must have either shape (1) or (3). Found {cell.shape} ")

        if isinstance(cell, torch.Tensor):
            # TODO assert size makes sense if you directly pass a tensor
            if len(cell) != 3:
                cell = torch.tile(cell, (3,))
        self.cell = cell

        self.in_features = self.n_atoms * 3
        # TODO add mode dependent out_features
        if mode=='distances_array': self.out_features = int( ( n_atoms * n_atoms - n_atoms) / 2 ) #TODO think about best way to output 

    def set_mode(self, mode : str):
        '''TODO This should set everything depending on mode'''
        # TODO add modification of n_out depending on mode
        # TODO cehck input parameters compatibility with mode
        if mode not in self.MODES:
            raise ValueError(f"Mode {mode} does not exist. The available modes are: {', '.join(self.MODES)}")
        self.mode = mode
        self.in_features = self.n_atoms * 3

        if self.mode == 'components_distances_matrix':
            self.out_features = int(3*self.n_atoms*self.n_atoms)
        elif self.mode == 'distances_matrix':
            self.out_features = int(self.n_atoms*self.n_atoms)
        elif self.mode == 'adjacency_matrix_continuous':
            self.out_features = int(self.n_atoms*self.n_atoms)
        elif self.mode == 'adjacency_matrix_discontinuous':
            self.out_features = int(self.n_atoms*self.n_atoms)
        elif self.mode == 'distances_array_no_cutoff':
            self.out_features = int(self.n_atoms*(self.n_atoms -1) / 2)
        elif self.mode == 'radius_graph_continuous':
            self.out_features = None # TODO check how to pass this in case
        elif self.mode == 'radius_graph_discontinuous':
            self.out_features = None
        

    # this is common to all the modes
    def compute_all_distances_components(self, 
                                    pos : torch.Tensor) -> torch.Tensor:
        """Compute the matrices of all the atomic pairwise distances along the cell dimensions.

        Parameters
        ----------
        pos : torch.Tensor
            Positions of the atoms, they can be given with shapes:
            - Shape: (n_batch (optional), n_atoms * 3), i.e [ [x1,y1,z1, x2,y2,z2, .... xn,yn,zn] ]
            - Shape: (n_batch (optional), n_atoms, 3),  i.e [ [ [x1,y1,z1], [x2,y2,z2], .... [xn,yn,zn] ] ]

        Returns
        -------
        torch.Tensor
            Components of all the atomic pairwise distances along the cell dimensions, index map: (batch_idx, atom_i_idx, atom_j_idx, component_idx)
        """
        # avoid always calling self 
        n_atoms = self.n_atoms
        cell = self.cell

        # check if we have batch dimension in positions tensor
        if len(pos.shape)==2:
            pos = pos.unsqueeze(0)
        batch_size = pos.shape[0]
        self.batch_size = batch_size

        if pos.shape[-2] != n_atoms:
            raise ValueError(f"The given positions tensor has the wrong number of atoms. Expected {n_atoms} found {pos.shape[-2]}")
        if pos.shape[-1] != 3:
            raise ValueError(f"The given position tensor has a wrong number of spatial coordinates. Expected 3 found {pos.shape[-1]}")
        if pos.mean() <= 1:
            warn(f'Are you using coordinates scaled to cell_size without scaled_coords flag active? ')

        # reshape positions tensor to (n_batch, n_atoms, n_components=3)
        pos = torch.reshape(pos, (batch_size, n_atoms, 3)) # this preserves the order
        pos = torch.transpose(pos, 1, 2)
        pos = pos.reshape((batch_size, 3, n_atoms))

        # expand tiling the coordinates to a tensor of shape (n_batch, n_atoms, n_atoms, 3)
        pos_expanded = torch.tile(pos,(1, 1, n_atoms)).reshape(batch_size, 3, n_atoms, n_atoms)

        # compute the distances with transpose trick
        dist_components = pos_expanded - torch.transpose(pos_expanded, -2, -1)  # transpose over the atom index dimensions

        # get PBC shifts
        if self.PBC:
            shifts = torch.zeros_like(dist_components)
            # avoid loop if cell is cubic
            if cell[0]==cell[1] and cell[1]==cell[2]:
                shifts = torch.div(dist_components, cell[0]/2, rounding_mode='trunc')*cell[0]
            else: 
                # loop over dimensions of the cell
                for d in range(3):
                    shifts[:, d, :, :] = torch.div(dist_components[:, d, :, :], cell[d]/2, rounding_mode='trunc')*cell[d]

        # apply shifts
        dist_components = dist_components - shifts
        return dist_components

    def compute_all_distances(self, pos):
        dist_components = self.compute_all_distances_components(pos)
        
        # mask out diagonal --> to keep the derivatives safe
        mask_diag = ~torch.eye(self.n_atoms, dtype=bool)
        mask_diag = torch.tile(mask_diag, (self.batch_size, 1, 1))
        self.mask_diag = mask_diag

        # sum squared components and get final distance
        dist = torch.sum( torch.pow(dist_components, 2), 1 )
        dist[mask_diag] = torch.sqrt( dist[mask_diag]) 
        return dist

    def compute_adjacency_matrix(self, pos, mode):
        dist = self.compute_all_distances(pos)
        adj_matrix = self.apply_cutoff(dist, mode=mode)
        return adj_matrix
    
    # TODO rename this to compute all distances array? no cutoff no nothing
    def compute_all_distances_array(self, pos): 
        dist = self.compute_all_distances(pos)
        aux_mask = torch.ones_like(dist) - torch.eye(dist.shape[-1])
        unique = aux_mask.triu().nonzero(as_tuple=True)
        return dist[unique].reshape((dist[unique].shape[0 ], -1)) 

    def compute_radius_graph(self, pos, mode):
        dist = self.compute_all_distances(pos)
        aux_switch = torch.clone(dist)

        if mode == 'continuous': # This has stable derivatives and the output size is not fixed!
            aux_switch = self.apply_cutoff(aux_switch, mode=mode)
            dist = dist * aux_switch 
            # TODO maybe change to torch.ge because we don't have zeros anymore 
            unique = torch.nonzero(torch.ge(dist.triu(), 1e-4), as_tuple=True)
        
        elif mode == 'discontinuous': # This does not have stable derivatives and the output size is not fixed!
            aux_switch = self.apply_cutoff(aux_switch, mode=mode)
            unique = dist.triu().nonzero(as_tuple=True) 

        return dist[unique], unique

    def apply_cutoff(self, x : torch.Tensor, mode : str):
        if mode == 'continuous':
            # This has stable derivatives
            if not hasattr(self, 'switch_function_param'):
                self.switch_function_param = 0.005    
            if not hasattr(self, 'switch'):
                self.switch = 'Fermi'    
            # TODO add other switching functions?
            if self.switch == 'Fermi': 
                x[self.mask_diag] = self.switch_fermi(x[self.mask_diag], self.cutoff, self.switch_function_param)
            return x        
        elif mode == 'discontinuous':
            # This does not have stable derivatives
            mask_cutoff = torch.ge(x, self.cutoff)      
            x[mask_cutoff] = x[mask_cutoff] * 0
            mask = torch.logical_and(~mask_cutoff, self.mask_diag)
            x[mask] = x[mask] ** 0
            return x
        elif mode == 'no_cutoff':
            return x


    def switch_fermi(self, x : torch.Tensor, cutoff : Union[float, torch.Tensor], q : Union[float, torch.Tensor] = 0.001):
        """Fermi-function-like switching function: y = 1 / 1 + exp[ (x-cutoff)/q ]
        - y -> 1  for x<cutoff
        - y = 1/2 for x=cutoff
        - y -> 0  for x>cutoff

        Parameters
        ----------
        x : torch.Tensor
            Input of the switching function 
        cutoff : float or torch.Tensor
            Cutoff for the switching function.
            NB. Need to be given in real distance units, not reduced on cell dimensions
        q : float or torch.Tensor, by default 0.05
            Sharpness regulating parameter, the smaller the sharper. 
            NB. To sharp may cause overflows in the exp

        Returns
        -------
        torch.Tensor
            - y -> 1  for x<cutoff
            - y = 1/2 for x=cutoff
            - y -> 0  for x>cutoff
        """
        # TODO maybe move to utils?
        # TODO maybe pass args as dict as for loss functions
        y = torch.div( 1, ( 1 + torch.exp( torch.div((x - 1.01*cutoff), q ))))
        return y










    def fix_cutoff_problem_with_strange_cells(self):
        if self.reduced_coords and self.mode in ['adjacency_matrix_continuous', 'adjacency_matrix_discontinuous', 'radius_graph_continuous', 'radius_graph_discontinuous']:
            x = torch.einsum('bij,j->bij', x, self.cell)
        















    def forward(self, x):
        if self.mode not in self.MODES:
            raise ValueError(f"Mode {self.mode} does not exist. The available modes are: {', '.join(self.MODES)}")

        # no_cutoff
        if self.mode == 'components_distances_matrix':
            x = self.compute_all_distances_components(x)
        elif self.mode == 'distances_matrix':
            x = self.compute_all_distances(x)
        elif self.mode == 'distances_array_no_cutoff':
            x = self.compute_all_distances_array(x)

        # cutoff
        elif self.mode == 'adjacency_matrix_continuous':
            x = self.compute_adjacency_matrix(x, mode='continuous')
        elif self.mode == 'adjacency_matrix_discontinuous':
            x = self.compute_adjacency_matrix(x, mode='discontinuous')

        elif self.mode == 'radius_graph_continuous':
            x = self.compute_radius_graph(x, mode='continuous')
        elif self.mode == 'radius_graph_discontinuous':
            x = self.compute_radius_graph(x, mode='discontinuous')
        return x
        # TODO add all cases
        

def test_pairwise_distance():
    pos = torch.Tensor([ [ [0., 0., 0.],
                           [0.4, 0., 0.]], 
                         [ [0., 0., 0.],
                           [0., 0.4, 0.]] 
                        ])
    pos.requires_grad = True
    print(pos.shape)
    cell = torch.Tensor([10., 6., 10.,])
    cutoff = None
    model = PairwiseDistances(n_atoms = 2,
                              mode = 'components_distances_matrix', 
                              reduced_coords=True,
                              cutoff = cutoff,
                              PBC = True,
                              cell = cell)
    
    print()
    print('CUTOFF', model.cutoff)
    print()

    for mode in ['components_distances_matrix', 'distances_matrix', 'distances_array_no_cutoff']:
        print('MODE: ', mode)
        model.set_mode(mode)
        out = model.forward(pos)
        print('Out: \n', out)
        grad = torch.autograd.grad(torch.sum(out**2), pos)
        print('Grad: \n', grad)
        print()
    
    model.cutoff = 3.0
    print()
    print('CUTOFF', model.cutoff)
    print()
    for mode in ['adjacency_matrix_continuous', 'adjacency_matrix_discontinuous']:
        print('MODE: ', mode)
        model.set_mode(mode)
        out = model.forward(pos)
        print('Out: \n', out)
        grad = torch.autograd.grad(torch.sum(out**2), pos)
        print('Grad: \n', grad) 
        print() 

    for mode in ['radius_graph_continuous', 'radius_graph_discontinuous', 'wrong_mode']:
        print('MODE: ', mode)
        model.set_mode(mode)
        out,out_idx = model.forward(pos)
        print('Out: \n', out)
        print('Idx: \n', out_idx)
        grad = torch.autograd.grad(torch.sum(out**2), pos)
        print('Grad: \n', grad) 
        print()    

if __name__ == "__main__":
    test_pairwise_distance()







    
################# PLEASE CHECK MEEEEE ###########################à