import torch
import pytorch_lightning as pl
from mlcvs.utils.decorators import decorate_methods, allowed_hooks, apply_hooks


@decorate_methods(apply_hooks,methods=allowed_hooks)
class PairwiseDistances(pl.LightningModule):
    """
    TODO
    """

    def __init__(self, 
                n_atoms: int,
                mode: str,
                cutoff : float = None, 
                PBC : bool = False, 
                cell : torch.tensor = None):
        """
        Parameters
        ----------
        n_atoms : int
            Number of atoms in the system
        mode : str
            Type of pairwise distances output, available options are:
            - 'distances_matrix': returns a symmetric matrix where the ij element is the scalar distance between i and j
            - 'graph_edges': returns the edge source and destination indeces for each distance and its scalar value
            - 'components_only': returns the symmetric matrices where the ij elements are the scalar xyz components of the distance between i and j
            - 'contact_matrix_smooth': returns the contacts matrix where the ij element goes to 1 according to a sigmoid switch function if i and j atoms are within a cutoff, goes to 0 otherwise. This is continuous and differentiable!
            - 'contact_matrix_hard': returns the contacts matrix where the ij element is 1 if i and j atoms are within a cutoff, 0 otherwise. This is discontinuous and not differentiable!
        PBC : bool, optional
            _description_, by default False
        cell : torch.tensor, optional
            _description_, by default None

        Raises
        ------
        ValueError
            _description_
        """
        super().__init__()

        # parse args 
        self.n_atoms = n_atoms
        self.mode = mode
        self.cutoff = cutoff
        # TODO montagne di assert

        self.PBC = PBC
        if PBC and cell is None:
            raise ValueError("To apply PBC, cell has to be provided!")
        self.cell = cell

        self.n_in = self.n_atoms * 3
        # TODO add mode dependent n_out
        self.n_out = int( ( n_atoms * n_atoms - n_atoms) / 2 ) #TODO think about best way to output 

        # the idea is to have forward of this thing
        #self.compute_distances(mode)

    def set_mode(self, mode : str):
        '''TODO This should set everything depending on mode'''
        # TODO add modification of n_out depending on mode
        # TODO cehck input parameters compatibility with mode


    def compute_distances(self, 
                          pos : torch.tensor,
                          cutoff : float = None,
                          mode : str = 'distances_matrix') -> torch.tensor:
        """
        TODO TODO TODO

        Parameters
        ----------
        pos : torch.tensor
            _description_
        cutoff : float, optional
            _description_, by default None
        mode : str, optional
            Type of output, by default 'distances_matrix'
            Available options:
            - 'distances_matrix': returns a symmetric matrix where the ij element is the scalar distance between i and j
            - 'distances_array': returns an array with the lists of all the unique pairwise distances obtained as the upper triangle part of the distances matrix 
            - 'graph_edges': returns the edge source and destination indeces for each distance and its scalar value
            - 'components_only': returns the symmetric matrices where the ij elements are the scalar xyz components of the distance between i and j
            - 'contact_matrix_smooth': returns the contacts matrix where the ij element goes to 1 according to a sigmoid switch function if i and j atoms are within a cutoff, goes to 0 otherwise. This is continuous and differentiable!
            - 'contact_matrix_hard': returns the contacts matrix where the ij element is 1 if i and j atoms are within a cutoff, 0 otherwise. This is discontinuous and not differentiable!
        switch_function : bool, optional
            _description_, by default False

        Returns
        -------
        torch.tensor
            _description_
        """
        # make positions tensor of size batch x 3 x n_atoms
        self.batch_size = pos.shape[0]
        pos = torch.reshape(pos, (self.batch_size, self.n_atoms, 3))
        pos = torch.transpose(pos, 1,2)
        pos = pos.reshape((self.batch_size, 3, self.n_atoms))
        
        # expand the positions to matrix
        pos_expanded = torch.tile(pos,(1,1,self.n_atoms)).reshape(self.batch_size,3,self.n_atoms,self.n_atoms)
        
        # compute the distances with transpose trick
        dist_components = pos_expanded - torch.transpose(pos_expanded, -2, -1) # transposes on the last two dimensions
        if self.PBC:
            # apply PBC
            mask_pbc = torch.ge(torch.abs(dist_components), self.cell/2)
            dist_components[mask_pbc] = dist_components[mask_pbc] - torch.sign(dist_components[mask_pbc])*self.cell
        
        if cutoff is None and mode == 'components_only': # we can skip the rest
            return dist_components
        
         # mask out diagonal --> to keep the derivatives safe
        mask_diag = ~torch.eye(self.n_atoms, dtype=bool)
        mask_diag = torch.tile(mask_diag, (self.batch_size, 1, 1))

        # sum squared components and get final distance
        dist = torch.sum( torch.pow(dist_components, 2), 1 )
        dist[mask_diag] = torch.sqrt( dist[mask_diag]) 
        
        # apply cutoff 
        if mode == 'contact_matrix_smooth' and cutoff is not None:
            if not hasattr(self, 'switch_function_exp'):
                self.switch_function_exp = 0.05    
            if not hasattr(self, 'switch'):
                self.switch = 'Fermi'    
            if self.switch == 'Fermi':
                dist[mask_diag] = torch.div(1, (torch.exp( 1 + torch.div((dist[mask_diag] - self.cutoff), self.switch_function_exp ) )))
            
            # dist[mask_diag] = torch.div((1 - torch.pow(dist[mask_diag]/cutoff, self.switch_function_exp)),
            #                 (1 - torch.pow(dist[mask_diag]/cutoff, self.switch_function_exp*2)  + 1e-6))
        
        # hard way setting zero above cutoff
        elif cutoff is not None:
            mask_cutoff = torch.gt(dist, self.cutoff)
            dist[mask_cutoff] = dist[mask_cutoff] * 0     
            if mode == 'contact_matrix_hard': # not differentiable!
                dist[torch.logical_and(~mask_cutoff, mask_diag)] = 1

        # matrix like output modes
        if mode == 'distances_matrix' or mode == 'contact_matrix_smooth' or mode == 'contact_matrix_hard': # contact matrix like output
            return dist
        
        if mode == 'distances_array':
            if cutoff is None: 
                unique = dist.triu().nonzero(as_tuple=True)
            else:
                aux_mask = torch.ones_like(dist, device=dist.device) - torch.eye(dist.shape[-1], device=dist.device)
                unique = aux_mask.triu().nonzero(as_tuple=True)
            return dist[unique].reshape((pos.shape[0 ], -1)) 
        
        if mode == 'components_only':
            dist_components[mask_cutoff] = dist_components[mask_cutoff] * 0
            return dist_components
        
        # array like output modes 
        elif mode == 'graph_edges': # graph like output
            # this gives (scalar_distances), (batch indeces), (source_index), (destination_index)
            indeces = dist.triu().nonzero(as_tuple=True)
            return dist[indeces], indeces

    def forward(self, x : torch.tensor) -> torch.tensor:
        x = self.compute_distances(pos=x, cutoff = self.cutoff, mode=self.mode)
        return x
        

def test_pairwise_distance():
    pos = torch.tensor([ [ [0., 0., 0.],
                           [1., 0., 0.],
                           [0., 1., 0.] ],
                         [ [0., 0., 0.],
                           [1.5, 0., 0.],
                           [0., 1.5, 0.] ] ])
    cell = torch.tensor([2.,])
    cutoff = None
    model = PairwiseDistances(n_atoms = 3,
                              mode = 'distances_matrix', 
                              cutoff = cutoff,
                              PBC = True,
                              cell = cell)
    #for mode in ['distances_matrix', 'graph_edges', 'components_only', 'contact_matrix_smooth', 'contact_matrix_hard']:
    model.mode = 'distances_matrix'
    out = model.forward(pos)
    print(out)
    model.mode = 'distances_array'
    out = model.forward(pos)
    print(out)
    # model.mode = 'graph_edges'
    # out = model.forward(pos)
    # print(out)
    # model = PairwiseDistances(n_atoms = 3,
    #                           mode = 'contact_matrix_smooth', 
    #                           cutoff = cutoff,
    #                           PBC = True,
    #                           cell = cell)    
    # out = model.forward(pos)
    # print(out)

    # model = PairwiseDistances(n_atoms = 3,
    #                           mode = 'contact_matrix_hard', 
    #                           cutoff = cutoff,
    #                           PBC = True,
    #                           cell = cell)    
    # out = model.forward(pos)
    # print(out)

if __name__ == "__main__":
    test_pairwise_distance()







    
    