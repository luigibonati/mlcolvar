#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Committor function Loss Function and Utils.
"""

__all__ = ["CommittorLoss", "committor_loss", "SmartDerivatives", "compute_descriptors_derivatives"]

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import torch
from torch_scatter import scatter, scatter_sum
from typing import Union
import torch_geometric

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class CommittorLoss(torch.nn.Module):
    """Compute a loss function based on Kolmogorov's variational principle for the determination of the committor function"""

    def __init__(self,
                atomic_masses: torch.Tensor,
                alpha: float,
                gamma: float = 10000.0,
                delta_f: float = 0.0,
                separate_boundary_dataset : bool = True,
                descriptors_derivatives : torch.nn.Module = None,
                log_var: bool = True
                 ):
        """Compute Kolmogorov's variational principle loss and impose boundary conditions on the metastable states

        Parameters
        ----------
        atomic_masses : torch.Tensor
            Atomic masses of the atoms in the system
        alpha : float
            Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        gamma : float, optional
            Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound), by default 10000
        delta_f : float, optional
            Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
            State B is supposed to be higher in energy.
        separate_boundary_dataset : bool, optional
            Switch to exculde boundary condition labeled data from the variational loss, by default True
        descriptors_derivatives : torch.nn.Module, optional
            `SmartDerivatives` object to save memory and time when using descriptors.
            See also mlcolvar.core.loss.committor_loss.SmartDerivatives

        """
        super().__init__()
        self.register_buffer("atomic_masses", atomic_masses)
        self.alpha = alpha
        self.gamma = gamma
        self.delta_f = delta_f
        self.descriptors_derivatives = descriptors_derivatives
        self.separate_boundary_dataset = separate_boundary_dataset
        self.log_var = log_var

    def forward(self, 
                x: Union[torch.Tensor, torch_geometric.data.Batch], 
                q: torch.Tensor, 
                labels: torch.Tensor, 
                w: torch.Tensor, 
                create_graph: bool = True
    ) -> torch.Tensor:
        return committor_loss(x=x,
                                q=q,
                                labels=labels,
                                w=w,
                                atomic_masses=self.atomic_masses,
                                alpha=self.alpha,
                                gamma=self.gamma,
                                delta_f=self.delta_f,
                                create_graph=create_graph,
                                separate_boundary_dataset=self.separate_boundary_dataset,
                                descriptors_derivatives=self.descriptors_derivatives,
                                log_var = self.log_var
                            )


def committor_loss(x: torch.Tensor, 
                  q: torch.Tensor, 
                  labels: torch.Tensor, 
                  w: torch.Tensor,
                  atomic_masses: torch.Tensor,
                  alpha: float,
                  gamma: float = 10000,
                  delta_f: float = 0,
                  create_graph: bool = True,
                  separate_boundary_dataset: bool = True,
                  descriptors_derivatives: torch.nn.Module = None,
                  log_var: bool = True
                  ):
    """Compute variational loss for committor optimization with boundary conditions

    Parameters
    ----------
    x : torch.Tensor
        Input of the NN
    q : torch.Tensor
        Committor quess q(x), it is the output of NN
    labels : torch.Tensor
        Labels for states, A and B states for boundary conditions
    w : torch.Tensor
        Reweighing factors to Boltzmann distribution. This should depend on the simulation in which the data were collected.
    atomic_masses : torch.Tensor
        List of masses of all the atoms we are using, for each atom we need to repeat three times for x,y,z.
        Can be created using `committor.utils.initialize_committor_masses`
    alpha : float
        Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
    gamma : float
        Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound) 
        By default 10000
    delta_f : float
        Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
    create_graph : bool
        Make loss backwardable, deactivate for validation to save memory, default True
    cell : float
        CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, default None 
    separate_boundary_dataset : bool, optional
            Switch to exculde boundary condition labeled data from the variational loss, by default True
    descriptors_derivatives : torch.nn.Module, optional
        `SmartDerivatives` object to save memory and time when using descriptors.
        See also mlcolvar.core.loss.committor_loss.SmartDerivatives

    Returns
    -------
    loss : torch.Tensor
        Loss value.
    gamma*loss_var : torch.Tensor
        The variational loss term
    gamma*alpha*loss_A : torch.Tensor
        The boundary loss term on basin A
    gamma*alpha*loss_B : torch.Tensor
        The boundary loss term on basin B
    """ 
    # ============================== SETUP ==============================
    # check if input is graph
    _is_graph_data = False
    if isinstance(x, torch_geometric.data.batch.Batch):
        batch = torch.clone(x['batch'])
        node_types = torch.where(x['node_attrs'])[1]
        x = x['positions']
        _is_graph_data = True
    

    # inherit right device
    device = x.device 
    dtype = x.dtype

    # Create masks to access different states data
    mask_A = torch.nonzero(labels.squeeze() == 0, as_tuple=True) 
    mask_B = torch.nonzero(labels.squeeze() == 1, as_tuple=True)
    if separate_boundary_dataset:
        if _is_graph_data: 
            # this needs to be on the batch index, not only the labels
            mask_var = torch.nonzero(labels.squeeze() > 1)
            aux = torch.where(mask_var)[0]
            mask_var_batches = torch.isin(batch, aux)
            mask_var_batches = (batch[mask_var_batches])
        else:
            mask_var = torch.nonzero(labels.squeeze() > 1, as_tuple=True)
            mask_var_batches = mask_var 
    else: 
            mask_var = torch.ones(len(x), dtype=torch.bool)
            mask_var_batches = mask_var

    # setup atomic masses
    atomic_masses = atomic_masses.to(device)

    # mass should have size [1, n_atoms*spatial_dims]
    if _is_graph_data:
        atomic_masses = atomic_masses[node_types[mask_var_batches]].unsqueeze(-1)
    else:
        atomic_masses = atomic_masses.unsqueeze(0)

    # Update weights for bc confs using the information on the delta_f
    delta_f = torch.Tensor([delta_f]) #.to(device)
    # B higher in energy --> A-B < 0
    if delta_f < 0: 
        w[mask_B] = w[mask_B] * torch.exp(delta_f.to(device))
    # A higher in energy --> A-B > 0
    elif delta_f > 0:
        w[mask_A] = w[mask_A] * torch.exp(-delta_f.to(device)) 

    # weights should have size [n_batch, 1]
    w = w.unsqueeze(-1)
    # ==============================  LOSS ==============================
    # Each loss contribution is scaled by the number of samples

    # 1. VARIATIONAL LOSS
    # Compute gradients of q(x) wrt x
    grad_outputs = torch.ones_like(q[mask_var])
    grad = torch.autograd.grad(q[mask_var], x, grad_outputs=grad_outputs, retain_graph=True, create_graph=create_graph)[0]
    grad = grad[mask_var_batches]
    if descriptors_derivatives is not None:
        # we use the precomputed derivatives from descriptors to pos
        grad_square = descriptors_derivatives(grad)
    else:
        grad_square = torch.pow(grad, 2)

    grad_square = torch.sum((grad_square * (1/atomic_masses)), axis=1, keepdim=True)    

    if _is_graph_data:
        # we need to sum on the right batch first
        grad_square = scatter_sum(grad_square, mask_var_batches, dim=0)

    grad_square = grad_square * w[mask_var]
    # variational contribution to loss: we sum over the batch
    loss_var = torch.mean(grad_square)
    if False:
        loss_var = loss_var.log()


    # 2. BOUNDARY LOSS
    loss_A = torch.mean( torch.pow(q[mask_A], 2))
    loss_B = torch.mean( torch.pow( (q[mask_B] - 1) , 2))


    # 3. TOTAL LOSS
    loss = gamma*( loss_var + alpha*(loss_A + loss_B) )
    
    # TODO maybe there is no need to detach them for logging
    return loss, gamma*loss_var.detach(), alpha*gamma*loss_A.detach(), alpha*gamma*loss_B.detach()

class SmartDerivatives(torch.nn.Module):
    """
    Utils to compute efficently (time and memory wise) the derivatives of q wrt some input descriptors.
    Rather than computing explicitly the derivatives wrt the positions, we compute those wrt the descriptors (right input)
    and multiply them by the matrix of the derivatives of the descriptors wrt the positions (left input).
    """
    def __init__(self,
                 der_desc_wrt_pos: torch.Tensor,
                 n_atoms: int,
                 setup_device  : str = 'cpu'
                 ):
        """Initialize the fixed matrices for smart derivatives, i.e. matrix of derivatives of descriptors wrt positions.
        The derivatives wrt positions are recovered by multiplying the derivatives of q wrt the descriptors (right input, computed at each epoch)
        by the non-zero elements of the derivatives of the descriptors wrt the positions (left input, compute once at the beginning on the whole dataset).
        The multiplication are done using scatte functions and keepoing track of the indeces of the batches, descriptors, atoms and dimensions.

        NB. It should be used with only training set and single batch with shuffle and random_split disabled.

        Parameters
        ----------
        der_desc_wrt_pos : torch.Tensor
            Tensor containing the derivatives of the descriptors wrt the atomic positions
        n_atoms : int
            Number of atoms in the systems, all the atoms should be used in at least one of the descriptors
        setup_device : str
            Device on which to perform the expensive calculations. Either 'cpu' or 'cuda', by default 'cpu'
        """
        super().__init__()
        self.batch_size = len(der_desc_wrt_pos)
        self.n_atoms = n_atoms

        # setup the fixed part of the computation, i.e. left input and indeces for the scatter
        self.left, self.mat_ind, self.scatter_indeces = self._setup_left(der_desc_wrt_pos, setup_device=setup_device)
        
    def _setup_left(self, left_input : torch.Tensor, setup_device : str = 'cpu'):
        """Setup the fixed part of the computation, i.e. left input"""
        # all the setup should be done on the CPU by defualt
        left_input = left_input.to(torch.device(setup_device))

        # the indeces in mat_ind are: batch, atom, descriptor and dimension
        left, mat_ind = self._create_nonzero_left(left_input)
        
        # it is possible that some atoms are not involved in anything
        n_effective_atoms = len(torch.unique(mat_ind[1]))
        if n_effective_atoms < self.n_atoms:
            raise ValueError(f"Some of the input atoms are useless LOL. The not used atom IDs are : {[i for i in range(self.n_atoms) if i not in torch.unique(mat_ind[1]).numpy()]}  ")
        
        scatter_indeces = self._get_scatter_indices(batch_ind = mat_ind[0], atom_ind=mat_ind[1], dim_ind=mat_ind[3])
        return left, mat_ind, scatter_indeces
    
    def _create_nonzero_left(self, x):
        """Find the indeces of the non-zero elements of the left input
        """
        # find indeces of nonzero entries of d_dist_d_x
        mat_ind = x.nonzero(as_tuple=True)
                
        # flatten matrix --> big nonzero vector
        x = x.ravel()
        # find indeces of nonzero entries of the flattened matrix
        vec_ind = x.nonzero(as_tuple=True)
        
        # create vector with the nonzero entries only
        x_vec = x[vec_ind[0].long()]
        
        # del(vec_ind)
        return x_vec, mat_ind
    
    def _get_scatter_indices(self, batch_ind, atom_ind, dim_ind):
        """Compute the general indices to map the long vector of nonzero derivatives to the right atom, dimension and descriptor also in the case of non homogenoeus input.
        We need to gather the derivatives with respect to the same atom coming from different descriptors to obtain the total gradient.
        """
        # ====================================== INITIAL NOTE  ======================================
        # in the comment there's the example of the distances in a 3 atoms system with 4 batches
        # i.e. 3desc*3*atom*3dim*2pairs*4batch = 72 values needs to be mappped to 3atoms*3dims*4batch = 36

        # Ref_idx:  tensor([ 0,  1,  2,  0,  1,  2,  3,  4,  5,  3,  4,  5,  6,  7,  8,  6,  7,  8,
        #          9, 10, 11,  9, 10, 11, 12, 13, 14, 12, 13, 14, 15, 16, 17, 15, 16, 17,
        #         18, 19, 20, 18, 19, 20, 21, 22, 23, 21, 22, 23, 24, 25, 26, 24, 25, 26,
        #         27, 28, 29, 27, 28, 29, 30, 31, 32, 30, 31, 32, 33, 34, 35, 33, 34, 35])
        # ==========================================================================================

        # these would be the indeces in the case of uniform batches and number of atom/descriptor dependence
        # it just repeats the atom index in a cycle 
        # e.g. [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8, 0, 1, 2, 0, 1, 2,
        #       3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5,
        #       6, 7, 8, 6, 7, 8, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8]
        not_shifted_indeces = atom_ind*3 + dim_ind

        # get the number of elements in each batch 
        # e.g. [17, 18, 18, 18] 
        batch_elements = scatter(torch.ones_like(batch_ind), batch_ind, reduce='sum')
        batch_elements[0] -= 1 # to make the later indexing consistent

        # compute the pointer idxs to the beginning of each batch by summing the number of elements in each batch
        # e.g. [ 0., 17., 35., 53.] NB. These are indeces!
        batch_pointers = torch.Tensor([batch_elements[:i].sum() for i in range(len(batch_elements))])
        del(batch_elements)

        # number of entries in the scattered vector before each batch
        # e.g. [ 0.,  9., 18., 27.]
        markers = not_shifted_indeces[batch_pointers.long()] # highest not_shifted index for each batch
        del(not_shifted_indeces)
        del(batch_pointers)
        cumulative_markers = torch.Tensor([markers[:i+1].sum() for i in range(len(markers))]).to(batch_ind.device) # stupid sum of indeces
        del(markers)
        cumulative_markers += torch.unique(batch_ind) # markers correction by the number of batches

        # get the index shift in the scattered vector based on the batch
        # e.g. [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9,  9,  9,  9,  9,  9,
        #        9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
        #       18, 18, 18, 18, 18, 18, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27 ]
        batch_shift = torch.gather(cumulative_markers, 0, batch_ind)
        del(cumulative_markers)
        
        # finally compute the scatter indeces by including also their shift due to the batch
        shifted_indeces = atom_ind*3 + dim_ind + batch_shift

        return shifted_indeces

    def forward(self, x : torch.Tensor):
        # ensure device consistency
        left = self.left.to(x.device)

        # get the vector with the derivatives of q wrt the descriptors
        right = self._create_right(x=x, batch_ind=self.mat_ind[0], des_ind=self.mat_ind[2])

        # do element-wise product
        src = left * right
        
        # compute square modulus
        out = self._compute_square_modulus(x=src, indeces=self.scatter_indeces, n_atoms=self.n_atoms, batch_size=self.batch_size)

        return out
        
    def _create_right(self, x : torch.Tensor, batch_ind : torch.Tensor, des_ind : torch.Tensor):
        # keep only the non zero elements of right input
        desc_vec = x[batch_ind, des_ind]
        return desc_vec

    def _compute_square_modulus(self, x : torch.Tensor, indeces : torch.Tensor, n_atoms : int, batch_size : torch.Tensor):
        indeces = indeces.long().to(x.device)
        
        # this sums the elements of x according to the indeces, this way we get the contributions of different descriptors to the same atom
        out = scatter(x, indeces.long())
        # now make the square
        out = out.pow(2)
        # reshape, this needs to have the correct number of atoms as we need to mulply it by the mass vector later
        out = out.reshape((batch_size, n_atoms*3))
        return out


from mlcolvar.core.transform.descriptors.utils import sanitize_positions_shape
def compute_descriptors_derivatives(dataset, descriptor_function, n_atoms, separate_boundary_dataset = True):
    """Compute the derivatives of a set of descriptors wrt input positions in a dataset for committor optimization

    Parameters
    ----------
    dataset :
        DictDataset with the positions under the 'data' key
    descriptor_function : torch.nn.Module
        Transform module for the computation of the descriptors
    n_atoms : int
        Number of atoms in the system
    separate_boundary_dataset : bool, optional
            Switch to exculde boundary condition labeled data from the variational loss, by default True

    Returns
    -------
    desc : torch.Tensor
        Computed descriptors
    d_desc_d_pos : torch.Tensor
        Derivatives of desc wrt to pos
    """
    pos = dataset['data']
    labels = dataset['labels']
    pos = sanitize_positions_shape(pos=pos, n_atoms=n_atoms)[0]
    pos.requires_grad = True

    desc = descriptor_function(pos)
    if separate_boundary_dataset:
        mask_var = torch.nonzero(labels.squeeze() > 1, as_tuple=True)[0]
        der_desc = desc[mask_var]
        if len(der_desc)==0:
            raise(ValueError('No points left after separating boundary and variational datasets. \n If you are using only unbiased data set separate_boundary_dataset=False here and in Committor or don\'t use SmartDerivatives!!'))
    else:
        der_desc = desc

    # compute derivatives of descriptors wrt positions, loop over the number of decriptors
    aux = []
    for i in range(len(der_desc[0])):
        aux_der = torch.autograd.grad(der_desc[:,i], pos, grad_outputs=torch.ones_like(der_desc[:,i]), retain_graph=True )[0]
        if separate_boundary_dataset:
            aux_der = aux_der[mask_var]
        aux.append(aux_der)

    d_desc_d_pos = torch.stack(aux, axis=2)
    return pos, desc, d_desc_d_pos.squeeze(-1)


def test_smart_derivatives():
    from mlcolvar.core.transform import PairwiseDistances
    from mlcolvar.core.nn import FeedForward
    from mlcolvar.data import DictDataset

    # compute some descriptors from positions --> distances
    n_atoms = 10
    pos = torch.Tensor([[ 1.4970,  1.3861, -0.0273, -1.4933,  1.5070, -0.1133, -1.4473, -1.4193,
                        -0.0553,  1.4940,  1.4990, -0.2403,  1.4780, -1.4173, -0.3363, -1.4243,
                        -1.4093, -0.4293,  1.3530, -1.4313, -0.4183,  1.3060,  1.4750, -0.4333,
                        1.2970, -1.3233, -0.4643,  1.1670, -1.3253, -0.5354]])
    pos = pos.repeat(4, 1)
    labels = torch.arange(0, 4)

    dataset = DictDataset({'data' : pos, 'labels' : labels})

    cell = torch.Tensor([3.0233])
    ref_distances = torch.Tensor([[0.1521, 0.2335, 0.2412, 0.3798, 0.4733, 0.4649, 0.4575, 0.5741, 0.6815,
                                0.1220, 0.1323, 0.2495, 0.3407, 0.3627, 0.3919, 0.4634, 0.5885, 0.2280,
                                0.2976, 0.3748, 0.4262, 0.4821, 0.5043, 0.6376, 0.1447, 0.2449, 0.2454,
                                0.2705, 0.3597, 0.4833, 0.1528, 0.1502, 0.2370, 0.2408, 0.3805, 0.2472,
                                0.3243, 0.3159, 0.4527, 0.1270, 0.1301, 0.2440, 0.2273, 0.2819, 0.1482]])
    ref_distances = ref_distances.repeat(4, 1)
  
    ComputeDescriptors = PairwiseDistances(n_atoms=n_atoms,
                              PBC=True,
                              cell=cell,
                              scaled_coords=False)
    
    for separate_boundary_dataset in [False, True]:
        if separate_boundary_dataset:
            mask = [labels > 1]
        else: 
            mask = torch.ones_like(labels, dtype=torch.bool)

        pos, desc, d_desc_d_x = compute_descriptors_derivatives(dataset=dataset, 
                                                                descriptor_function=ComputeDescriptors, 
                                                                n_atoms=n_atoms, 
                                                                separate_boundary_dataset=separate_boundary_dataset)
        
        assert(torch.allclose(desc, ref_distances, atol=1e-3))


        # apply simple NN
        NN = FeedForward(layers = [45, 2, 1])
        out = NN(desc)

        # compute derivatives of out wrt input
        d_out_d_x = torch.autograd.grad(out, pos, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=True )[0]
        # compute derivatives of out wrt descriptors
        d_out_d_d = torch.autograd.grad(out, desc, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=True )[0]
        ref = torch.einsum('badx,bd->bax ',d_desc_d_x,d_out_d_d[mask])
        ref = ref.pow(2).sum(dim=(-2,-1))

        Ref = d_out_d_x[mask].pow(2).sum(dim=(-2,-1))

        # apply smart derivatives
        smart_derivatives = SmartDerivatives(d_desc_d_x, n_atoms=n_atoms)
        right_input = d_out_d_d.squeeze(-1)
        smart_out = smart_derivatives(right_input).sum(dim=1)

        # do checks
        assert(torch.allclose(smart_out, ref))
        assert(torch.allclose(smart_out, Ref))

        smart_out.sum().backward()

if __name__ == "__main__":
    test_smart_derivatives()