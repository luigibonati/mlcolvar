import torch
from torch_scatter import scatter

class Smart_square_modulus_nabla_q(torch.nn.Module):
    def __init__(self,
                 der_desc_wrt_coord: torch.Tensor,
                 n_atoms: int,
                 return_device : torch.device,
                 storage_device=None,
                 computation_device=None):
        super().__init__()

        # handle devices 
        self.return_device = return_device

        if storage_device is None:
            storage_device = torch.device('cpu')
        self.storage_device = storage_device

        if computation_device is None:
            computation_device = return_device
        self.computation_device = computation_device

        self.batch_size = len(der_desc_wrt_coord)
        self.n_atoms = n_atoms

        # setup the fixed part of the computation
        self.left, self.mat_ind, self.scatter_indeces = self.setup(der_desc_wrt_coord)


    def setup(self, left_input):
        # all the setup should be done on the CPU
        left_input = left_input.to(self.storage_device)
        # the indeces in mat_ind are: batch, atom, descriptor and dimension
        left, mat_ind = self.create_nonzero_left(left_input)
        scatter_indeces = self.get_scatter_indices(batch_ind = mat_ind[0], atom_ind=mat_ind[1], dim_ind=mat_ind[3])
        return left, mat_ind, scatter_indeces
    
    def create_nonzero_left(self, x):
        # find indeces of nonzero entries of d_dist_d_x
        mat_ind = x.nonzero(as_tuple=True)
        
        # flatten matrix --> big nonzero vector
        x = x.ravel()
        # find indeces of nonzero entries of the flattened matrix
        vec_ind = x.nonzero(as_tuple=True)

        # create vector with the nonzero entries only
        x_vec = x[vec_ind[0].long()]
        del(vec_ind)
        return x_vec, mat_ind
    
    def get_scatter_indices(self, batch_ind, atom_ind, dim_ind):
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
        batch_elements = scatter(torch.ones_like(batch_ind),batch_ind, reduce='sum')
        batch_elements[0] -= 1 # to make the later indexing consistent

        # compute the pointer idxs to the beginning of each batch by summing the number of elements in each batch
        # e.g. [ 0., 17., 35., 53.] NB. These are indeces!
        batch_pointers = torch.Tensor([batch_elements[:i].sum() for i in range(len(batch_elements))])
        del(batch_elements)
        # number of entries in the scattered vector before each batch
        # e.g. [ 0.,  9., 18., 27.]
        markers = not_shifted_indeces[batch_pointers.long()] #highest not_shifted index for each batch
        del(not_shifted_indeces)
        del(batch_pointers)
        cumulative_markers = torch.Tensor([markers[:i+1].sum() for i in range(len(markers))]) # stupid sum of indeces
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

    def forward(self, x):
        if self.left.device != self.computation_device:
            left = self.left.to(self.computation_device)
        else:
            left = self.left
        # get the vector with the derivatives of q wrt the descriptors
        right = self.create_right(x=x, batch_ind=self.mat_ind[0], des_ind=self.mat_ind[2], device=self.computation_device)    

        # do element-wise product
        src = left * right
        del(right)
        del(left)
        out = self.compute_square_modulus(x=src, indeces=self.scatter_indeces, n_atoms=self.n_atoms, batch_size=self.batch_size, device=self.computation_device)
        del(src)
        if self.computation_device == self.return_device:
            return out
        else:
            return out.to(self.return_device)
        
    def create_right(self, x, batch_ind, des_ind, device=None):
        desc_vec = x[batch_ind, des_ind]
        if device is not None:
            desc_vec = desc_vec.to(device)
        return desc_vec


    def compute_square_modulus(self,x, indeces, n_atoms, batch_size, device=None):
        out = scatter(x.to(device), indeces.to(device).long())

        # now make the square
        out = out.pow(2)
        # and get the final sum
        out = out.reshape((batch_size, n_atoms, 3))
        out = out.sum(dim=(-2,-1))
        return out
    
def test_smart_derivatives():
    from mlcolvar.core.transform import PairwiseDistances
    from mlcolvar.core.nn import FeedForward

    # compute some descriptors from positions --> distances
    n_atoms = 3
    pos = torch.Tensor([ [ [0., 0., 0.],
                           [1., 1., 1.],
                           [1., 1., 1.1] ],
                         [ [0., 0., 0.],
                           [1., 1.1, 1.],
                           [1., 1., 1.] ] ]
                      )
    pos.requires_grad = True
    
    real_cell = torch.Tensor([1., 2., 1.])
  
    Dist = PairwiseDistances(n_atoms = 3,
                              PBC = True,
                              cell = real_cell,
                              scaled_coords = False)
    
    dist = Dist(pos)
    
    # compute derivatives of descriptors wrt positions
    aux = []
    for i in range(len(dist[0])):
        aux_der = torch.autograd.grad(dist[:,i], pos, grad_outputs=torch.ones_like(dist[:,i]), retain_graph=True )[0]
        aux.append(aux_der)

    d_dist_d_x = torch.stack(aux, axis=2)

    # apply simple NN
    NN = FeedForward(layers = [3, 2, 1])
    out = NN(dist)

    # compute derivatives of out wrt input
    d_out_d_x = torch.autograd.grad(out, pos, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=True )[0]
    # compute derivatives of out wrt descriptors
    d_out_d_d = torch.autograd.grad(out, dist, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=True )[0]

    # apply smart derivatives
    smart_derivatives = Smart_square_modulus_nabla_q(d_dist_d_x, n_atoms=n_atoms, return_device=torch.device('cpu'), computation_device=torch.device('cpu'))
    right_input = d_out_d_d.squeeze(-1)
    smart_out = smart_derivatives(right_input)

    ref = torch.einsum('badx,bd->bax ',d_dist_d_x,d_out_d_d)
    ref = ref.pow(2).sum(dim=(-2,-1))

    Ref = d_out_d_x.pow(2).sum(dim=(-2,-1))


    assert(torch.allclose(smart_out, ref))
    assert(torch.allclose(smart_out, Ref))

    smart_out.sum().backward()

if __name__ == "__main__":
    test_smart_derivatives()