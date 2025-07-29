import torch
import gc
import numpy as np
from mlcolvar.utils._code import scatter_sum
from mlcolvar.data import DictDataset
from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors.utils import sanitize_positions_shape

__all__ = ["SmartDerivatives", "compute_descriptors_derivatives"]

class SmartDerivatives(torch.nn.Module):
    """
    Utils to compute efficently (time and memory wise) the derivatives of the model output wrt the positions
    used to compute the input descriptors.
    Rather than computing explicitly the derivatives wrt the positions, we compute those wrt the descriptors (right input)
    and multiply them by the matrix of the derivatives of the descriptors wrt the positions (left input).

    Overview
        Preparation:
            - Finds the non-zero entries of the derivatives of the descriptors wrt the positions (left)
            - Stores such entries in a big 1D tensor
            - Stores the indeces to find such entries in the original derivatives matrix
            - Creates a big 1D tensor of indeces that allows properly taking together the contributions
        Forward:      
            - Use the matrix indeces to retrieve the corresponding elements from the derivatives of output wrt the descriptors (right) 
              into a big 1D tensor
            - Get the single contributions via element-wise multiplication (i.e., of each atom to the output due 
              to a single descriptor along a single space dimension) 
            - Scatter the single contributions to global contributions (of each atom to each output along each space dimension)

        When working with batches or splits the scatter indeces are rescaled from the whole dataset to the batched entry.
    """
    def __init__(self,
                 setup_device : str = 'cpu',
                 force_all_atoms : bool = False
                 ):
        """Initialize the smart derivatives object.
        To setup the class, use the `setup` method.
    
        Parameters
        ----------
        der_desc_wrt_pos : torch.Tensor
            Tensor containing the derivatives of the descriptors wrt the atomic positions
        n_atoms : int
            Number of atoms in the systems, all the atoms should be used in at least one of the descriptors
        setup_device : str
            Device on which to perform the expensive calculations. Either 'cpu' or 'cuda', by default 'cpu'
        force_all_atoms: bool
            Whether to allow the use of atoms that are non involved in the calculation of any descriptor, by default False
        """
        super().__init__()
        self.force_all_atoms = force_all_atoms
        self.setup_device = setup_device

        # auxiliary variable to check if the moduel has been properly set up
        self._check_setup = False

        # auxiliary variable to check if elements have been loaded on computation device
        self._device_preload = False

    def setup(self,
              dataset: DictDataset, 
              descriptor_function: Transform, 
              n_atoms : int, 
              separate_boundary_dataset = False, 
              positions_noise : float = 0.0,
              descriptors_batch_size : int = None
              ) -> DictDataset:
        """Setup the smart derivatives object from a dataset and a descriptor function.
        Returns a properly formatted new dataset with the descriptors as data. 

        Parameters
        ----------
        dataset : DictDataset
            Input dataset containing atomic positions as `data` and the needed entries
        descriptor_function : Transform
            Function to compute the descriptors from the atomic positions, it should be taken from the mlcolvar.core.tranform module
        n_atoms : int
            Number of atoms in the dataset
        separate_boundary_dataset : bool, optional
            Whether to separate the boundary dataset from the variational one, by default False
            NB: Should be used only for mlcolvar.cvs.committor. 
        positions_noise : float, optional
            Order of magnitude of small noise to be added to the positions to avoid atoms having the exact same coordinates on some dimension and thus zero derivatives, by default 0.
            Ideally the smaller the better, e.g., 1e-6 for single precision, even lower for double precision.
        batch_size : int
            Size of batches to process data, useful for heavy computation to avoid memory overflows, if None a singel batch is used, by default None 

        Returns
        -------
        DictDataset
            Updated dataset with the computed descriptors as 'data'.
        """
        
        self.n_atoms = n_atoms
        # compute descriptors and their derivatives from original dataset
        pos, desc, d_desc_d_x = compute_descriptors_derivatives(dataset=dataset, 
                                                                descriptor_function=descriptor_function, 
                                                                n_atoms=n_atoms, 
                                                                separate_boundary_dataset=separate_boundary_dataset,
                                                                positions_noise=positions_noise,
                                                                batch_size=descriptors_batch_size)
        
        # create a new dataset with the descriptors and reference indeces
        smart_dataset = create_smart_dataset(desc=desc,
                                             dataset=dataset,
                                             separate_boundary_dataset=separate_boundary_dataset)
        
        # initialize the fixed part of the calculation of smart derivatives (i.e., left part)
        self._setup_left(left_input=d_desc_d_x, setup_device=self.setup_device)

        self._check_setup = True

        return smart_dataset
       
    def _setup_left(self, 
                    left_input : torch.Tensor, 
                    setup_device : str = 'cpu'):
        """Setup the fixed part of the computation: the non-zero elements of the derivatives of the descriptors wrt the positions and the related indeces
        """
        with torch.no_grad():
            self.total_dataset_length = len(left_input)

            # all the setup should be done on the CPU by default
            left_input = left_input.to(torch.device(setup_device))

            # the indeces in mat_ind are: batch, atom, descriptor and dimension
            self.left, mat_ind = self._create_nonzero_left(left_input)
            
            # save them with clearer names
            self.batch_ind = mat_ind[0].long().detach()
            self.atom_ind = mat_ind[1].long().detach()
            self.desc_ind = mat_ind[2].long().detach()
            self.dim_ind = mat_ind[3].long().detach()

            # get indeces to scatter the contributions to the right place at the end
            self.scatter_indeces, self.batch_shift = self._get_scatter_indices(batch_ind = self.batch_ind, 
                                                                               atom_ind=self.atom_ind, 
                                                                               dim_ind=self.dim_ind)
            self.scatter_indeces = self.scatter_indeces.long().detach()
    

    def _create_nonzero_left(self, x):
        """Find the indeces of the non-zero elements of the left input (i.e., derivatives of descriptors wrt positions)
        """
        # find indeces of nonzero entries of the d_desc_d_pos
        mat_ind = x.nonzero(as_tuple=True)

        # it is possible that some atoms are not involved in any descriptor
        used_atoms = torch.unique(mat_ind[1])
        n_effective_atoms = len(used_atoms)

        if n_effective_atoms < self.n_atoms:
            # find not used atoms
            missing_atoms = torch.arange(self.n_atoms)[torch.logical_not(torch.isin(torch.arange(self.n_atoms), used_atoms))]

            if self.force_all_atoms:
                # add by hand a contribute to at least one batch and one descriptor
                # we use it to add the correct indeces, then we revert it
                x[:, missing_atoms, 0, :] = x[:, missing_atoms, 0, :] + 10
                
                # find indeces of nonzero entries of augmented d_desc_d_pos
                mat_ind = x.nonzero(as_tuple=True)
                # find indeces of nonzero entries of flattened augmented matrix
                vec_ind = x.ravel().nonzero(as_tuple=True)

                # revert the modification
                x[:, missing_atoms, 0, :] = x[:, missing_atoms, 0, :] - 10
            else:
                raise ValueError(f"Some of the input atoms are not used in any of the descriptors. The not used atom IDs are : {missing_atoms}. If you want to include all atoms even if not used swtich the force_all_atoms key on. ")
        else:
            # find indeces of nonzero entries of flattened matrix
            vec_ind = x.ravel().nonzero(as_tuple=True)    
               
        # create vector with the nonzero entries only
        x_vec = x.ravel()[vec_ind[0].long()]
        del(vec_ind)
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
        batch_elements = scatter_sum(torch.ones_like(batch_ind), batch_ind)
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
        return shifted_indeces, batch_shift

    def _preload_on_device(self, device):
        """Preloads the tensors used in the forward pass onto the desired device for speeding up.
        This can be reverted moving everything to cpu using the method `move._to_cpu`.
        """
        for attr in ["left", "batch_ind", "desc_ind", "scatter_indeces", "batch_shift"]:
            if self.__dict__[attr].device != device:
                print(f"[SmartDerivatives] Moving {attr} to {device}")
                self.__setattr__(attr, self.__dict__[attr].to(device))
        print("[SmartDerivatives] To move the preloaded tensors back to cpu, use the `SmartDerivatives.move_to_cpu` method")
        self._device_preload = True


    def move_to_cpu(self):
        """Moves the tensors used in the forward pass onto the cpu."""
        for attr in ["left", "batch_ind", "desc_ind", "scatter_indeces", "batch_shift"]:
            print(f"[SmartDerivatives] Moving {attr} to cpu")
            self.__setattr__(attr, self.__dict__[attr].to(torch.device("cpu")))
        self._device_preload = False


    def forward(self, x : torch.Tensor, ref_idx : torch.Tensor = None):
        """Adds the derivatives of descriptors wrt atomic positions to the derivatives of output using the chain rule only for non-zero contributions.

        Parameters
        ----------
        x : torch.Tensor
            Derivatives of output wrt to descriptors
        ref_idx : torch.Tensor
            Reference indeces of the pristine dataset (i.e., before splitting, shuffling..)

        Returns
        -------
        torch.Tensor
            Derivatives of the output wrt atomic positions, shape (N, n_atoms, n_dim, (n_out)))
        """
        if not self._device_preload:
            self._preload_on_device(device=x.device)

        if ref_idx is None:
            ref_idx = torch.arange(x.size(0), dtype=torch.int, device=x.device)

        # =========================== SORT DATA ==========================
        # order by ref_idx, this way it's easier to handle later
        ref_idx, ordering = torch.sort(ref_idx)
        x = x.index_select(0, ordering)

        # we store the indeces to properly re-order the output
        revert_ordering = torch.empty_like(ordering)
        revert_ordering[ordering] = torch.arange(ordering.size(0), device=ordering.device)
        

        batch_size = x.size(0)

        # =========================== HANDLE BATCHING/SPLITTING ==========================
        # If we have batches we need to get the right: 
        # 1) non-zero elements from self.left
        # 2) scatter indeces from self.scatter_indeces and shift them to be consistent with the batch

        # if there is no batching, the shift to the scatter indeces will be fake
        scatter_indeces = self.scatter_indeces
        shifts = torch.zeros_like(self.scatter_indeces)

        # check, based on ref_idx, which batch entries are used. If there are no batches, we just get a fully true mask
        max_val = max(self.batch_ind.max(), ref_idx.max()) + 1
        lookup = torch.zeros(max_val, dtype=torch.bool, device=self.batch_ind.device)
        lookup[ref_idx] = True
        used_batch = lookup[self.batch_ind]
        
        # if we detect batches/splits we update scatter_indeces and shifts
        if not used_batch.all():
            # get the corresponding scatter indeces, these allow properly recombining the contributions
            scatter_indeces = self.scatter_indeces[used_batch]  
            
            # get the indeces shift due to batches, these map how many entries there were before the indeces we took 
            batch_shift_used = self.batch_shift[used_batch]    # This is increasing but *not* sequential!
        
            # find uniques to get markers and index used batches
            uniques, indeces = torch.unique_consecutive(batch_shift_used, return_inverse=True)   # This is increasing *and also* sequential!
        
            # we need to shift the indeces of each batch so that they start after the ones of the previous batch
            # Get max and min scatter index for each group
            num_groups = uniques.numel()
            scatter_min = torch.full((num_groups,), 1e8, device=x.device, dtype=torch.long)
            scatter_max = torch.full((num_groups,), -1e8, device=x.device, dtype=torch.long)

            scatter_min.scatter_reduce_(0, indeces, scatter_indeces, reduce='amin', include_self=True)
            scatter_max.scatter_reduce_(0, indeces, scatter_indeces, reduce='amax', include_self=True)

            # Compute group spans
            group_spans = (scatter_max - scatter_min + 1)

            # Compute exclusive cumulative sum
            n_previous_entries = torch.cat([torch.zeros(1, device=x.device, dtype=torch.int64),
                                            torch.cumsum(group_spans[:-1], dim=0)])
            
            # get the final shifts tensor, uniques make all of them zero-based, n_previous_entries make shifts them to remove overlaps
            shifts = torch.gather(uniques - n_previous_entries, 0, indeces).to(torch.int64)    
            
        # apply shift to the original scatter_indeces
        scatter_indeces = scatter_indeces - shifts

        # get the used part of the left elements
        left = self.left[used_batch]
        
        # get the vector with non-zero elements of derivatives of q wrt the descriptors
        right = self._create_right(x=x, used_batch=used_batch)
        
        # do element-wise product between:
        #   left:  desc/pos derivatives matrix non-zero elements
        #   right: out/desc derivatives matrix non-zero elements
        if left.shape == right.shape:
            src = left * right
        else:
            src = torch.einsum("j,jr->jr", left, right)

        # sum contributions from different descriptors to the same atoms
        out = self._sum_desc_contributions(x=src, scatter_indeces=scatter_indeces, batch_size=batch_size)
        
        # get the original order in case
        out = out.index_select(0, revert_ordering)
        return out
        
    def _create_right(self, x : torch.Tensor, used_batch : torch.Tensor):        
        """Create a big 1D tensor with the elements of the derivatives of the output 
        wrt the descriptors needed to propagate the derivatives to the positions.
        """
        # NOTE: for batching, x here is already batched and doesn't need slicing
        # make general batch idx consistent with the batch
        _, used_batch_ind = torch.unique_consecutive(self.batch_ind[used_batch], return_inverse=True)

        # descriptors indeces need to be corrected by the used batch
        desc_ind = self.desc_ind[used_batch]
        
        # keep only the non zero elements of right input
        desc_vec = x[used_batch_ind, desc_ind]
        return desc_vec
   

    def _sum_desc_contributions(self, x : torch.Tensor, scatter_indeces : torch.Tensor, batch_size : int):
        """Sums the elements of x according to the indeces to obtain the contribution of each atom to the output due 
              to a single descriptor along a single space dimension"""
        try:
            # single output case
            if scatter_indeces.shape == x.shape:
                # scatter to the right indeces
                out = scatter_sum(x, scatter_indeces)
                # reshape to the right shape
                out = out.reshape((batch_size, self.n_atoms, 3))
            
            # multiple outputs case
            else:
                out = torch.stack([scatter_sum(x[:, i], scatter_indeces) for i in range(x.shape[-1])], dim=1 )
                out = out.reshape((batch_size, self.n_atoms, 3, x.shape[-1]))


        # in case somehting's wrong, it may be because of vanishing gradient components     
        except RuntimeError as e:
            raise RuntimeError(e, f"""It may be that some descriptor have a zero component because a low precision in the positions.
                                Try adding a small random noise to the positions by hand or by tweaking the positions_noise key in get_descriptors_and_derivatives or compute_descriptors_derivatives utils""")
    
        return out


def compute_descriptors_derivatives(dataset, 
                                    descriptor_function, 
                                    n_atoms : int, 
                                    separate_boundary_dataset = False, 
                                    positions_noise : float = 0.0,
                                    batch_size : int = None):
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
            Switch to exculde boundary condition labeled data from the variational loss, by default False
    positions_noise : float
        Order of magnitude of small noise to be added to the positions to avoid atoms having the exact same coordinates on some dimension and thus zero derivatives, by default 0.
        Ideally the smaller the better, e.g., 1e-6 for single precision, even lower for double precision.
    batch_size : int
        Size of batches to process data, useful for heavy computation to avoid memory overflows, if None a singel batch is used, by default None 

    Returns
    -------
    pos : torch.Tensor
        Positions tensor (detached)
    desc : torch.Tensor
        Computed descriptors (detached)
    d_desc_d_pos : torch.Tensor
        Derivatives of desc wrt to pos (detached)
    """
    
    # apply noise if given
    if positions_noise > 0:
        noise = torch.rand_like(dataset['data'], )*positions_noise
        dataset['data'] = dataset['data'] + noise

    # get and prepare positions
    pos = dataset['data']
    labels = dataset['labels']
    pos = sanitize_positions_shape(pos=pos, n_atoms=n_atoms)[0]
    
    # get_device 
    device = pos.device

    # check if to separate boundary data
    if separate_boundary_dataset:
        mask_var = labels.squeeze() > 1
        if mask_var.sum()==0:
            raise(ValueError('No points left after separating boundary and variational datasets. \n If you are using only unbiased data set separate_boundary_dataset=False here and in Committor or don\'t use SmartDerivatives!!'))
    else:
        mask_var = torch.ones_like(labels.squeeze()).to(torch.bool)
    
    # check batches size for calculation
    if batch_size is None or batch_size == -1:
        batch_size = len(pos)
    else:
        if batch_size <= 0:
            raise ( ValueError(f"Batch size must be larger than zero if set! Found {batch_size}"))
    n_batches = int(np.ceil(len(pos) / batch_size))

    # compute descriptors and derivatives
    # we loop over batches and compute everything only for that part of the data, inside we loop over descriptors
    # we save lists and make them proper tensors later
    batch_aux_der = []
    batch_aux_desc = []
    batch_count = 0
    
    for batch_count in range(0, n_batches + 1):
        print(f"Processing batch {batch_count}/{n_batches}", end='\r')

        # get batch slicing indexes, they don't need to be all of the same size
        batch_start, batch_stop = batch_count*batch_size, (batch_count+1) * batch_size
        
        batch_mask_var = mask_var[batch_start:batch_stop]   # separate_dataset mask
        batch_pos = pos[batch_start:batch_stop]             # batch positions
        batch_pos = batch_pos[batch_mask_var, :, :]         # batch_positions for variational dataset only
        batch_pos.requires_grad = True

        if len(batch_pos) > 0:
            batch_desc = descriptor_function(batch_pos)

            # we store things always on the cpu
            batch_aux = []
            
            for i in range(len(batch_desc[0])):
                aux_der = torch.autograd.grad(batch_desc[:,i], batch_pos, grad_outputs=torch.ones_like(batch_desc[:,i]), retain_graph=True )[0].contiguous()
                batch_aux.append(aux_der.detach())
            
            # derivatives
            batch_d_desc_d_pos = torch.stack(batch_aux, axis=2).to('cpu')         # derivatives of this batch
            batch_aux_der.append(batch_d_desc_d_pos.detach().cpu())               # derivatives of all batches

            # descriptors
            batch_aux_desc.append(batch_desc.detach().cpu())

            # cleanup
            del aux_der    
            del batch_pos
            del batch_desc

            gc.collect()
            # to be sure, clean the gpu cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Processed all data in {n_batches} batches!")

    if batch_count == 1:
        d_desc_d_pos = batch_d_desc_d_pos
    else:
        d_desc_d_pos = torch.cat(batch_aux_der, dim=0)
    
    # get descriptors
    desc_var = torch.cat(batch_aux_desc, axis=0)
    
    # we compute the descriptors on the whole dataset to always have all of them, no need for grads   
    if separate_boundary_dataset:
        with torch.no_grad():
            desc_not_var = descriptor_function(pos[~mask_var])
            desc = torch.zeros((len(dataset), desc_not_var.shape[-1]))

            desc[mask_var] = desc_var
            desc[~mask_var] = desc_not_var
    else:
        desc = desc_var

    # detach and move back to original device
    pos = pos.detach().to(device)
    desc = desc.detach().to(device)
    d_desc_d_pos = d_desc_d_pos.detach().to(device)

    # to be sure, clean the gpu cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pos, desc, d_desc_d_pos.squeeze(-1)

def create_smart_dataset(desc, dataset, separate_boundary_dataset):
        """Creates the 'smart' dataset with the descriptors and with the correct reference indeces to handle batching/splitting/shuffling"""
        # check if to separate boundary data
        if separate_boundary_dataset:
            mask_var = dataset["labels"].squeeze() > 1
        else:
            mask_var = torch.ones(len(dataset)).to(torch.bool)

        # create reference indeces for batching
        ref_idx = torch.zeros(len(dataset), dtype=torch.int)
        ref_idx[mask_var] = torch.arange(len(ref_idx[mask_var]), dtype=torch.int)
        ref_idx[~mask_var] = -1

        # update dataset with the descriptors as data
        smart_dataset = DictDataset({'data' : desc.detach(), 
                                     'labels': torch.clone(dataset['labels']), 
                                     'weights' : torch.clone(dataset['weights']),
                                     'ref_idx': ref_idx
                                     })
        
        return smart_dataset
    

def test_smart_derivatives():
    from mlcolvar.core.transform import PairwiseDistances
    from mlcolvar.core.nn import FeedForward
    from mlcolvar.data import DictDataset
    
    default_dtype = torch.get_default_dtype()
    # this way tests are less prone to fail on different OS
    torch.set_default_dtype(torch.float64)

    # full atoms with all distances
    n_atoms_1 = 10
    pos_1 = torch.Tensor([[ 1.4970,  1.3861, -0.0273, -1.4933,  1.5070, -0.1133, -1.4473, -1.4193,
                        -0.0553,  1.4940,  1.4990, -0.2403,  1.4780, -1.4173, -0.3363, -1.4243,
                        -1.4093, -0.4293,  1.3530, -1.4313, -0.4183,  1.3060,  1.4750, -0.4333,
                        1.2970, -1.3233, -0.4643,  1.1670, -1.3253, -0.5354]])
    ref_distances_1 = torch.Tensor([[0.1521, 0.2335, 0.2412, 0.3798, 0.4733, 0.4649, 0.4575, 0.5741, 0.6815,
                                0.1220, 0.1323, 0.2495, 0.3407, 0.3627, 0.3919, 0.4634, 0.5885, 0.2280,
                                0.2976, 0.3748, 0.4262, 0.4821, 0.5043, 0.6376, 0.1447, 0.2449, 0.2454,
                                0.2705, 0.3597, 0.4833, 0.1528, 0.1502, 0.2370, 0.2408, 0.3805, 0.2472,
                                0.3243, 0.3159, 0.4527, 0.1270, 0.1301, 0.2440, 0.2273, 0.2819, 0.1482]])
    force_all_atoms_1 = False
    slicing_pairs_1 = None
    pos_noise_1 = 0
    batch_size_1 = None
    

    # five atoms and only two distances --> useless atoms
    n_atoms_2 = 5
    pos_2 = torch.Tensor([[ 1.4970,  1.3861, -0.0273, -1.4933,  1.5070, -0.1133, -1.4473, -1.4193,
                        -0.0553, 1.4940,  1.4990, -0.2403, 1.4780, -1.4173, -0.3363]])
    ref_distances_2 = torch.Tensor([[0.1521, 0.1220]])
    force_all_atoms_2 = True
    slicing_pairs_2 = [[0, 1], [1, 2]]
    pos_noise_2 = 0
    batch_size_2 = None

    # three atoms, disappearing components and batches
    n_atoms_3 = 3
    pos_3 = torch.Tensor([[ 1.4970,  1.3861, -0.0273, 
                            1.4970,  1.5070, -0.1133, 
                            -1.4473, -1.4193, -0.0553]])
    ref_distances_3 = torch.Tensor([[0.1521, 0.1220]])
    force_all_atoms_3 = True
    slicing_pairs_3 = [[0, 1], [1, 2]]
    pos_noise_3 = 1e-5
    batch_size_3 = 3


    aux_pos = [pos_1, pos_2, pos_3]
    aux_ref_distances = [ref_distances_1, ref_distances_2, ref_distances_3]
    aux_n_atoms = [n_atoms_1, n_atoms_2, n_atoms_3]
    aux_force_all_atoms = [force_all_atoms_1, force_all_atoms_2, force_all_atoms_3]
    aux_slicing_pairs = [slicing_pairs_1, slicing_pairs_2, slicing_pairs_3]
    aux_pos_noise = [pos_noise_1, pos_noise_2, pos_noise_3]
    aux_batch_size = [batch_size_1, batch_size_2, batch_size_3]


    zipped = zip(aux_pos, 
                 aux_ref_distances, 
                 aux_n_atoms, 
                 aux_force_all_atoms, 
                 aux_slicing_pairs,
                 aux_pos_noise,
                 aux_batch_size)
    
    for pos,ref_distances,n_atoms,force_all_atoms,slicing_pairs,pos_noise,batch_size in zipped: 
        pos = pos.repeat(4, 1)
        labels = torch.arange(0, 4)
        if pos_noise != 0:
            labels[-1] = 0
        weights = torch.ones_like(labels)

        dataset = DictDataset({'data' : pos, 'labels' : labels, 'weights': weights})

        cell = torch.Tensor([3.0233])
        ref_distances = ref_distances.repeat(4, 1)
    
        ComputeDescriptors = PairwiseDistances(n_atoms=n_atoms,
                                PBC=True,
                                cell=cell,
                                scaled_coords=False,
                                slicing_pairs=slicing_pairs)
        
        for separate_boundary_dataset in [False, True]:
            if separate_boundary_dataset:
                mask = [labels > 1]
            else: 
                mask = torch.ones_like(labels, dtype=torch.bool)

            pos, desc, d_desc_d_x = compute_descriptors_derivatives(dataset=dataset, 
                                                                    descriptor_function=ComputeDescriptors, 
                                                                    n_atoms=n_atoms, 
                                                                    separate_boundary_dataset=separate_boundary_dataset)
            
            if pos_noise == 0:
                assert(torch.allclose(desc, ref_distances, atol=1e-3))

            # compute descriptors outside to have their derivatives for checks
            pos.requires_grad = True
            desc = ComputeDescriptors(pos)

            # apply simple NN
            NN = FeedForward(layers = [desc.shape[-1], 2, 1])
            out = NN(desc)

            # compute derivatives of out wrt input
            d_out_d_x = torch.autograd.grad(out, pos, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=False )[0]
            # compute derivatives of out wrt descriptors
            d_out_d_d = torch.autograd.grad(out, desc, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=True )[0]
            ref = torch.einsum('badx,bd->bax ',d_desc_d_x,d_out_d_d[mask])
            Ref = d_out_d_x[mask]

            # apply smart derivatives
            smart_derivatives = SmartDerivatives(force_all_atoms=force_all_atoms)
            smart_dataset = smart_derivatives.setup(dataset=dataset,
                                                    descriptor_function=ComputeDescriptors,
                                                    n_atoms=n_atoms,
                                                    separate_boundary_dataset=separate_boundary_dataset,
                                                    positions_noise=pos_noise,
                                                    descriptors_batch_size=batch_size
                                                    )
            # check dataset has the right data
            assert(torch.allclose(smart_dataset['data'], desc, atol=1e-3))

            # check forward
            right_input = d_out_d_d.squeeze(-1)
            smart_out = smart_derivatives(right_input, smart_dataset['ref_idx'][mask])
            
            # do checks
            if pos_noise == 0:
                assert(torch.allclose(smart_out, ref))
                assert(torch.allclose(smart_out, Ref))

            smart_out.sum().backward()


    # Test with multiple outputs
    # compute some descriptors from positions --> distances
    n_atoms = 10
    pos = pos_1
    pos = pos.repeat(4, 1)
    labels = torch.arange(0, 4)
    weights = torch.ones_like(labels)
    dataset = DictDataset({'data' : pos, 'labels' : labels, 'weights': weights})

    cell = torch.Tensor([3.0233])
    ref_distances = ref_distances_1
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

        # compute descriptors outside to have their derivatives for checks
        pos.requires_grad = True
        desc = ComputeDescriptors(pos)

        # apply simple NN
        torch.manual_seed(42)
        NN = FeedForward(layers = [45, 2, 2])
        out = NN(desc)

        # compute derivatives of out wrt input
        d_out_d_x = torch.stack([torch.autograd.grad(out[:, i], pos, grad_outputs=torch.ones_like(out[:, i]), retain_graph=True, create_graph=False )[0] for i in range(out.shape[-1])], dim=3)
        # compute derivatives of out wrt descriptors
        d_out_d_d = torch.stack([torch.autograd.grad(out[:, i], desc, grad_outputs=torch.ones_like(out[:, i]), retain_graph=True, create_graph=True )[0] for i in range(out.shape[-1])], dim=2)
        
        ref = torch.einsum('badx,bdo->baxo ',d_desc_d_x,d_out_d_d[mask])
        Ref = d_out_d_x[mask]

        # apply smart derivatives
        smart_derivatives = SmartDerivatives(force_all_atoms=force_all_atoms)
        smart_dataset = smart_derivatives.setup(dataset=dataset,
                                                descriptor_function=ComputeDescriptors,
                                                n_atoms=n_atoms,
                                                separate_boundary_dataset=separate_boundary_dataset,
                                                positions_noise=pos_noise,
                                                descriptors_batch_size=batch_size
                                                )
        # check dataset has the right data
        assert(torch.allclose(smart_dataset['data'], desc, atol=1e-3))

        # check forward
        right_input = d_out_d_d
        smart_out = smart_derivatives(right_input, smart_dataset['ref_idx'][mask])
    
        assert(torch.allclose(smart_out, ref, atol=1e-3))
        assert(torch.allclose(smart_out, Ref, atol=1e-3))
        smart_out.sum().backward()
    
    # reset orginal default dtype
    torch.set_default_dtype(default_dtype)


            
def test_batched_smart_derivatives():
    from mlcolvar.core.transform import PairwiseDistances
    from mlcolvar.core.nn import FeedForward
    from mlcolvar.data import DictDataset, DictModule

    torch.manual_seed(45)

    # compute some descriptors from positions --> distances
    n_atoms = 3
    pos = torch.Tensor([[ 1.4970,  1.3861, -0.0273, -1.4933,  1.5070, -0.1133, -1.4473, -1.4193,
                        -0.0553
                        ]])
    pos = pos.repeat(20, 1)
    pos = pos + torch.randn_like(pos)*1e-2
    
    labels = torch.arange(0, 4).repeat(5).sort()[0]
    weights = torch.ones_like(labels)

    cell = torch.Tensor([3.0233])
    
    ComputeDescriptors = PairwiseDistances(n_atoms=n_atoms,
                              PBC=True,
                              cell=cell,
                              scaled_coords=False)

    dataset = DictDataset({'data' : pos, 'labels' : labels, 'weights': weights})

    for separate_boundary_dataset in [False, True]:
        print(f"********************************************** {separate_boundary_dataset} **********************************************")
        if separate_boundary_dataset:
            mask = [labels > 1]
        else: 
            mask = torch.ones_like(labels, dtype=torch.bool)

        # apply smart derivatives
        smart_derivatives = SmartDerivatives()
        smart_dataset = smart_derivatives.setup(dataset=dataset,
                                                descriptor_function=ComputeDescriptors,
                                                n_atoms=n_atoms,
                                                separate_boundary_dataset=separate_boundary_dataset
                                                )
        
        pos, desc, d_desc_d_x = compute_descriptors_derivatives(dataset=dataset, 
                                                                descriptor_function=ComputeDescriptors, 
                                                                n_atoms=n_atoms, 
                                                                separate_boundary_dataset=separate_boundary_dataset)

        # compute descriptors outside to have their derivatives for checks
        pos.requires_grad = True
        desc = ComputeDescriptors(pos)

        # check dataset has the right data
        assert(torch.allclose(smart_dataset['data'], desc, atol=1e-3))

        # apply simple NN
        torch.manual_seed(42)
        NN = FeedForward(layers = [3, 2, 1])
        out = NN(desc)

        # here we compute things on the whole dataset and we slice it later to get the right entries
        # compute derivatives of out wrt input
        d_out_d_x = torch.autograd.grad(out, pos, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=False )[0]
        # compute derivatives of out wrt descriptors
        d_out_d_d = torch.autograd.grad(out, desc, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=True )[0]
        # get total reference values
        ref = torch.einsum('badx,bd->bax ', d_desc_d_x, d_out_d_d[mask])
        Ref = d_out_d_x[mask]

        # test for different seeds for dataloader
        for i in [42, 420]:
            print(f"====================== {i} ======================")
            torch.manual_seed(i)
            datamodule = DictModule(smart_dataset, lengths=[0.8, 0.2], batch_size=4, shuffle=True, random_split=True)
            datamodule.setup()

            for loader in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
                for b, batch in enumerate(iter(loader)):
                    print(f"==================== BATCH {b} ====================")
                    aux_dataset = DictDataset(batch)

                    # we have to mimic what happens during training
                    if separate_boundary_dataset:
                        aux_mask = aux_dataset['labels'] > 1
                    else:
                        aux_mask = torch.ones_like(aux_dataset['labels'], dtype=torch.bool)

                    # we get the ref indeces only for the "var" part
                    ref_idx = torch.clone(aux_dataset['ref_idx'])[aux_mask]
                    # we get only the right input for the "var" part
                    right_input = d_out_d_d.squeeze(-1)[ref_idx]
                    # get smart out
                    smart_out = smart_derivatives(right_input, ref_idx)
            
                    # do checks with the reference value for the elements present in the batch
                    assert(torch.allclose(smart_out, ref[ref_idx], atol=1e-3))
                    assert(torch.allclose(smart_out, Ref[ref_idx], atol=1e-3))

                    smart_out.sum().backward(retain_graph=True)

def test_compute_descriptors_and_derivatives():
    from mlcolvar.core.transform import PairwiseDistances

    # full atoms with all distances
    n_atoms = 10
    pos = torch.Tensor([[ 1.4970,  1.3861, -0.0273, -1.4933,  1.5070, -0.1133, -1.4473, -1.4193,
                        -0.0553,  1.4940,  1.4990, -0.2403,  1.4780, -1.4173, -0.3363, -1.4243,
                        -1.4093, -0.4293,  1.3530, -1.4313, -0.4183,  1.3060,  1.4750, -0.4333,
                        1.2970, -1.3233, -0.4643,  1.1670, -1.3253, -0.5354]])
    ref_distances = torch.Tensor([[0.1521, 0.2335, 0.2412, 0.3798, 0.4733, 0.4649, 0.4575, 0.5741, 0.6815,
                                0.1220, 0.1323, 0.2495, 0.3407, 0.3627, 0.3919, 0.4634, 0.5885, 0.2280,
                                0.2976, 0.3748, 0.4262, 0.4821, 0.5043, 0.6376, 0.1447, 0.2449, 0.2454,
                                0.2705, 0.3597, 0.4833, 0.1528, 0.1502, 0.2370, 0.2408, 0.3805, 0.2472,
                                0.3243, 0.3159, 0.4527, 0.1270, 0.1301, 0.2440, 0.2273, 0.2819, 0.1482]])

    pos = pos.repeat(5, 1)
    labels = torch.arange(0, 5)
    weights = torch.ones_like(labels)

    dataset = DictDataset({'data' : pos, 'labels' : labels, 'weights': weights})

    cell = torch.Tensor([3.0233])
    ref_distances = ref_distances.repeat(5, 1)

    ComputeDescriptors = PairwiseDistances(n_atoms=n_atoms,
                            PBC=True,
                            cell=cell,
                            scaled_coords=False,
                            slicing_pairs=None)

    for batch_size in [2,3,5]:    
        for separate_boundary_dataset in [False, True]:
            if separate_boundary_dataset:
                mask = [labels > 1]
            else: 
                mask = torch.ones_like(labels, dtype=torch.bool)

            pos, desc, d_desc_d_x = compute_descriptors_derivatives(dataset=dataset, 
                                                                    descriptor_function=ComputeDescriptors, 
                                                                    n_atoms=n_atoms, 
                                                                    separate_boundary_dataset=separate_boundary_dataset,
                                                                    batch_size=batch_size)
            
            assert(torch.allclose(desc, ref_distances, atol=1e-3))

            # compute descriptors outside to have their derivatives for checks
            pos.requires_grad = True
            desc_ref = ComputeDescriptors(pos)

            aux = []
            # compute derivatives of descriptors wrt positions
            for i in range(len(desc_ref[0])):
                    aux_der = torch.autograd.grad(desc_ref[:, i], pos, grad_outputs=torch.ones_like(desc[:,i]), retain_graph=True )[0]
                    aux.append(aux_der.detach().cpu())
                
            # derivatives
            d_desc_d_x_ref = torch.stack(aux, axis=2) 

            # checks
            assert( torch.allclose(desc, desc_ref) )
            assert( torch.allclose(d_desc_d_x, d_desc_d_x_ref[mask]) )

def test_train_with_smart_derivatives():
    from mlcolvar.core.transform import PairwiseDistances
    from mlcolvar.data import DictModule, DictDataset
    from mlcolvar.cvs import Committor, Generator
    from mlcolvar.cvs.committor.utils import initialize_committor_masses
    from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
    from mlcolvar.explain.sensitivity import sensitivity_analysis

    import lightning

    # committor
    # full atoms with all distances
    n_atoms = 10
    pos = torch.Tensor([[ 1.4970,  1.3861, -0.0273, -1.4933,  1.5070, -0.1133, -1.4473, -1.4193,
                        -0.0553,  1.4940,  1.4990, -0.2403,  1.4780, -1.4173, -0.3363, -1.4243,
                        -1.4093, -0.4293,  1.3530, -1.4313, -0.4183,  1.3060,  1.4750, -0.4333,
                        1.2970, -1.3233, -0.4643,  1.1670, -1.3253, -0.5354]])
    
    pos = pos.repeat(200, 1)
    labels = torch.arange(0, 5, dtype=torch.float32).unsqueeze(-1).repeat(40,1).sort()[0]
    weights = torch.ones_like(labels)
    atomic_masses = initialize_committor_masses(atom_types=[0, 0, 1, 2, 0, 0, 0, 1, 2, 0], 
                                            masses=[12.011, 15.999, 14.007])

    dataset = DictDataset({'data' : pos, 'labels' : labels, 'weights': weights})

    cell = torch.Tensor([3.0233])

    ComputeDescriptors = PairwiseDistances(n_atoms=n_atoms,
                            PBC=True,
                            cell=cell,
                            scaled_coords=False,
                            slicing_pairs=None)
    
    smart_derivatives = SmartDerivatives()
    smart_dataset = smart_derivatives.setup(dataset=dataset, 
                                        descriptor_function=ComputeDescriptors,
                                        n_atoms=n_atoms,
                                        separate_boundary_dataset=True,
                                        descriptors_batch_size=25)
    
    datamodule = DictModule(dataset=smart_dataset, lengths=[0.8, 0.2], batch_size=80)
    
    model = Committor(layers=[45, 10, 1],
                      atomic_masses=atomic_masses,
                      alpha=1,
                      separate_boundary_dataset=True,
                      descriptors_derivatives=smart_derivatives 
                      )
    
    trainer = lightning.Trainer(max_epochs=3, logger=False, enable_checkpointing=False)
    
    trainer.fit(model, datamodule)

    # check that sensitivity works
    sensitivity_analysis(model=model, dataset=smart_dataset)

    # Generator
    kT = 2.49432

    # create friction tensor
    #### This part should be made easier using committor utils TODO
    masses = torch.Tensor([ 12.011, 12.011, 15.999, 14.0067, 12.011, 12.011, 12.011, 15.999, 14.0067, 12.011])
    gamma = 1 / 0.05
    friction = kT / (gamma*masses)
    ref_weights = torch.ones(len(pos))

    dataset = DictDataset({'data' : pos, 'labels' : labels, 'weights': ref_weights})

    # --------------------------------- TRAIN MODEL ---------------------------------
    # ------------ Descriptors as input + SmartDerivatives ------------
    # initialize smart derivatives, we do it explicitly to test different functionalities
    smart_derivatives = SmartDerivatives()
    smart_dataset = smart_derivatives.setup(dataset=dataset,
                                            descriptor_function=ComputeDescriptors,
                                            n_atoms=n_atoms,
                                            separate_boundary_dataset=False)
    
    datamodule = DictModule(smart_dataset, lengths=[0.8, 0.2], random_split=True, shuffle=True)

    # seed for reproducibility
    torch.manual_seed(42)
    options = {"nn": {"activation": "tanh"},
            "optimizer": {"lr": 1e-3, "weight_decay": 1e-5}
            }
    model = Generator(
        r=3,
        layers=[45, 20, 20, 1],
        eta=0.005,
        alpha=0.01,
        friction=friction,
        cell=None,
        descriptors_derivatives=smart_derivatives,
        options=options,
    )

     # save outputs as a reference
    X = smart_dataset["data"]
    q = model(X)

    trainer = lightning.Trainer(
        accelerator='cpu',
        callbacks=None,
        max_epochs=6,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    # fit
    trainer.fit(model, datamodule)

    # save outputs as a reference
    X = smart_dataset["data"]
    q = model(X)

    # compute eigenfunctions
    eigfuncs, eigvals, eigvecs = model.compute_eigenfunctions(dataset=smart_dataset, descriptors_derivatives=smart_derivatives)

    print(eigfuncs.shape)
    print(eigvals.shape)
    print(eigvecs.shape)

    # check that sensitivity works
    sensitivity_analysis(model=model, dataset=smart_dataset)

    
if __name__ == "__main__":
    test_smart_derivatives()
    test_batched_smart_derivatives()
    test_compute_descriptors_and_derivatives()
    test_train_with_smart_derivatives()