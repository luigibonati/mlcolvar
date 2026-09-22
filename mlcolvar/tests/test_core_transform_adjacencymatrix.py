import torch

from mlcolvar.core.transform.descriptors.eigs_adjacency_matrix import EigsAdjMat
from mlcolvar.core.transform.tools.switching_functions import SwitchingFunctions


def test_eigs_of_adj_matrix():
    n_atoms=2 
    pos = torch.Tensor([ [ [0., 0., 0.], 
                           [1., 1., 1.] ], 
                         [ [0., 0., 0.], 
                           [1., 1.1, 1.] ] ] 
                      ) 
    pos.requires_grad = True 
    cell = torch.Tensor([1., 2., 1.]) 

    cutoff = 1.8 
    switching_function=SwitchingFunctions(in_features=n_atoms*3, name='Fermi', cutoff=cutoff, options={'q':0.05}) 
   
    model = EigsAdjMat(mode='continuous', 
                       cutoff=cutoff,  
                       n_atoms=n_atoms, 
                       PBC=True, 
                       cell=cell, 
                       scaled_coords=False, 
                       switching_function=switching_function) 
    out = model(pos) 
    out.sum().backward() 

    pos = torch.einsum('bij,j->bij', pos, 1/cell) 
    model = EigsAdjMat(mode='continuous', 
                       cutoff=cutoff,  
                       n_atoms=n_atoms, 
                       PBC=True, 
                       cell=cell, 
                       scaled_coords=True, 
                       switching_function=switching_function) 
    out = model(pos) 
    assert(out.shape[-1] == model.out_features) 
    out.sum().backward() 

    # runtime cell is allowed only when init cell is None 
    model = EigsAdjMat(mode='continuous', 
                       cutoff=cutoff, 
                       n_atoms=n_atoms, 
                       PBC=True, 
                       cell=None, 
                       scaled_coords=False, 
                       switching_function=switching_function) 
    _ = model(torch.einsum('bij,j->bij', pos.clone().detach(), cell), cell=cell) 

    model = EigsAdjMat(mode='continuous', 
                       cutoff=cutoff, 
                       n_atoms=n_atoms, 
                       PBC=True, 
                       cell=cell, 
                       scaled_coords=False, 
                       switching_function=switching_function) 
    try: 
        _ = model(torch.einsum('bij,j->bij', pos.clone().detach(), cell), cell=cell) 
        raise AssertionError("Expected ValueError when passing `cell` both at init and runtime.") 
    except ValueError as e: 
        assert "provided at initialization" in str(e) 

    # ---------------- mock varying-cell case ---------------- 
    scales = torch.tensor([0.9, 1.0, 1.1], dtype=pos.dtype) 
    pos_abs = torch.Tensor([[[0., 0., 0.], 
                             [1., 1., 1.]], 
                            [[0., 0., 0.], 
                             [1., 1.1, 1.]]]) 
    pos_batched = torch.cat([pos_abs * s for s in scales], dim=0).clone().detach().requires_grad_(True) 
    frame_scales = scales.repeat_interleave(pos_abs.shape[0]) 
    cell_batched = (cell * frame_scales.unsqueeze(-1)) 

    switching_function = SwitchingFunctions( 
        in_features=n_atoms * 3, 
        name='Fermi', 
        cutoff=cutoff, 
        options={'q': 0.05}, 
    ) 
    model = EigsAdjMat( 
        mode='continuous', 
        cutoff=cutoff, 
        n_atoms=n_atoms, 
        PBC=True, 
        cell=None, 
        scaled_coords=False, 
        switching_function=switching_function, 
    ) 
    out_batched = model(pos_batched, cell=cell_batched) 
    out_frames = torch.cat( 
        [model(pos_batched[i:i+1], cell=cell_batched[i]) for i in range(pos_batched.shape[0])], 
        dim=0, 
    ) 
    assert torch.allclose(out_batched, out_frames, atol=1e-6) 
    out_batched.sum().backward()