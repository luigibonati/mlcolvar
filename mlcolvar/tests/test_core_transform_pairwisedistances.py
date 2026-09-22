import torch

from mlcolvar.core.transform.descriptors.pairwise_distances import PairwiseDistances


def test_pairwise_distances():
    # simple test based on alanine distances
    pos_abs = torch.Tensor([[ 1.4970,  1.3861, -0.0273, -1.4933,  1.5070, -0.1133, -1.4473, -1.4193,
                        -0.0553,  1.4940,  1.4990, -0.2403,  1.4780, -1.4173, -0.3363, -1.4243,
                        -1.4093, -0.4293,  1.3530, -1.4313, -0.4183,  1.3060,  1.4750, -0.4333,
                        1.2970, -1.3233, -0.4643,  1.1670, -1.3253, -0.5354]])

    cell = torch.Tensor([3.0233])
    
    pos_scaled = torch.clone(pos_abs) / cell
    
    pos_abs.requires_grad = True
    pos_scaled.requires_grad = True


    ref_distances = torch.Tensor([[0.1521, 0.2335, 0.2412, 0.3798, 0.4733, 0.4649, 0.4575, 0.5741, 0.6815,
                                0.1220, 0.1323, 0.2495, 0.3407, 0.3627, 0.3919, 0.4634, 0.5885, 0.2280,
                                0.2976, 0.3748, 0.4262, 0.4821, 0.5043, 0.6376, 0.1447, 0.2449, 0.2454,
                                0.2705, 0.3597, 0.4833, 0.1528, 0.1502, 0.2370, 0.2408, 0.3805, 0.2472,
                                0.3243, 0.3159, 0.4527, 0.1270, 0.1301, 0.2440, 0.2273, 0.2819, 0.1482]])
  
    # PBC no scaled coords
    model = PairwiseDistances(n_atoms=10, PBC=True, cell=cell, scaled_coords=False)
    out = model(pos_abs)
    assert(out.reshape(pos_abs.shape[0], -1).shape[-1] == model.out_features)
    assert(torch.allclose(out, ref_distances, atol=1e-3))
    out.sum().backward()

    # PBC no scaled coords slicing
    model = PairwiseDistances(n_atoms=10, PBC=True, cell=cell, scaled_coords=False, slicing_pairs=[[0, 1], [0, 2]])
    out = model(pos_abs)
    assert(torch.allclose(out, ref_distances[:, [0, 1]], atol=1e-3))
    out.sum().backward()

    # PBC and scaled coords
    model = PairwiseDistances(n_atoms=10, PBC=True, cell=cell, scaled_coords=True)
    out = model(pos_scaled)
    assert(out.reshape(pos_scaled.shape[0], -1).shape[-1] == model.out_features)
    assert(torch.allclose(out, ref_distances, atol=1e-3))
    out.sum().backward()

    # PBC and scaled coords slicing
    model = PairwiseDistances(n_atoms=10, PBC=True, cell=cell, scaled_coords=True, slicing_pairs=[[0, 1], [0, 2]])
    out = model(pos_scaled)
    assert(torch.allclose(out, ref_distances[:, [0, 1]], atol=1e-3))
    out.sum().backward()

    # runtime cell is allowed only when init cell is None
    model = PairwiseDistances(n_atoms=10, PBC=True, cell=None, scaled_coords=False)
    _ = model(pos_abs, cell=cell)
    model = PairwiseDistances(n_atoms=10, PBC=True, cell=cell, scaled_coords=False)
    try:
        _ = model(pos_abs, cell=cell)
        raise AssertionError("Expected ValueError when passing `cell` both at init and runtime.")
    except ValueError as e:
        assert "provided at initialization" in str(e)


    # ---------------- test varying-cell case (batched cells) ----------------
    # Mixed cell sizes in the same batch: check batched-cell behavior is
    # consistent with frame-wise evaluation for fixed cutoff.    scales = torch.tensor([1.0, 3.7, 11.0], dtype=pos_abs.dtype)
    scales = torch.tensor([0.9, 1.0, 1.1], dtype=pos_abs.dtype)
    cell_batched = torch.stack([(cell * s).repeat(3) for s in scales], dim=0)

    # Absolute coordinates: keep reduced coordinates fixed by scaling positions with each cell.
    pos_abs_batched = torch.cat([pos_abs * s for s in scales], dim=0)
    pos_abs_batched = pos_abs_batched.clone().detach().requires_grad_(True)

    model = PairwiseDistances(n_atoms=10, PBC=True, cell=None, scaled_coords=False)
    out = model(pos_abs_batched, cell=cell_batched)
    ref_batched_single = torch.cat(
        [model(pos_abs_batched[i:i+1], cell=cell_batched[i]) for i in range(len(scales))],
        dim=0,
    )
    ref_batched_scaled = torch.cat([ref_distances * s for s in scales], dim=0)
    assert(out.reshape(pos_abs_batched.shape[0], -1).shape[-1] == model.out_features)
    assert(torch.allclose(out, ref_batched_single, atol=2e-3))
    assert(torch.allclose(out, ref_batched_scaled, atol=2e-3))
    out.sum().backward()

    model = PairwiseDistances(n_atoms=10, PBC=True, cell=None, scaled_coords=False, slicing_pairs=[[0, 1], [0, 2]])
    out = model(pos_abs_batched, cell=cell_batched)
    assert(torch.allclose(out, ref_batched_single[:, [0, 1]], atol=2e-3))
    assert(torch.allclose(out, ref_batched_scaled[:, [0, 1]], atol=2e-3))
    out.sum().backward()

    # Scaled coordinates: physical distances scale with cell.
    pos_scaled_batched = torch.cat([pos_abs * s / (cell * s) for s in scales], dim=0)
    pos_scaled_batched = pos_scaled_batched.clone().detach().requires_grad_(True)

    model = PairwiseDistances(n_atoms=10, PBC=True, cell=None, scaled_coords=True)
    out = model(pos_scaled_batched, cell=cell_batched)
    ref_scaled_batched_single = torch.cat(
        [model(pos_scaled_batched[i:i+1], cell=cell_batched[i]) for i in range(len(scales))],
        dim=0,
    )
    ref_scaled_batched_scaled = torch.cat([ref_distances * s for s in scales], dim=0)
    assert(out.reshape(pos_scaled_batched.shape[0], -1).shape[-1] == model.out_features)
    assert(torch.allclose(out, ref_scaled_batched_single, atol=2e-3))
    assert(torch.allclose(out, ref_scaled_batched_scaled, atol=2e-3))
    out.sum().backward()

    model = PairwiseDistances(n_atoms=10, PBC=True, cell=None, scaled_coords=True, slicing_pairs=[[0, 1], [0, 2]])
    out = model(pos_scaled_batched, cell=cell_batched)
    assert(torch.allclose(out, ref_scaled_batched_single[:, [0, 1]], atol=2e-3))
    assert(torch.allclose(out, ref_scaled_batched_scaled[:, [0, 1]], atol=2e-3))
    out.sum().backward()
