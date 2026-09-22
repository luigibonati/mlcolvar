import numpy as np
import torch

from mlcolvar.core.transform.descriptors.torsional_angles import TorsionalAngles


def test_torsional_angle():
    # simple test on alanine phi angle
    pos = torch.Tensor([[ 1.4970,  1.3861, -0.0273, -1.4933,  1.5070, -0.1133, -1.4473, -1.4193,
                        -0.0553,  1.4940,  1.4990, -0.2403,  1.4780, -1.4173, -0.3363, -1.4243,
                        -1.4093, -0.4293,  1.3530, -1.4313, -0.4183,  1.3060,  1.4750, -0.4333,
                        1.2970, -1.3233, -0.4643,  1.1670, -1.3253, -0.5354],
                        [ 1.4932,  1.3759, -0.0133,  1.4651,  1.4984, -0.1044,  1.4294, -1.4193,
                        -0.0564,  1.4877,  1.4869, -0.2352,  1.4949, -1.4240, -0.3326, -1.3827,
                        -1.4189, -0.3863,  1.3877, -1.4264, -0.4353,  1.3727,  1.4973, -0.5063,
                        1.3086, -1.3215, -0.4382,  1.2070, -1.3063, -0.5443]])
    pos.requires_grad = True

    ref_phi = torch.Tensor([[-2.3687], [-2.0190]])
    cell = torch.Tensor([3.0233, 3.0233, 3.0233])

    # Test single angle (backward compatible)
    model = TorsionalAngles(indices=[1,3,4,6], n_atoms=10, mode=['angle'], PBC=True, cell=cell, scaled_coords=False)
    angle = model(pos)
    assert(torch.allclose(angle, ref_phi, atol=1e-3))
    angle.sum().backward()

    # Test multiple angles
    model = TorsionalAngles(indices=[[1,3,4,6], [1,3,4,6]], n_atoms=10, mode=['angle'], PBC=True, cell=cell, scaled_coords=False)
    angle = model(pos)
    assert(angle.shape == (2, 2))  # [batch_size, n_angles]
    assert(torch.allclose(angle[:, 0], ref_phi.squeeze(), atol=1e-3))
    assert(torch.allclose(angle[:, 1], ref_phi.squeeze(), atol=1e-3))
    angle.sum().backward()

    # Test multiple angles with multiple modes
    model = TorsionalAngles(indices=[[1,3,4,6], [1,3,4,6]], n_atoms=10, mode=['angle', 'sin', 'cos'], PBC=True, cell=cell, scaled_coords=False)
    angle = model(pos)
    assert(angle.shape == (2, 6))  # [batch_size, n_angles * n_modes]
    angle.sum().backward()

    # Original tests
    model = TorsionalAngles(np.array([1,3,4,6]), n_atoms=10, mode=['angle', 'sin', 'cos'], PBC=True, cell=cell, scaled_coords=False)
    angle = model(pos)
    angle.sum().backward()
    assert(torch.allclose(angle[:, 0].unsqueeze(-1), ref_phi, atol=1e-3))
    assert(torch.allclose(angle[:, 1].unsqueeze(-1), torch.sin(ref_phi), atol=1e-3))
    assert(torch.allclose(angle[:, 2].unsqueeze(-1), torch.cos(ref_phi), atol=1e-3))

    model = TorsionalAngles(torch.Tensor([1,3,4,6]), n_atoms=10, mode=['sin', 'cos'], PBC=True, cell=cell, scaled_coords=False)
    angle = model(pos)
    angle.sum().backward()
    assert(torch.allclose(angle[:, 0].unsqueeze(-1), torch.sin(ref_phi), atol=1e-3))
    assert(torch.allclose(angle[:, 1].unsqueeze(-1), torch.cos(ref_phi), atol=1e-3))

    # runtime cell is allowed only when init cell is None
    model = TorsionalAngles(np.array([1,3,4,6]), n_atoms=10, mode=['angle'], PBC=True, cell=None, scaled_coords=False)
    _ = model(pos, cell=cell)
    model = TorsionalAngles(np.array([1,3,4,6]), n_atoms=10, mode=['angle'], PBC=True, cell=cell, scaled_coords=False)
    try:
        _ = model(pos, cell=cell)
        raise AssertionError("Expected ValueError when passing `cell` both at init and runtime.")
    except ValueError as e:
        assert "provided at initialization" in str(e)

    # ---------------- test varying-cell case (batched cells) ----------------
    # Mix different cell sizes in the same batch.
    scales = torch.tensor([0.9, 1.0, 1.1], dtype=pos.dtype)
    pos_base = pos.clone().detach()
    pos_abs_batched = torch.cat([pos_base * s for s in scales], dim=0).clone().detach().requires_grad_(True)
    frame_scales = scales.repeat_interleave(pos_base.shape[0])
    cell_batched = torch.stack([cell * s for s in frame_scales], dim=0)

    # Absolute coordinates: torsional quantities are invariant to uniform scaling.
    model = TorsionalAngles(np.array([1,3,4,6]), n_atoms=10, mode=['angle', 'sin', 'cos'], PBC=True, cell=None, scaled_coords=False)
    out = model(pos_abs_batched, cell=cell_batched)
    out_single = torch.cat(
        [model(pos_abs_batched[i:i+1], cell=cell_batched[i]) for i in range(pos_abs_batched.shape[0])],
        dim=0,
    )
    assert(torch.allclose(out, out_single, atol=1e-6))
    ref_stack = torch.cat([torch.cat([ref_phi, torch.sin(ref_phi), torch.cos(ref_phi)], dim=1) for _ in scales], dim=0)
    assert(torch.allclose(out, ref_stack, atol=1e-3))
    out.sum().backward()

    # Scaled coordinates with varying cells: same invariance.
    pos_scaled_batched = torch.cat(
        [(pos_base * s).reshape(-1, 10, 3) / (cell * s) for s in scales],
        dim=0,
    ).reshape(-1, 30).clone().detach().requires_grad_(True)
    model = TorsionalAngles(np.array([1,3,4,6]), n_atoms=10, mode=['angle', 'sin', 'cos'], PBC=True, cell=None, scaled_coords=True)
    out = model(pos_scaled_batched, cell=cell_batched)
    out_single = torch.cat(
        [model(pos_scaled_batched[i:i+1], cell=cell_batched[i]) for i in range(pos_scaled_batched.shape[0])],
        dim=0,
    )
    assert(torch.allclose(out, out_single, atol=1e-6))
    assert(torch.allclose(out, ref_stack, atol=1e-3))
    out.sum().backward()
