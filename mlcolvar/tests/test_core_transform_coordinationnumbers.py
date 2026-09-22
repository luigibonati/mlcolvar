import torch

from mlcolvar.core.transform.descriptors.coordination_numbers import CoordinationNumbers
from mlcolvar.core.transform.tools.switching_functions import SwitchingFunctions


def test_coordination_number():    
    # simple example based on calixarene water coordination numbers
    pos = torch.Tensor([[[-0.410219, -0.680065, -2.016121],
                         [-0.164329, -0.630426, -2.120843],
                         [-0.250341, -0.392700, -1.534535],
                         [-0.277187, -0.615506, -1.335904],
                         [-0.762276, -1.041939, -1.546581],
                         [-0.200766, -0.851481, -1.534129],
                         [ 0.051099, -0.898884, -1.628219],
                         [-1.257225,  1.671602,  0.166190],
                         [-0.486917, -0.902610, -1.554715],
                         [-0.020386, -0.566621, -1.597171],
                         [-0.507683, -0.541252, -1.540805],
                         [-0.527323, -0.206236, -1.532587]],
                        [[-0.410387, -0.677657, -2.018355],
                         [-0.163502, -0.626094, -2.123348],
                         [-0.250672, -0.389610, -1.536810],
                         [-0.275395, -0.612535, -1.338175],
                         [-0.762197, -1.037856, -1.547382],
                         [-0.200948, -0.847825, -1.536010],
                         [ 0.051170, -0.896311, -1.629396],
                         [-1.257530,  1.674078, 0.165089],
                         [-0.486894, -0.900076, -1.556366],
                         [-0.020235, -0.563252, -1.601229],
                         [-0.507242, -0.537527, -1.543025],
                         [-0.528576, -0.202031, -1.534733]]])

    cell = 4.0273098
    pos.requires_grad = True

    n_atoms = 12
    cutoff=0.25
    switching_function=SwitchingFunctions(in_features=n_atoms*3, name='Rational', cutoff=cutoff, options={'n': 2, 'm' : 6, 'eps' : 1e0})

    model = CoordinationNumbers(group_A=[0, 1],
                                group_B=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                cutoff=cutoff,
                                n_atoms=n_atoms, 
                                PBC=True,
                                cell=cell,
                                mode='continuous',
                                scaled_coords=False,
                                switching_function=switching_function)
    
    out = model(pos)
    out.sum().backward()

    # we swap by hand the 0,1 atoms with 2,3
    pos = torch.Tensor([[[-0.250341, -0.392700, -1.534535],
                         [-0.277187, -0.615506, -1.335904],
                         [-0.410219, -0.680065, -2.016121],
                         [-0.164329, -0.630426, -2.120843],
                         [-0.762276, -1.041939, -1.546581],
                         [-0.200766, -0.851481, -1.534129],
                         [ 0.051099, -0.898884, -1.628219],
                         [-1.257225,  1.671602,  0.166190],
                         [-0.486917, -0.902610, -1.554715],
                         [-0.020386, -0.566621, -1.597171],
                         [-0.507683, -0.541252, -1.540805],
                         [-0.527323, -0.206236, -1.532587]],
                        [[-0.250672, -0.389610, -1.536810],
                         [-0.275395, -0.612535, -1.338175],
                         [-0.410387, -0.677657, -2.018355],
                         [-0.163502, -0.626094, -2.123348],
                         [-0.762197, -1.037856, -1.547382],
                         [-0.200948, -0.847825, -1.536010],
                         [ 0.051170, -0.896311, -1.629396],
                         [-1.257530,  1.674078, 0.165089],
                         [-0.486894, -0.900076, -1.556366],
                         [-0.020235, -0.563252, -1.601229],
                         [-0.507242, -0.537527, -1.543025],
                         [-0.528576, -0.202031, -1.534733]]])
    
    pos.requires_grad = True
    switching_function=SwitchingFunctions(in_features=n_atoms*3, name='Rational', cutoff=cutoff, options={'n': 2, 'm' : 6, 'eps' : 1e0})

    model = CoordinationNumbers(group_A=[2, 3],
                                group_B=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11],
                                cutoff=cutoff,
                                n_atoms=n_atoms, 
                                PBC=True,
                                cell=cell,
                                mode='continuous',
                                scaled_coords=False,
                                switching_function=switching_function)
    
    out_2 = model(pos)
    out_2.sum().backward()
    assert(torch.allclose(out, out_2))

    # check using only subset of atoms
    model = CoordinationNumbers(group_A=[2, 3],
                                group_B=[0, 1, 4, 5, 6],
                                cutoff=cutoff,
                                n_atoms=n_atoms, 
                                PBC=True,
                                cell=cell,
                                mode='continuous',
                                scaled_coords=False,
                                switching_function=switching_function)
    
    out = model(pos)
    out.sum().backward()
    
    # check using dmax
    switching_function=SwitchingFunctions(in_features=n_atoms*3, name='Rational', cutoff=cutoff, dmax=0.6, options={'n': 2, 'm' : 6, 'eps' : 1e0})
    model = CoordinationNumbers(group_A=[2, 3],
                                group_B=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11],
                                cutoff=cutoff,
                                n_atoms=n_atoms, 
                                PBC=True,
                                cell=cell,
                                mode='continuous',
                                scaled_coords=False,
                                switching_function=switching_function,
                                dmax=0.6)
    
    out = model(pos)
    out.sum().backward()

    # runtime cell is allowed only when init cell is None
    model = CoordinationNumbers(group_A=[2, 3],
                                group_B=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11],
                                cutoff=cutoff,
                                n_atoms=n_atoms,
                                PBC=True,
                                cell=None,
                                mode='continuous',
                                scaled_coords=False,
                                switching_function=switching_function)
    _ = model(pos, cell=torch.tensor([cell]))
    model = CoordinationNumbers(group_A=[2, 3],
                                group_B=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11],
                                cutoff=cutoff,
                                n_atoms=n_atoms,
                                PBC=True,
                                cell=cell,
                                mode='continuous',
                                scaled_coords=False,
                                switching_function=switching_function)
    try:
        _ = model(pos, cell=torch.tensor([cell]))
        raise AssertionError("Expected ValueError when passing `cell` both at init and runtime.")
    except ValueError as e:
        assert "provided at initialization" in str(e)


    # ---------------- mock varying-cell case ----------------
    # Mixed cell sizes in the same batch: check batched-cell behavior is
    # consistent with frame-wise evaluation for fixed cutoff.
    pos_base = pos.clone().detach()
    scales = torch.tensor([0.9, 1.0, 1.1], dtype=pos_base.dtype)
    pos_batched = torch.cat([pos_base * s for s in scales], dim=0).clone().detach().requires_grad_(True)
    frame_scales = scales.repeat_interleave(pos_base.shape[0])
    cell_batched = (cell * frame_scales).unsqueeze(-1)

    switching_function = SwitchingFunctions(
        in_features=n_atoms * 3,
        name='Rational',
        cutoff=cutoff,
        options={'n': 2, 'm': 6, 'eps': 1e0},
    )
    model = CoordinationNumbers(
        group_A=[2, 3],
        group_B=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11],
        cutoff=cutoff,
        n_atoms=n_atoms,
        PBC=True,
        cell=None,
        mode='continuous',
        scaled_coords=False,
        switching_function=switching_function,
    )
    out_batched = model(pos_batched, cell=cell_batched)
    out_frames = torch.cat(
        [model(pos_batched[i:i + 1], cell=cell_batched[i]) for i in range(pos_batched.shape[0])],
        dim=0,
    )
    assert torch.allclose(out_batched, out_frames, atol=1e-6)
    out_batched.sum().backward()