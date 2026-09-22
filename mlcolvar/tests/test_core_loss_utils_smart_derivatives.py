import lightning
import torch

from mlcolvar.core.loss.utils.smart_derivatives import (
    SmartDerivatives,
    compute_descriptors_derivatives,
)
from mlcolvar.core.nn import FeedForward
from mlcolvar.core.transform import PairwiseDistances
from mlcolvar.cvs import Committor, DeepGenerator
from mlcolvar.cvs.committor.utils import initialize_committor_masses
from mlcolvar.data import DictDataset, DictModule
from mlcolvar.explain.sensitivity import sensitivity_analysis


def test_smart_derivatives():
    
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
    do_check_1 = True
    batch_size_1 = None
    

    # five atoms and only two distances --> useless atoms
    n_atoms_2 = 5
    pos_2 = torch.Tensor([[ 1.4970,  1.3861, -0.0273, -1.4933,  1.5070, -0.1133, -1.4473, -1.4193,
                        -0.0553, 1.4940,  1.4990, -0.2403, 1.4780, -1.4173, -0.3363]])
    ref_distances_2 = torch.Tensor([[0.1521, 0.1220]])
    force_all_atoms_2 = True
    slicing_pairs_2 = [[0, 1], [1, 2]]
    do_check_2 = True
    batch_size_2 = None

    # three atoms, disappearing components and batches
    n_atoms_3 = 3
    pos_3 = torch.Tensor([[ 1.4970,  1.3861, -0.0273, 
                            1.4970,  1.5070, -0.1133, 
                            -1.4473, -1.4193, -0.0553]])
    ref_distances_3 = torch.Tensor([[0.1521, 0.1220]])
    force_all_atoms_3 = True
    slicing_pairs_3 = [[0, 1], [1, 2]]
    do_check_3 = False
    batch_size_3 = 3


    aux_pos = [pos_1, pos_2, pos_3]
    aux_ref_distances = [ref_distances_1, ref_distances_2, ref_distances_3]
    aux_n_atoms = [n_atoms_1, n_atoms_2, n_atoms_3]
    aux_force_all_atoms = [force_all_atoms_1, force_all_atoms_2, force_all_atoms_3]
    aux_slicing_pairs = [slicing_pairs_1, slicing_pairs_2, slicing_pairs_3]
    aux_do_check = [do_check_1, do_check_2, do_check_3]
    aux_batch_size = [batch_size_1, batch_size_2, batch_size_3]


    zipped = zip(aux_pos, 
                 aux_ref_distances, 
                 aux_n_atoms, 
                 aux_force_all_atoms, 
                 aux_slicing_pairs,
                 aux_do_check,
                 aux_batch_size)
    
    for pos_original,ref_distances_original,n_atoms,force_all_atoms,slicing_pairs,do_check,batch_size in zipped: 
        for cell_mode in ["fixed", "varying"]:
        
            pos = pos_original.repeat(4, 1)
            labels = torch.arange(0, 4)
            if not do_check:
                labels[-1] = 0
            weights = torch.ones_like(labels)

            if cell_mode == 'fixed':
                cell = torch.Tensor([3.0233])
                dataset = DictDataset({'data' : pos, 'labels' : labels, 'weights': weights})            
                ComputeDescriptors = PairwiseDistances(n_atoms=n_atoms,
                                        PBC=True,
                                        cell=cell,
                                        scaled_coords=False,
                                        slicing_pairs=slicing_pairs)
            elif cell_mode == 'varying':
                cell = torch.Tensor([3.0233]).repeat(len(pos), 1)
                dataset = DictDataset({'data' : pos, 'labels' : labels, 'weights': weights, 'cell': cell})
                ComputeDescriptors = PairwiseDistances(n_atoms=n_atoms,
                                        PBC=True,
                                        cell=None,
                                        scaled_coords=False,
                                        slicing_pairs=slicing_pairs)
                
            ref_distances = ref_distances_original.repeat(4, 1)
            
            for separate_boundary_dataset in [False, True]:
                if separate_boundary_dataset:
                    mask = [labels > 1]
                else: 
                    mask = torch.ones_like(labels, dtype=torch.bool)

                pos, desc, d_desc_d_x = compute_descriptors_derivatives(dataset=dataset, 
                                                                        descriptor_function=ComputeDescriptors, 
                                                                        n_atoms=n_atoms, 
                                                                        separate_boundary_dataset=separate_boundary_dataset)
                
                if do_check:
                    assert(torch.allclose(desc, ref_distances, atol=1e-3))

                # compute descriptors outside to have their derivatives for checks
                pos.requires_grad = True
                desc = ComputeDescriptors(pos, cell=cell if cell_mode == 'varying' else None)

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
                                                        descriptors_batch_size=batch_size
                                                        )
                # check dataset has the right data
                assert(torch.allclose(smart_dataset['data'], desc, atol=1e-3))

                # check forward
                right_input = d_out_d_d.squeeze(-1)
                smart_out = smart_derivatives(right_input, smart_dataset['ref_idx'][mask])
                
                # do checks
                if do_check:
                    assert(torch.allclose(smart_out, ref))
                    assert(torch.allclose(smart_out, Ref))

                smart_out.sum().backward()


    # Test with multiple outputs
    # compute some descriptors from positions --> distances
    n_atoms = 10
    pos_original = pos_1
    ref_distances = ref_distances_1
    force_all_atoms = force_all_atoms_1
    batch_size = batch_size_1

    for cell_mode in ["fixed", "varying"]:
        pos = pos_original.repeat(4, 1)
        labels = torch.arange(0, 4)
        weights = torch.ones_like(labels)

        if cell_mode == "fixed":
            cell = torch.Tensor([3.0233])
            dataset = DictDataset({"data": pos, "labels": labels, "weights": weights})
            ComputeDescriptors = PairwiseDistances(n_atoms=n_atoms,
                                                   PBC=True,
                                                   cell=cell,
                                                   scaled_coords=False)
        elif cell_mode == "varying":
            cell = torch.Tensor([3.0233]).repeat(len(pos), 1)
            dataset = DictDataset({"data": pos, "labels": labels, "weights": weights, "cell": cell})
            ComputeDescriptors = PairwiseDistances(n_atoms=n_atoms,
                                                   PBC=True,
                                                   cell=None,
                                                   scaled_coords=False)
        ref_distances_this = ref_distances.repeat(4, 1)

        for separate_boundary_dataset in [False, True]:
            if separate_boundary_dataset:
                mask = [labels > 1]
            else:
                mask = torch.ones_like(labels, dtype=torch.bool)

            pos, desc, d_desc_d_x = compute_descriptors_derivatives(dataset=dataset,
                                                                    descriptor_function=ComputeDescriptors,
                                                                    n_atoms=n_atoms,
                                                                    separate_boundary_dataset=separate_boundary_dataset)

            assert torch.allclose(desc, ref_distances_this, atol=1e-3)

            # compute descriptors outside to have their derivatives for checks
            pos.requires_grad = True
            desc = ComputeDescriptors(pos, cell=cell if cell_mode == "varying" else None)

            # apply simple NN
            torch.manual_seed(42)
            NN = FeedForward(layers=[45, 2, 2])
            out = NN(desc)

            # compute derivatives of out wrt input
            d_out_d_x = torch.stack([torch.autograd.grad(out[:, i], pos, grad_outputs=torch.ones_like(out[:, i]), retain_graph=True, create_graph=False )[0]
                                     for i in range(out.shape[-1])], dim=3)
            # compute derivatives of out wrt descriptors
            d_out_d_d = torch.stack([torch.autograd.grad(out[:, i], desc, grad_outputs=torch.ones_like(out[:, i]), retain_graph=True, create_graph=True )[0]
                                     for i in range(out.shape[-1])], dim=2)

            ref = torch.einsum("badx,bdo->baxo ", d_desc_d_x, d_out_d_d[mask])
            Ref = d_out_d_x[mask]

            # apply smart derivatives
            smart_derivatives = SmartDerivatives(force_all_atoms=force_all_atoms)
            smart_dataset = smart_derivatives.setup(dataset=dataset,
                                                    descriptor_function=ComputeDescriptors,
                                                    n_atoms=n_atoms,
                                                    separate_boundary_dataset=separate_boundary_dataset,
                                                    descriptors_batch_size=batch_size)
            # check dataset has the right data
            assert torch.allclose(smart_dataset["data"], desc, atol=1e-3)

            # check forward
            right_input = d_out_d_d
            smart_out = smart_derivatives(right_input, smart_dataset["ref_idx"][mask])

            assert torch.allclose(smart_out, ref, atol=1e-3)
            assert torch.allclose(smart_out, Ref, atol=1e-3)
            smart_out.sum().backward()
    
    # reset orginal default dtype
    torch.set_default_dtype(default_dtype)


            
def test_batched_smart_derivatives():

    torch.manual_seed(45)

    # compute some descriptors from positions --> distances
    n_atoms = 3
    pos_original = torch.Tensor([[ 1.4970,  1.3861, -0.0273, -1.4933,  1.5070, -0.1133, -1.4473, -1.4193,
                                 -0.0553 ]])

    for cell_mode in ["fixed", "varying"]:
        pos = pos_original.repeat(20, 1)
        pos = pos + torch.randn_like(pos) * 1e-2

        labels = torch.arange(0, 4).repeat(5).sort()[0]
        weights = torch.ones_like(labels)

        if cell_mode == "fixed":
            cell = torch.Tensor([3.0233])
            dataset = DictDataset({"data": pos, "labels": labels, "weights": weights})
            ComputeDescriptors = PairwiseDistances(n_atoms=n_atoms,
                                                   PBC=True,
                                                   cell=cell,
                                                   scaled_coords=False)
        elif cell_mode == "varying":
            cell = torch.Tensor([3.0233]).repeat(len(pos), 1)
            dataset = DictDataset({"data": pos, "labels": labels, "weights": weights, "cell": cell})
            ComputeDescriptors = PairwiseDistances(n_atoms=n_atoms,
                                                   PBC=True,
                                                   cell=None,
                                                   scaled_coords=False)

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
                                                    separate_boundary_dataset=separate_boundary_dataset)

            pos, desc, d_desc_d_x = compute_descriptors_derivatives(dataset=dataset,
                                                                    descriptor_function=ComputeDescriptors,
                                                                    n_atoms=n_atoms,
                                                                    separate_boundary_dataset=separate_boundary_dataset)

            # compute descriptors outside to have their derivatives for checks
            pos.requires_grad = True
            desc = ComputeDescriptors(pos, cell=cell if cell_mode == "varying" else None)

            # check dataset has the right data
            assert torch.allclose(smart_dataset["data"], desc, atol=1e-3)

            # apply simple NN
            torch.manual_seed(42)
            NN = FeedForward(layers=[3, 2, 1])
            out = NN(desc)

            # here we compute things on the whole dataset and we slice it later to get the right entries
            # compute derivatives of out wrt input
            d_out_d_x = torch.autograd.grad(out, pos, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=False )[0]
            # compute derivatives of out wrt descriptors
            d_out_d_d = torch.autograd.grad(out, desc, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=True )[0]
            # get total reference values
            ref = torch.einsum("badx,bd->bax ", d_desc_d_x, d_out_d_d[mask])
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
                            aux_mask = aux_dataset["labels"] > 1
                        else:
                            aux_mask = torch.ones_like(aux_dataset["labels"], dtype=torch.bool)

                        # we get the ref indeces only for the "var" part
                        ref_idx = torch.clone(aux_dataset["ref_idx"])[aux_mask]
                        # we get only the right input for the "var" part
                        right_input = d_out_d_d.squeeze(-1)[ref_idx]
                        # get smart out
                        smart_out = smart_derivatives(right_input, ref_idx)

                        # do checks with the reference value for the elements present in the batch
                        assert torch.allclose(smart_out, ref[ref_idx], atol=1e-3)
                        assert torch.allclose(smart_out, Ref[ref_idx], atol=1e-3)

                        smart_out.sum().backward(retain_graph=True)

def test_compute_descriptors_and_derivatives():

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

    # ---------------------------------------------------------------------
    # Mock check: repeated runtime cell should reproduce fixed-cell results
    # ---------------------------------------------------------------------
    labels = torch.arange(0, 5)
    weights = torch.ones_like(labels)
    pos_mock = dataset["data"].detach().clone()
    dataset_fixed = DictDataset({'data': pos_mock, 'labels': labels, 'weights': weights})
    repeated_cell = cell.repeat(len(pos_mock), 1)
    dataset_repeated_cell = DictDataset({'data': pos_mock, 'labels': labels, 'weights': weights, 'cell': repeated_cell})

    descriptor_fixed = PairwiseDistances(
        n_atoms=n_atoms,
        PBC=True,
        cell=cell,
        scaled_coords=False,
        slicing_pairs=None,
    )
    descriptor_runtime = PairwiseDistances(
        n_atoms=n_atoms,
        PBC=True,
        cell=None,
        scaled_coords=False,
        slicing_pairs=None,
    )

    for separate_boundary_dataset in [False, True]:
        pos_fix, desc_fix, d_desc_d_x_fix = compute_descriptors_derivatives(
            dataset=dataset_fixed,
            descriptor_function=descriptor_fixed,
            n_atoms=n_atoms,
            separate_boundary_dataset=separate_boundary_dataset,
            batch_size=3,
        )
        pos_rep, desc_rep, d_desc_d_x_rep = compute_descriptors_derivatives(
            dataset=dataset_repeated_cell,
            descriptor_function=descriptor_runtime,
            n_atoms=n_atoms,
            separate_boundary_dataset=separate_boundary_dataset,
            batch_size=3,
        )

        assert torch.allclose(pos_fix, pos_rep, atol=1e-8)
        assert torch.allclose(desc_fix, desc_rep, atol=1e-8)
        assert torch.allclose(d_desc_d_x_fix, d_desc_d_x_rep, atol=1e-8)


def test_compute_descriptors_and_derivatives_varying_cell():

    torch.manual_seed(42)
    n_atoms = 2
    n_frames = 6

    # Reduced coordinates for two atoms and corresponding frame-dependent cells.
    pos_reduced = torch.rand((n_frames, n_atoms, 3))
    pos_reduced[:, 0, :] = 0.1
    pos_reduced[:, 1, :] = 0.9

    cell = torch.stack(
        [
            torch.tensor([2.5, 2.5, 2.5]),
            torch.tensor([3.0, 3.0, 3.0]),
            torch.tensor([3.5, 3.5, 3.5]),
            torch.tensor([2.8, 2.8, 2.8]),
            torch.tensor([3.2, 3.2, 3.2]),
            torch.tensor([2.2, 2.2, 2.2]),
        ],
        dim=0,
    )
    pos_abs = (pos_reduced * cell[:, None, :]).reshape(n_frames, -1)

    labels = torch.arange(n_frames)
    weights = torch.ones_like(labels)
    dataset = DictDataset({"data": pos_abs, "labels": labels, "weights": weights, "cell": cell})

    descriptor = PairwiseDistances(
        n_atoms=n_atoms,
        PBC=True,
        cell=None,
        scaled_coords=False,
        slicing_pairs=[[0, 1]],
    )

    for separate_boundary_dataset in [False, True]:
        if separate_boundary_dataset:
            mask = labels > 1
        else:
            mask = torch.ones_like(labels, dtype=torch.bool)

        pos, desc, d_desc_d_x = compute_descriptors_derivatives(
            dataset=dataset,
            descriptor_function=descriptor,
            n_atoms=n_atoms,
            separate_boundary_dataset=separate_boundary_dataset,
            batch_size=2,
        )

        # Descriptor values should match frame-wise runtime-cell evaluation.
        desc_ref = torch.cat(
            [descriptor(pos[i : i + 1], cell=cell[i]) for i in range(n_frames)],
            dim=0,
        )
        assert torch.allclose(desc, desc_ref, atol=1e-8)

        # Derivatives should match direct autograd on variational subset.
        pos_var = pos[mask].clone().detach().requires_grad_(True)
        cell_var = cell[mask]
        desc_var = descriptor(pos_var, cell=cell_var)
        aux = []
        for i in range(desc_var.shape[1]):
            aux_der = torch.autograd.grad(
                desc_var[:, i],
                pos_var,
                grad_outputs=torch.ones_like(desc_var[:, i]),
                retain_graph=True,
            )[0]
            aux.append(aux_der.detach())
        d_desc_d_x_ref = torch.stack(aux, axis=2)

        assert torch.allclose(d_desc_d_x, d_desc_d_x_ref, atol=1e-8)


def test_train_with_smart_derivatives():

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
    
    model = Committor(model=[45, 10, 1],
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
    model = DeepGenerator(
        r=3,
        model=[45, 20, 20, 3],
        eta=0.005,
        alpha=0.01,
        friction=friction,
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
    eigfuncs, eigvals, eigvecs = model.compute_eigenfunctions(datamodule=datamodule, descriptors_derivatives=smart_derivatives)

    print(eigfuncs.shape)
    print(eigvals.shape)
    print(eigvecs.shape)

    # check that sensitivity works
    sensitivity_analysis(model=model, dataset=smart_dataset)
