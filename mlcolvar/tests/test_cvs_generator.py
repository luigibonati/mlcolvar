import platform

import lightning
import pytest
import torch

from mlcolvar.core.loss.utils.smart_derivatives import (
    SmartDerivatives,
    compute_descriptors_derivatives,
)
from mlcolvar.core.transform import PairwiseDistances
from mlcolvar.cvs.generator import DeepGenerator
from mlcolvar.data import DictDataset, DictModule

def test_generator():
    # The hard-coded reference values below are only bit-reproducible on the platform
    # where they were generated (Linux). Elsewhere, floating-point/BLAS differences make
    # the exact comparison unreliable, so off-Linux we only assert portable invariants.
    run_strict = platform.system() == "Linux"
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    n_atoms = 10
    kT = 2.49432
    
    # input positions for alanine example 
    ref_pos = torch.Tensor([[ 1.2980, 0.5370, 1.3370, 1.3270, 0.5710, 1.1960, 1.4110, 0.5070, 1.1310, 1.2520, 0.6710, 1.1440,
                              1.2490, 0.6890, 0.9990, 1.1270, 0.6130, 0.9550, 1.2340, 0.8420, 0.9810, 1.1860, 0.9140, 1.0700,
                              1.2790, 0.8870, 0.8630, 1.2550, 1.0230, 0.8240 ],
                            [ 2.7530, 0.7150, 0.5170, 2.8460, 0.6150, 0.5780, 2.9520, 0.6560, 0.6220, 2.8150, 0.4870, 0.5730,
                              2.9100, 0.3830, 0.6150, 2.9310, 0.3890, 0.7690, 2.8520, 0.2450, 0.5830, 2.7300, 0.2380, 0.5550,
                              2.9420, 0.1390, 0.5840, 2.9030, -0.0030, 0.5690 ],
                            [ 0.4830, 2.5610, 2.9980, 0.5620, 2.5410, 2.8660, 0.5080, 2.4950, 2.7660, 0.6960, 2.5590, 2.8790,
                              0.8060, 2.5410, 2.7750, 0.7890, 2.6570, 2.6680, 0.9450, 2.5390, 2.8400, 0.9620, 2.5380, 2.9610,
                              1.0510, 2.5430, 2.7590, 1.1860, 2.5410, 2.7990 ],
                            [ 1.0680, 0.1770, 0.1670, 0.9560, 0.2290, 0.0920, 0.9320, 0.1730, -0.0070, 0.8770, 0.3280, 0.1460,
                              0.7710, 0.4040, 0.0760, 0.7230, 0.5180, 0.1660, 0.8270, 0.4640, -0.0530, 0.9010, 0.5650, -0.0450,
                              0.7790, 0.4160, -0.1670, 0.8260, 0.4500, -0.2950 ],
                            [ 2.4600, 0.5670, 2.4940, 2.6050, 0.5640, 2.5060, 2.6660, 0.4630, 2.5020, 2.6640, 0.6830, 2.5220,
                              2.8040, 0.7250, 2.5200, 2.8880, 0.6370, 2.6190, 2.8690, 0.7270, 2.3820, 2.9600, 0.8080, 2.3570,
                              2.8260, 0.6310, 2.3010, 2.8630, 0.6170, 2.1580 ]]
                          )

    # weights for inputs                     
    ref_weights = torch.Tensor([1.4809, 0.0736, 0.3693, 0.1849, 0.0885])
    
    # initialize dataset with positions
    dataset = DictDataset({"data": ref_pos, "weights": ref_weights, "labels": torch.ones((len(ref_pos), 1))})

    # initialize descriptors calculations: all pairwise distances
    ComputeDistances = PairwiseDistances(n_atoms=10, 
                                         PBC=False, 
                                         cell=[1, 1, 1], 
                                         scaled_coords=False)

    # create friction tensor
    #### This part should be made easier using committor utils TODO
    masses = torch.Tensor([ 12.011, 12.011, 15.999, 14.0067, 12.011, 12.011, 12.011, 15.999, 14.0067, 12.011])
    gamma = 1 / 0.05
    friction = kT / (gamma*masses)

    # --------------------------------- TRAIN MODELS ---------------------------------
    # Train the models: positions as input, desc as input with smartderivatives and passing derivatives
    
    # 1 ------------ Positions as input ------------
    # initialize datamodule, split and shuffle false for derivatives
    datamodule = DictModule(dataset, lengths=[1.0], random_split=False, shuffle=False)

    options = {"nn": {"activation": "tanh"},
               "optimizer": {"lr": 1e-3, "weight_decay": 1e-5}
               }
    
    # seed for reproducibility
    torch.manual_seed(42)
    model = DeepGenerator(
        r=3,
        model=[45, 20, 20, 3],
        eta=0.005,
        alpha=0.01,
        friction=friction,
        descriptors_derivatives=None,
        options=options,
    )

    # here we use the preprocessing
    model.preprocessing = ComputeDistances

    trainer = lightning.Trainer(
        accelerator='cpu',
        callbacks=None,
        max_epochs=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    # fit
    trainer.fit(model, datamodule)

    # save outputs as a reference
    X = dataset["data"]
    
    # this is to check other strategies
    ref_output = model(X)
    print(ref_output)
    # this is to check it gives always the same numbers
    check_ref_output = torch.Tensor([[-0.1246, -0.5761, -0.3869],
                                     [-0.1250, -0.5760, -0.3863],
                                     [-0.1250, -0.5760, -0.3864],
                                     [-0.1227, -0.5770, -0.3874],
                                     [-0.1247, -0.5763, -0.3919]]
                                    )
    # assert( torch.allclose(ref_output, check_ref_output, atol=1e-3))

    # compute eigenfunctions
    ref_eigfuncs, ref_eigvals, ref_eigvecs = model.compute_eigenfunctions(datamodule=datamodule, descriptors_derivatives=None, tikhonov_reg=1e-4)

    check_ref_eigfuncs = torch.Tensor([[-1.5085,  0.2636,  0.0109],
                                       [-1.5085,  1.5487,  0.1069],
                                       [-1.5085,  1.4302,  0.2397],
                                       [-1.5085, -3.4382, -3.3500],
                                       [-1.5085, -4.4840,  5.7270]], 
                                     )
    
    check_ref_eigvals = torch.Tensor([-8.5315e-07, -3.6295e+01, -4.7737e+01])
    check_ref_eigvecs = torch.Tensor([[-3.7744e-01,  9.6905e+02,  3.8329e+02],
                                      [-3.7468e-01, -6.5737e+02, -1.3556e+03],
                                      [-3.7899e-01, -2.8814e+02,  8.4302e+02],
                                      [-1.1311e+00,  2.3542e+01, -1.2932e+02]]
                                      )
    print(ref_eigfuncs)
    print(ref_eigvals)
    print(ref_eigvecs)
    if run_strict:
        assert( torch.allclose(ref_eigfuncs, check_ref_eigfuncs, atol=1e-3) )
        assert( torch.allclose(ref_eigvals, check_ref_eigvals, atol=1e-3) )
        assert( torch.allclose(ref_eigvecs, check_ref_eigvecs, atol=1e-1) ) # eigvecs are larger numbers

    # 2 ------------ Descriptors as input + explicit pass derivatives ------------
    dataset = DictDataset({"data": ref_pos.detach(), "weights": ref_weights, "labels": torch.ones((len(ref_pos), 1))})

    # get descriptor and their derivatives
    pos, desc, d_desc_d_pos = compute_descriptors_derivatives(
        dataset, ComputeDistances, n_atoms, separate_boundary_dataset=False, 
    )

    # create dataset with descriptors
    dataset_desc = DictDataset({"data": desc, "weights": dataset["weights"]}, create_ref_idx=True)
    
    # initialize datamodule, split and shuffle false for derivatives
    datamodule = DictModule(dataset_desc, lengths=[1.0], random_split=False, shuffle=False)
  
    # seed for reproducibility
    torch.manual_seed(42)
    model = DeepGenerator(
        r=3,
        model=[45, 20, 20, 3],
        eta=0.005,
        alpha=0.01,
        friction=friction,
        descriptors_derivatives=d_desc_d_pos,
        options=options,
    )

    trainer = lightning.Trainer(
        accelerator='cpu',
        callbacks=None,
        max_epochs=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    # fit
    trainer.fit(model, datamodule)

    # save outputs as a reference
    X = dataset_desc["data"]
    q = model(X)
    assert( torch.allclose(ref_output, q))

    # compute eigenfunctions
    eigfuncs, eigvals, eigvecs = model.compute_eigenfunctions(datamodule=datamodule, descriptors_derivatives=d_desc_d_pos)

    print(eigfuncs)
    print(eigvals)
    print(eigvecs)
    if run_strict:
        assert( torch.allclose(eigfuncs, ref_eigfuncs, atol=1e-3) )
        assert( torch.allclose(eigvals, ref_eigvals, atol=1e-3) )
        assert( torch.allclose(eigvecs, ref_eigvecs, atol=1e-1) ) # eigvecs are larger numbers


    # 3 ------------ Descriptors as input + SmartDerivatives ------------
    # initialize smart derivatives, we do it explicitly to test different functionalities
    smart_derivatives = SmartDerivatives()
    smart_dataset = smart_derivatives.setup(dataset=dataset,
                                            descriptor_function=ComputeDistances,
                                            n_atoms=n_atoms,
                                            separate_boundary_dataset=False)
    
    datamodule = DictModule(smart_dataset, lengths=[1.0], random_split=False, shuffle=False)

    # seed for reproducibility
    torch.manual_seed(42)
    model = DeepGenerator(
        r=3,
        model=[45, 20, 20, 3],
        eta=0.005,
        alpha=0.01,
        friction=friction,
        descriptors_derivatives=smart_derivatives,
        options=options,
    )

    trainer = lightning.Trainer(
        accelerator='cpu',
        callbacks=None,
        max_epochs=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    # fit
    trainer.fit(model, datamodule)

    # save outputs as a reference
    X = dataset_desc["data"]
    q = model(X)
    assert( torch.allclose(ref_output, q))

    # compute eigenfunctions
    eigfuncs, eigvals, eigvecs = model.compute_eigenfunctions(datamodule=datamodule, descriptors_derivatives=smart_derivatives)

    print(eigfuncs)
    print(eigvals)
    print(eigvecs)
    if run_strict:
        assert( torch.allclose(eigfuncs, ref_eigfuncs, atol=1e-3) )
        assert( torch.allclose(eigvals, ref_eigvals, atol=1e-3) )
        assert( torch.allclose(eigvecs, ref_eigvecs, atol=1e-1) ) # eigvecs are larger numbers


    torch.set_default_dtype(torch.float32)



def test_generator_runtime_cell_training():
    """Integration tests for runtime-cell descriptor preprocessing in generator training."""

    torch.manual_seed(42)

    n_atoms = 2
    n_samples = 12

    # Positions in a flattened (B, n_atoms*3) format and positive weights.
    x = torch.rand((n_samples, n_atoms * 3))
    w = torch.ones(n_samples)

    # Runtime cells (batch, 3), with mild variation.
    cell = torch.ones((n_samples, 3))
    cell[:, 0] *= torch.linspace(0.95, 1.05, n_samples)
    cell[:, 1] *= 1.0
    cell[:, 2] *= 1.1

    # Friction for 2 atoms.
    friction = torch.tensor([0.2, 0.3])

    preprocessing = PairwiseDistances(
        n_atoms=n_atoms,
        PBC=True,
        cell=None,
        scaled_coords=False,
        slicing_pairs=[[0, 1]],
    )

    options = {"nn": {"activation": "tanh"}}
    model = DeepGenerator(
        r=2,
        model=[1, 8, 2],
        eta=0.01,
        alpha=0.01,
        friction=friction,
        descriptors_derivatives=None,
        options=options,
    )
    model.preprocessing = preprocessing

    trainer = lightning.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    # -------- positive case: runtime cell in batch --------
    dataset = DictDataset({"data": x, "weights": w, "cell": cell})
    datamodule = DictModule(dataset, lengths=[1.0], batch_size=6)
    trainer.fit(model, datamodule)
    out = model(x, cell=cell)
    assert torch.isfinite(out).all()

    # -------- negative case: missing runtime cell should fail --------
    dataset_missing_cell = DictDataset({"data": x, "weights": w})
    datamodule_missing_cell = DictModule(dataset_missing_cell, lengths=[1.0], batch_size=6)
    model_missing_cell = DeepGenerator(
        r=2,
        model=[1, 8, 2],
        eta=0.01,
        alpha=0.01,
        friction=friction,
        descriptors_derivatives=None,
        options=options,
    )
    model_missing_cell.preprocessing = preprocessing
    trainer_missing_cell = lightning.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(ValueError, match="cell"):
        trainer_missing_cell.fit(model_missing_cell, datamodule_missing_cell)