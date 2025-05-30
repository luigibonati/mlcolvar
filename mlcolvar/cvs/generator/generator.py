import torch
import lightning
from typing import Union, Tuple
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
from mlcolvar.core.loss.generator_loss import GeneratorLoss
from mlcolvar.cvs.generator.utils import compute_eigenfunctions
from mlcolvar.core.loss.committor_loss import SmartDerivatives
from mlcolvar.data import DictDataset

__all__ = ["Generator"]


class Generator(BaseCV, lightning.LightningModule):
    """
    Baseclass for learning a representation for the eigenfunctions of the infinitesimal generator.
    The representation is expressed as a concatenation of the output of r neural networks.
    **Data**: for training it requires a DictDataset with the keys 'data', and 'weights'
    **Loss**: Minimize the representation loss and the orthonormalization loss
    """

    BLOCKS = ["nn"]

    def __init__(self,
                 r: int, # TODO add check on dimensions
                 layers: list,
                 eta: float,
                 alpha: float,
                 friction=None,
                 cell: float = None,
                 descriptors_derivatives : Union[SmartDerivatives, torch.Tensor] = None,
                 options: dict = None,
                 **kwargs
                 ):
        """Define a NN-based generator model

        Parameters
        ----------
        r : int
            Number of eigenfunctions wanted, i.e., number of neural networks to be initialized
        layers : list
            Number of neurons per layer of each of the `r` neural networks
        eta : float
            Hyperparameter for the shift to define the resolvent, i.e., $(\eta I-_mathcal{L})^{-1}$
        alpha : float
            Hyperparamer that scales the contribution of orthonormality loss to the total loss, i.e., L = L_ef + alpha*L_ortho        
        friction: torch.tensor
            Langevin friction, i.e., $\sqrt{k_B*T/(gamma*m_i)}$
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        descriptors_derivatives : Union[SmartDerivatives, torch.Tensor], optional
            Derivatives of descriptors wrt atomic positions (if used) to speed up calculation of gradients, by default None. 
            Can be either:
                - A `SmartDerivatives` object to save both memory and time, see also mlcolvar.core.loss.committor_loss.SmartDerivatives
                - A torch.Tensor with the derivatives to save time, memory-wise could be less efficient
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['nn'] .
        """
        super().__init__(in_features=layers[0], out_features=r, **kwargs)

        # =======  LOSS  =======
        self.loss_fn = GeneratorLoss(r=r,
                                     eta=eta, 
                                     alpha=alpha, 
                                     friction=friction, 
                                     cell=cell,
                                     descriptors_derivatives=descriptors_derivatives
                                     )
        self.r = r
        self.eta = eta
        self.friction = friction
        self.cell = cell

        # these are initialized by compute_eigenfunctions method
        self.evecs = None
        self.evals = None

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======
        # initialize NN turning
        o = "nn"
        # set default activation to tanh
        if "activation" not in options[o]:
            options[o]["activation"] = "tanh"
        self.nn = torch.nn.ModuleList(
            [FeedForward(layers, **options[o]) for idx in range(r)]
        )

    def compute_eigenfunctions(self,
                               dataset : DictDataset,        
                               eta : float = None, 
                               friction : float = None,      
                               cell : float = None,      
                               tikhonov_reg : float = 1e-4,      
                               recompute : bool = False,        
                               descriptors_derivatives : Union[SmartDerivatives, torch.Tensor] = None
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the eigenfunctions based on the representation learned given by the neural networks.

        Parameters
        ----------
        dataset : DictDataset
            Dictionary containing:
            - 'data' : Input descriptors or positions.
            - 'weights' : Biasing weights associated with the data points.
        eta : float, optional
            Set only if different from the one used in training, Hyperparameter for the shift to define the resolvent, i.e., $(\eta I-_mathcal{L})^{-1}$
        friction:torch.tensor, optional
            Set only if different from the one used in training, Langevin friction, i.e., $\sqrt{k_B*T/(gamma*m_i)}$
        cell : float, optional
            Set only if different form the one used in training, CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        tikhonov_reg: float, optional
            Hyperparameter for the regularization of the inverse (Ridge regression parameter)
        recompute: Boolean, optional
            Whether to recompute the eigenfucntions or not, by default False

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            eigenfunctions : torch.Tensor, shape (N, r)
                The computed eigenfunctions evaluated at each data point.
            evals : torch.Tensor, shape (r,)
                The eigenvalues associated with the generator, sorted in descending order.
            evecs : torch.Tensor, shape (r, r)
                The eigenvectors of the operator.
        """
        if friction is None:
            friction = self.friction
        if eta is None:
            eta = self.eta
        if cell is None:
            cell = self.cell

        # If the calculation has not been done previously, or we want to compute again the eigenpairs due to a change of parameters
        if (recompute or self.evecs is None): 
            # get data
            input = dataset["data"]
            weights = dataset['weights']
            input.requires_grad = True

            # get output
            output = self.forward(input)

            # get eigenfunctions
            eigenfunctions, evals, evecs = compute_eigenfunctions(
                input=input,
                output=output,
                weights=weights,
                r=self.r,
                eta=eta,
                friction=friction,
                cell=cell,
                tikhonov_reg=tikhonov_reg,
                descriptors_derivatives=descriptors_derivatives,
            )
            self.evals = evals
            self.evecs = evecs

            return eigenfunctions, evals, evecs

        else:
            eigenfunctions = self.forward(input) @ self.evecs
            return eigenfunctions, self.evals, self.evecs

    def forward_cv(self, 
                   x: torch.Tensor
                   ) -> torch.Tensor:
        return torch.cat([nn(x) for nn in self.nn], dim=1)

    def training_step(self, 
                      train_batch, 
                      batch_idx):
        """Compute and return the training loss and record metrics."""
        # =================get data===================
        x = train_batch["data"]
        # check data are have shape (n_data, -1)
        x = x.reshape((x.shape[0], -1))

        x.requires_grad = True

        weights = train_batch["weights"]

        # =================forward====================
        # we use forward and not forward_cv to also apply the preprocessing (if present)
        q = self.forward(x)
        # ===================loss=====================
        if self.training:
            loss, loss_ef, loss_ortho = self.loss_fn(x, q, weights)
        else:
            loss, loss_ef, loss_ortho = self.loss_fn(x, q, weights)
        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_var", loss_ef, on_epoch=True)
        self.log(f"{name}_loss_ortho", loss_ortho, on_epoch=True)
        return loss


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- TESTS ----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

def test_generator():
    from mlcolvar.cvs.generator import Generator
    from mlcolvar.data import DictModule, DictDataset
    from mlcolvar.core.loss.committor_loss import SmartDerivatives,compute_descriptors_derivatives
    from mlcolvar.core.transform import PairwiseDistances

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
    masses = torch.tensor([ 12.011, 12.011, 15.999, 14.0067, 12.011, 12.011, 12.011, 15.999, 14.0067, 12.011])
    gamma = 1 / 0.05
    friction = torch.zeros(n_atoms * 3)
    for i_atom in range(10):
        friction[3 * i_atom : 3 * i_atom + 3] = torch.Tensor([kT / (gamma * masses[i_atom])] * 3)
    friction = torch.Tensor(friction)


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
    model = Generator(
        r=3,
        layers=[45, 20, 20, 1],
        eta=0.005,
        alpha=0.01,
        friction=friction,
        cell=None,
        descriptors_derivatives=None,
        options=options,
    )

    # here we use the preprocessing
    model.preprocessing = ComputeDistances

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
    X = dataset["data"]
    
    # this is to check other strategies
    ref_output = model(X)

    # this is to check it gives always the same numbers
    check_ref_output = torch.Tensor([[ 0.5640, -0.2441, -0.3938],
                                     [ 0.5692, -0.2448, -0.4029],
                                     [ 0.5725, -0.2469, -0.4076],
                                     [ 0.5494, -0.2433, -0.3780],
                                     [ 0.5468, -0.2386, -0.3818]]
                                     )
    assert( torch.allclose(ref_output, check_ref_output, atol=1e-3))

    # compute eigenfunctions
    ref_eigfuncs, ref_eigvals, ref_eigvecs = model.compute_eigenfunctions(dataset=dataset, descriptors_derivatives=None)
    check_ref_eigfuncs = torch.Tensor([[-1.5081, -1.4932, -1.3340],
                                       [-1.5258, -1.5784, -2.2185],
                                       [-1.5373, -1.6217, -2.2533],
                                       [-1.4677, -1.3893,  0.6337],
                                       [-1.4633, -1.4526, -0.7747]]
                                       )
    
    check_ref_eigvals = torch.Tensor([-0.0043, -0.1379, -0.9307])
    check_ref_eigvecs = torch.Tensor([[  -1.7516,    8.0564,  -60.8461],
                                      [   0.6750,    2.4214, -268.0190],
                                      [   0.9024,   13.8275,   82.3854]]
                                      )

    print(ref_eigfuncs)
    print(ref_eigvals)
    print(ref_eigvecs)

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
    dataset_desc = DictDataset({"data": desc, "weights": dataset["weights"]})
    
    # initialize datamodule, split and shuffle false for derivatives
    datamodule = DictModule(dataset_desc, lengths=[1.0], random_split=False, shuffle=False)
  
    # seed for reproducibility
    torch.manual_seed(42)
    model = Generator(
        r=3,
        layers=[45, 20, 20, 1],
        eta=0.005,
        alpha=0.01,
        friction=friction,
        cell=None,
        descriptors_derivatives=d_desc_d_pos,
        options=options,
    )

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
    X = dataset_desc["data"]
    q = model(X)
    assert( torch.allclose(ref_output, q))

    # compute eigenfunctions
    eigfuncs, eigvals, eigvecs = model.compute_eigenfunctions(dataset=dataset_desc, descriptors_derivatives=d_desc_d_pos)

    print(eigfuncs)
    print(eigvals)
    print(eigvecs)

    assert( torch.allclose(eigfuncs, ref_eigfuncs, atol=1e-3) )
    assert( torch.allclose(eigvals, ref_eigvals, atol=1e-3) )
    assert( torch.allclose(eigvecs, ref_eigvecs, atol=1e-1) ) # eigvecs are larger numbers


    # 3 ------------ Descriptors as input + SmartDerivatives ------------
    # initialize smart derivatives, we do it explicitly to test different functionalities
    smart_derivatives = SmartDerivatives(der_desc_wrt_pos=d_desc_d_pos,
                                         n_atoms=n_atoms,
                                         setup_device='cpu',
                                         force_all_atoms=False)

    # seed for reproducibility
    torch.manual_seed(42)
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
    X = dataset_desc["data"]
    q = model(X)
    assert( torch.allclose(ref_output, q))

    # compute eigenfunctions
    eigfuncs, eigvals, eigvecs = model.compute_eigenfunctions(dataset=dataset_desc, descriptors_derivatives=smart_derivatives)

    print(eigfuncs)
    print(eigvals)
    print(eigvecs)

    assert( torch.allclose(eigfuncs, ref_eigfuncs, atol=1e-3) )
    assert( torch.allclose(eigvals, ref_eigvals, atol=1e-3) )
    assert( torch.allclose(eigvecs, ref_eigvecs, atol=1e-1) ) # eigvecs are larger numbers

if __name__ == '__main__':
    test_generator()