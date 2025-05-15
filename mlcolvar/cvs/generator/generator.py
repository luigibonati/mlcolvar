import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
from mlcolvar.core.loss import GeneratorLoss, compute_eigenfunctions

__all__ = ["Generator"]
class Generator(BaseCV, lightning.LightningModule):
    """
    Baseclass for learning a representation for the eigenfunctions of the generator. 
    The representation is expressed as a concatenation of the output of r neural networks.
    **Data**: for training it requires a DictDataset with the keys 'data', and 'weights' 
              and optionally 'derivatives' which should contain the descriptors derivatives
    **Loss**: Minimize the representation loss and the orthonormalization loss
    
    """
    BLOCKS = ["nn"]

    def __init__(
        self, 
        layers: list,
        eta: float,
        r: int,
        alpha: float,
        friction = None,
        cell: float = None,
        
        options: dict = None,

        **kwargs,
    ):
        """Define a NN-based generator model

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        eta : float
            Hyperparameter for the shift to define the resolvent. $(\eta I-_mathcal{L})^{-1}$
        r : int
            Hyperparamer for the number of eigenfunctions wanted
        alpha : float
            Hyperparamer that scales the orthonormality loss
        friction: torch.tensor
            Langevin friction which should contain \sqrt{k_B*T/(gamma*m_i)}
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['nn'] .
        """
        super().__init__(in_features=layers[0], out_features=r, **kwargs) 
        
        # =======  LOSS  =======
        self.loss_fn = GeneratorLoss(
                                     eta=eta,
                                     alpha=alpha,
                                     cell=cell,
                                     friction=friction,
                                     n_cvs=r
        )
        self.r = r
        self.eta = eta
        self.friction = friction
        self.cell = cell
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
        self.nn = torch.nn.ModuleList([FeedForward(layers, **options[o]) for idx in range(r)])
    
    def compute_eigenfunctions(self, dataset, friction=None, eta=None,cell=None, tikhonov_reg=1e-4, recompute=False):
        """Computes the eigenfunctions based on the representation learned given by the neural networks. 

        Parameters
        ----------
        dataset : DictDataset
        Dictionary containing:
            - 'data' (torch.Tensor, shape (N, d)): Input descriptors or positions.
            - 'weights' (torch.Tensor, shape (N,)): Biasing weights associated with the data points. 
            - 'derivatives', optional, (torch.Tensor, shape (N,natoms, d, 3)): derivatives of the descriptors with respect to the atomic positions
        friction:torch.tensor, optional
            If different from the one used in training: Langevin friction which should contain \sqrt{k_B*T/(gamma*m_i)}
        eta : float, optional
            If different from the one used in training, Hyperparameter for the shift to define the resolvent. $(\eta I-_mathcal{L})^{-1}$
        
        cell : float, optional
            If different form the one used in training, CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        
        tikhonov_reg: float, optional
            Hyperparameter for the regularization of the inverse (Ridge regression parameter)
        recompute: Boolean, optional
            Is used to know if the eigenvectors are needed to be recomputed or not
        """
        if friction is None:
            friction = self.friction
        if eta is None:
            eta = self.eta
        if cell is None:
            cell = self.cell
        if recompute or self.evecs is None: # If the calculation has not been done previously, or we want to compute again the eigenpairs due to a change of parameters
            dataset["data"].requires_grad = True
            output = self.forward(dataset["data"])
            if "derivatives" in dataset.keys:
                descriptors_derivatives = dataset["derivatives"]
            else:
                descriptors_derivatives = None
            eigenfunctions, evals, evecs = compute_eigenfunctions( dataset["data"], output, dataset["weights"], friction, eta, self.r, cell, tikhonov_reg, descriptors_derivatives=descriptors_derivatives)
            self.evals = evals
            self.evecs = evecs
            return eigenfunctions, evals, evecs
        
        else:
            eigenfunctions = self.forward(dataset["data"]) @ self.evecs.real
            return eigenfunctions, self.evals, self.evecs

    def forward_cv(self, x: torch.Tensor) -> (torch.Tensor):

        return torch.cat([nn(x) for nn in self.nn], dim=1)

    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        torch.set_grad_enabled(True)
        # =================get data===================
        x = train_batch["data"]
        # check data are have shape (n_data, -1)
        x = x.reshape((x.shape[0], -1))

        
        x.requires_grad = True

        weights = train_batch["weights"]
        if "derivatives" in train_batch.keys():
            derivatives = train_batch["derivatives"]
        else:
            derivatives = None

        # =================forward====================
        # we use forward and not forward_cv to also apply the preprocessing (if present)
        q = self.forward(x)
        # ===================loss=====================
        if self.training:
            loss, loss_ef, loss_ortho = self.loss_fn(
                x, q, weights, derivatives 
            )
        else:
            loss, loss_ef, loss_ortho = self.loss_fn(
                x, q, weights, derivatives 
            )
        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_var", loss_ef, on_epoch=True)
        self.log(f"{name}_loss_ortho", loss_ortho, on_epoch=True)
        return loss

def test_generator():
    from mlcolvar.cvs.generator import Generator
    from mlcolvar.data import DictModule, DictDataset
    from mlcolvar.core.loss.committor_loss import compute_descriptors_derivatives
    from mlcolvar.core.transform import PairwiseDistances
    from mlcolvar.utils.trainer import MetricsCallback
    torch.manual_seed(42)
    n_atoms = 10
    kT = 2.49432
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_pos = torch.tensor([[ 1.2980,  0.5370,  1.3370,  1.3270,  0.5710,  1.1960,  1.4110,  0.5070,
            1.1310,  1.2520,  0.6710,  1.1440,  1.2490,  0.6890,  0.9990,  1.1270,
           0.6130,  0.9550,  1.2340,  0.8420,  0.9810,  1.1860,  0.9140,  1.0700,
            1.2790,  0.8870,  0.8630,  1.2550,  1.0230,  0.8240],
        [ 2.7530,  0.7150,  0.5170,  2.8460,  0.6150,  0.5780,  2.9520,  0.6560,
            0.6220,  2.8150,  0.4870,  0.5730,  2.9100,  0.3830,  0.6150,  2.9310,
            0.3890,  0.7690,  2.8520,  0.2450,  0.5830,  2.7300,  0.2380,  0.5550,
            2.9420,  0.1390,  0.5840,  2.9030, -0.0030,  0.5690],
        [ 0.4830,  2.5610,  2.9980,  0.5620,  2.5410,  2.8660,  0.5080,  2.4950,
            2.7660,  0.6960,  2.5590,  2.8790,  0.8060,  2.5410,  2.7750,  0.7890,
            2.6570,  2.6680,  0.9450,  2.5390,  2.8400,  0.9620,  2.5380,  2.9610,
            1.0510,  2.5430,  2.7590,  1.1860,  2.5410,  2.7990],
        [ 1.0680,  0.1770,  0.1670,  0.9560,  0.2290,  0.0920,  0.9320,  0.1730,
            -0.0070,  0.8770,  0.3280,  0.1460,  0.7710,  0.4040,  0.0760,  0.7230,
            0.5180,  0.1660,  0.8270,  0.4640, -0.0530,  0.9010,  0.5650, -0.0450,
            0.7790,  0.4160, -0.1670,  0.8260,  0.4500, -0.2950],
        [ 2.4600,  0.5670,  2.4940,  2.6050,  0.5640,  2.5060,  2.6660,  0.4630,
            2.5020,  2.6640,  0.6830,  2.5220,  2.8040,  0.7250,  2.5200,  2.8880,
            0.6370,  2.6190,  2.8690,  0.7270,  2.3820,  2.9600,  0.8080,  2.3570,
            2.8260,  0.6310,  2.3010,  2.8630,  0.6170,  2.1580]], device=device)
    ref_weights = torch.tensor([1.4809, 0.0736, 0.3693, 0.1849, 0.0885], device=device)
    dataset = DictDataset({"data":ref_pos, "weights":ref_weights, "labels":torch.ones_like(ref_pos)})


    ComputeDistances = PairwiseDistances(n_atoms=10, PBC=False, cell=[1,1,1], scaled_coords=False)
    pos, desc, d_desc_d_pos = compute_descriptors_derivatives(dataset, ComputeDistances, n_atoms, separate_boundary_dataset = False)
    dataset_desc = DictDataset({"data":desc.clone().detach(), "weights":dataset["weights"],"derivatives":d_desc_d_pos.clone().detach()})
    datamodule = DictModule(dataset_desc, lengths=[1.0],random_split=True,shuffle=True)


    #### This part should be made easier using committor utils
    masses = torch.tensor([12.011,12.011,15.999,14.0067,12.011,12.011,12.011,15.999,14.0067,12.011],device=device)
    gamma = 1/0.05
    friction = torch.zeros(n_atoms*3)
    for i_atom in range(10):
        friction[3*i_atom:3*i_atom+3] = torch.tensor([kT / (gamma*masses[i_atom])]*3,device=device) 
    friction = torch.tensor(friction, device=device).to(torch.float32)

    # Train the model

    options = { 'nn':{'activation':'tanh'},
            'optimizer' : {'lr': 1e-3, 'weight_decay': 1e-5}, }
    model = Generator(layers=[45,20,20,1],eta=0.005,r=3,cell=None,alpha=0.01,friction=friction, options=options)
    metrics = MetricsCallback()

    trainer = lightning.Trainer(callbacks=[metrics],#,early_stop_callback], 
                                max_epochs=6, 
                                enable_progress_bar=False,
                                limit_val_batches=0, num_sanity_val_steps=0
                                )
    trainer.fit(model, datamodule
                )
    ##### Now we compare outputs:
    X = dataset_desc["data"].detach()
    X.requires_grad= True
    q = model.to(device)(X)
    ref_output = torch.tensor([[ 0.5618, -0.2438, -0.3921],
            [ 0.5670, -0.2444, -0.4012],
            [ 0.5703, -0.2465, -0.4059],
            [ 0.5473, -0.2430, -0.3764],
            [ 0.5447, -0.2383, -0.3802]], device=device)
  
    assert(torch.allclose(q, ref_output, atol=1e-3))
    ref_total_loss = -8.4130
    ref_los_var = -8.4345
    ref_loss_ortho = 0.0215
    total_loss, loss_var, loss_ortho = model.loss_fn(X, q, ref_weights, dataset_desc["derivatives"])


    #Compute eigenfunctions from the learned representation:
    ref_eigenfunctions = torch.tensor([[ 1.5082,  0.2829,  0.9739],
            [ 1.5256, -1.8986,  0.1828],
            [ 1.5369, -2.5428, -1.9121],
            [ 1.4681,  3.4370, -3.2778],
            [ 1.4635,  0.7059, -1.7016]], device=device)
    ref_evals =torch.tensor([-3.6827e-03+0.j, -6.2541e+01+0.j, -7.1030e+02+0.j], device=device)

    ref_evecs = torch.tensor([[ 1.8062e+00+0.j,  1.3173e+02+0.j,  9.4339e+02+0.j],
            [-6.6215e-01+0.j, -2.3322e+02+0.j,  1.3187e+03+0.j],
            [-8.4665e-01+0.j,  3.3298e+02+0.j,  5.2938e+02+0.j]], device=device)
  
    eigenfunctions, evals, evecs = model.compute_eigenfunctions(dataset_desc)
    print(evecs-ref_evecs)
    assert(torch.allclose(eigenfunctions, ref_eigenfunctions, atol=1e-3))
    assert(torch.allclose(evals, ref_evals, atol=1e-3))
    assert(torch.allclose(evecs.real, ref_evecs.real, atol=1e-1)) #Values are very big, so I put a lower tolerance
