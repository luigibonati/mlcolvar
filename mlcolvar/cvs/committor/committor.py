import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
from mlcolvar.core.loss import CommittorLoss
from mlcolvar.core.nn.utils import Custom_Sigmoid

__all__ = ["Committor"]


class Committor(BaseCV, lightning.LightningModule):
    """Base class for data-driven learning of committor function.
    The committor function q is expressed as the output of a neural network optimized with a self-consistent
    approach based on the Kolmogorov's variational principle for the committor and on the imposition of its boundary conditions. 

    **Data**: for training it requires a DictDataset with the keys 'data', 'labels' and 'weights'

    **Loss**: Minimize Kolmogorov's variational functional of q and impose boundary condition on the metastable states (CommittorLoss)
    
    References
    ----------
    .. [*] P. Kang, E. Trizio, and M. Parrinello, "Computing the committor using the committor to study the transition state ensemble", Nat. Comput. Sci., 2024, DOI: 10.1038/s43588-024-00645-0
    .. [*] E. Trizio, P. Kang and M. Parrinello, "Everything everywhere all at once: a probability-based enhanced sampling approach to rare events", Nat. Comput. Sci., 2025, DOI: 10.1038/s43588-025-00799-5

    See also
    --------
    mlcolvar.cvs.committor.utils.compute_committor_weights
        Utils to compute the appropriate weights for the training set
    mlcolvar.cvs.committor.utils.initialize_committor_masses
        Utils to initialize the masses tensor for the training
    mlcolvar.core.loss.CommittorLoss
        Kolmogorov's variational optimization of committor and imposition of boundary conditions
    mlcolvar.core.loss.utils.SmartDerivatives
        Class to optimize the gradients calculation imporving speed and memory efficiency.
    """

    BLOCKS = ["nn", "sigmoid"]

    def __init__(
        self, 
        layers: list,
        atomic_masses: torch.Tensor,
        alpha: float,
        gamma: float = 10000,
        delta_f: float = 0,
        cell: float = None,
        separate_boundary_dataset : bool = True,
        descriptors_derivatives : torch.nn.Module = None,
        log_var: bool = False,
        z_regularization: float = 0.0,
        z_threshold: float = None,
        n_dim : int = 3,
        options: dict = None,
        **kwargs,
    ):
        """Define a NN-based committor model

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        atomic_masses : torch.Tensor
            List of masses of all the atoms we are using, for each atom we need to repeat three times for x,y,z.
            The mlcolvar.cvs.committor.utils.initialize_committor_masses can be used to simplify this.
        alpha : float
            Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
        gamma : float, optional
            Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound), by default 10000
        delta_f : float, optional
            Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
            State B is supposed to be higher in energy.
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        separate_boundary_dataset : bool, optional
            Switch to exculde boundary condition labeled data from the variational loss, by default True
        descriptors_derivatives : torch.nn.Module, optional
            `SmartDerivatives` object to save memory and time when using descriptors.
            See also mlcolvar.core.loss.committor_loss.SmartDerivatives
        log_var : bool, optional
            Switch to minimize the log of the variational functional, by default False.
        z_regularization : float, optional
            Scales a regularization on the learned z space preventing it from exceeding the threshold given with 'z_threshold'.
            The magnitude of the regularization is scaled by the given number, by default 0.0
        z_threshold : float, optional
            Sets a maximum threshold for the z value during the training, by default None. 
            The magnitude of the regularization term is scaled via the `z_regularization` key.
        n_dim : int
            Number of dimensions, by default 3.
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['nn'] .
        """
        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs) 
        
        # =======  LOSS  =======
        self.loss_fn = CommittorLoss(atomic_masses=atomic_masses,
                                     alpha=alpha,
                                     gamma=gamma,
                                     delta_f=delta_f,
                                     cell=cell,
                                     separate_boundary_dataset=separate_boundary_dataset,
                                     descriptors_derivatives=descriptors_derivatives,
                                     log_var=log_var,
                                     z_regularization=z_regularization,
                                     z_threshold=z_threshold,
                                     n_dim=n_dim
        )

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======
        # initialize NN turning
        o = "nn"
        # set default activation to tanh
        if "activation" not in options[o]: 
            options[o]["activation"] = "tanh"
        self.nn = FeedForward(layers, **options[o])

        # separately add sigmoid activation on last layer, this way it can be deactived
        o = "sigmoid"
        if (options[o] is not False) and (options[o] is not None):
            self.sigmoid = Custom_Sigmoid(**options[o])

    def forward_nn(self, x):
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        z = self.nn(x)
        return z

    def training_step(self, train_batch, batch_idx):
        torch.set_grad_enabled(True)

        """Compute and return the training loss and record metrics."""
        # =================get data===================
        x = train_batch["data"]
        # check data are have shape (n_data, -1)
        x = x.reshape((x.shape[0], -1))
        x.requires_grad = True

        labels = train_batch["labels"]
        weights = train_batch["weights"]
        try:
            ref_idx = train_batch["ref_idx"]
        except KeyError:
            ref_idx = None

        # =================forward====================
        z = self.forward_nn(x)
        
        if self.sigmoid is not None:
            q = self.sigmoid(z)
        else:
            q = z        
        
        # ===================loss=====================
        if self.training:
            loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
                x, z, q, labels, weights, ref_idx 
            )
        else:
            loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
                x, z, q, labels, weights, ref_idx 
            )

        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_var", loss_var, on_epoch=True)
        self.log(f"{name}_loss_bound_A", loss_bound_A, on_epoch=True)
        self.log(f"{name}_loss_bound_B", loss_bound_B, on_epoch=True)
        return loss


def test_committor():
    from mlcolvar.data import DictDataset, DictModule
    from mlcolvar.cvs.committor.utils import initialize_committor_masses, KolmogorovBias
    
    torch.manual_seed(42)
    # create two fake atoms and use their fake positions
    atomic_masses = initialize_committor_masses(atom_types=[0,1], masses=[15.999, 1.008])
    # create dataset
    samples = 20
    X = torch.randn((4*samples, 6))
    
    # create labels
    y = torch.zeros(X.shape[0])
    y[samples:] += 1
    y[int(2*samples):] += 1
    y[int(3*samples):] += 1
    
    # create weights
    w = torch.ones(X.shape[0])

    dataset = DictDataset({"data": X, "labels": y, "weights": w})
    datamodule = DictModule(dataset, lengths=[1])
    
    # train model
    trainer = lightning.Trainer(max_epochs=5, logger=None, enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0)
    
    # dataset separation
    ref_out = torch.Tensor([[0.6544],[0.6197],[0.5898],[0.5733],[0.6533],[0.5534],[0.5616],[0.6202],[0.5582],[0.5430],
                            [0.6364],[0.5984],[0.6382],[0.5967],[0.6440],[0.5697],[0.6061],[0.6010],[0.6399],[0.6172],
                            [0.6164],[0.6528],[0.6583],[0.6236],[0.6641],[0.5834],[0.5832],[0.6204],[0.6409],[0.6558],
                            [0.5891],[0.5879],[0.5890],[0.6583],[0.6577],[0.6467],[0.6405],[0.6590],[0.6463],[0.5581],
                            [0.6154],[0.6368],[0.6196],[0.5162],[0.5998],[0.6041],[0.5513],[0.6476],[0.5742],[0.6162],
                            [0.6462],[0.6371],[0.5295],[0.6148],[0.5999],[0.5870],[0.6352],[0.6145],[0.5708],[0.4992],
                            [0.6539],[0.6014],[0.6470],[0.6299],[0.6254],[0.5268],[0.6286],[0.6056],[0.6077],[0.6055],
                            [0.5861],[0.5991],[0.6449],[0.6500],[0.6295],[0.5627],[0.6269],[0.6392],[0.5961],[0.6694]])
    ref_bias = torch.Tensor([-6.2043, -6.8591, -7.7645, -7.8704, -5.8342, -7.5036, -7.8780, -6.9957,
                             -7.8679, -7.7473, -7.2451, -7.6833, -6.7631, -7.7863, -6.6693, -7.6212,
                             -7.6929, -7.5685, -6.6894, -7.4857, -7.5187, -4.9488, -6.4961, -7.3898,
                             -6.0350, -7.8837, -7.8748, -7.2552, -7.1221, -5.8647, -7.9190, -7.7184,
                             -7.7073, -4.7898, -5.4073, -5.9113, -6.5451, -4.7149, -5.8899, -7.7421,
                             -7.3999, -7.3456, -7.3005, -7.5067, -7.7396, -7.7099, -7.8664, -6.3275,
                             -7.8864, -7.7243, -6.4288, -5.7041, -7.9351, -7.1991, -7.7027, -7.7947,
                             -6.7121, -7.6094, -7.9009, -7.0479, -5.2398, -7.8241, -5.8642, -7.0701,
                             -7.0348, -7.2577, -6.6142, -7.6322, -7.3279, -7.6393, -7.8608, -7.7037,
                             -6.6949, -6.3947, -7.2246, -7.7009, -6.7359, -7.2186, -7.7849, -5.6882])
    model = Committor(layers=[6, 4, 2, 1], atomic_masses=atomic_masses, alpha=1e-1, delta_f=0)
    trainer.fit(model, datamodule)
    out = model(X)
    out.sum().backward()
    assert( torch.allclose(out, ref_out, atol=1e-3) )
    bias_model = KolmogorovBias(input_model=model, beta=1, epsilon=1e-6, lambd=1)
    bias = bias_model(X)
    assert( torch.allclose(bias, ref_bias, atol=1e-3) )


    # naive whole dataset
    ref_out = torch.Tensor([[0.1206],[0.0688],[0.0941],[0.1026],[0.0739],[0.1279],[0.1115],[0.0629],[0.0994],[0.1012],
                            [0.0886],[0.1218],[0.0785],[0.0704],[0.0948],[0.1193],[0.0877],[0.0964],[0.0774],[0.0874],
                            [0.0948],[0.0636],[0.0869],[0.0664],[0.0659],[0.0927],[0.0654],[0.0927],[0.0743],[0.0787],
                            [0.0802],[0.1074],[0.1105],[0.0595],[0.0693],[0.0620],[0.0688],[0.0669],[0.0591],[0.0986],
                            [0.0706],[0.1180],[0.0894],[0.1030],[0.1012],[0.0606],[0.1408],[0.0766],[0.1063],[0.1049],
                            [0.0749],[0.0588],[0.1177],[0.1127],[0.1090],[0.0806],[0.0954],[0.0799],[0.1048],[0.1378],
                            [0.0783],[0.1384],[0.0689],[0.0649],[0.0983],[0.1548],[0.0778],[0.0934],[0.0858],[0.1203],
                            [0.1073],[0.1139],[0.0716],[0.0988],[0.0918],[0.1109],[0.0918],[0.0928],[0.1070],[0.0742]])
    trainer = lightning.Trainer(max_epochs=5, logger=None, enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0)
    model = Committor(layers=[6, 4, 2, 1], atomic_masses=atomic_masses, alpha=1e-1, delta_f=0, separate_boundary_dataset=False)
    trainer.fit(model, datamodule)
    out = model(X)
    out.sum().backward()
    assert( torch.allclose(out, ref_out, atol=1e-3) )

    # test log loss
    ref_out = torch.Tensor([[0.7286],[0.6506],[0.5595],[0.6759],[0.7482],[0.6804],[0.7313],[0.6763],[0.6874],[0.6267],
                            [0.6363],[0.8129],[0.5853],[0.5263],[0.6359],[0.5264],[0.4840],[0.7292],[0.6884],[0.6376],
                            [0.6232],[0.6997],[0.5906],[0.6248],[0.5876],[0.7198],[0.6356],[0.5933],[0.6229],[0.7093],
                            [0.5619],[0.5006],[0.7924],[0.6965],[0.6541],[0.5477],[0.6151],[0.7042],[0.6190],[0.5363],
                            [0.6275],[0.5959],[0.7194],[0.6123],[0.4874],[0.6654],[0.6742],[0.7011],[0.7207],[0.5864],
                            [0.6041],[0.7643],[0.6696],[0.6424],[0.6886],[0.5776],[0.6620],[0.7105],[0.7517],[0.7387],
                            [0.7714],[0.5826],[0.6442],[0.5796],[0.6132],[0.5923],[0.7023],[0.5731],[0.7308],[0.6404],
                            [0.5781],[0.6850],[0.5960],[0.6718],[0.6626],[0.6069],[0.7319],[0.5498],[0.6772],[0.5847]])
    trainer = lightning.Trainer(max_epochs=5, logger=None, enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0)
    model = Committor(layers=[6, 4, 2, 1], atomic_masses=atomic_masses, alpha=1e-1, delta_f=0, log_var=True)
    trainer.fit(model, datamodule)
    out = model(X)
    out.sum().backward()
    assert( torch.allclose(out, ref_out, atol=1e-3) )

    # test z regularization
    ref_out = torch.Tensor([[0.2878],[0.1591],[0.1665],[0.1166],[0.1349],[0.1053],[0.1544],[0.1113],[0.1435],[0.1232],
                            [0.1130],[0.1261],[0.1726],[0.2098],[0.2091],[0.1407],[0.1942],[0.1400],[0.1382],[0.1630],
                            [0.1573],[0.1742],[0.1613],[0.1289],[0.1703],[0.1390],[0.1184],[0.2557],[0.1520],[0.1328],
                            [0.2220],[0.2254],[0.1823],[0.1426],[0.1744],[0.2594],[0.1105],[0.1390],[0.1557],[0.1985],
                            [0.1340],[0.1971],[0.1429],[0.1270],[0.2239],[0.1134],[0.1999],[0.1416],[0.1707],[0.2238],
                            [0.2054],[0.1560],[0.2357],[0.2971],[0.1445],[0.1906],[0.2130],[0.1457],[0.1382],[0.1432],
                            [0.1337],[0.1444],[0.1603],[0.1396],[0.2043],[0.1964],[0.1459],[0.2243],[0.1930],[0.1893],
                            [0.2634],[0.1868],[0.1340],[0.2483],[0.1550],[0.1559],[0.1614],[0.2020],[0.1270],[0.2555]])
    trainer = lightning.Trainer(max_epochs=5, logger=None, enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0)
    model = Committor(layers=[6, 4, 2, 1], atomic_masses=atomic_masses, alpha=1e-1, delta_f=0, z_regularization=100, z_threshold=0.000001)
    trainer.fit(model, datamodule)
    out = model(X)
    out.sum().backward()
    assert( torch.allclose(out, ref_out, atol=1e-3) )

    # test z_regularization errors
    trainer = lightning.Trainer(max_epochs=5, logger=None, enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0)
    for z_regularization, z_threshold in zip([10,   0,      -1,     10], 
                                             [None, 10,      1,     -1]):
        try:
            model = Committor(layers=[6, 4, 2, 1], atomic_masses=atomic_masses, alpha=1e-1, delta_f=0, z_regularization=z_regularization, z_threshold=z_threshold, n_dim=2)
            trainer.fit(model, datamodule)
        except ValueError as e:
            print("[TEST LOG] Checked this error: ", e)

    # test dimension error
    try:
        trainer = lightning.Trainer(max_epochs=5, logger=None, enable_checkpointing=False, limit_val_batches=0, num_sanity_val_steps=0)
        model = Committor(layers=[6, 4, 2, 1], atomic_masses=atomic_masses, alpha=1e-1, delta_f=0, z_regularization=10, z_threshold=1, n_dim=2)
        trainer.fit(model, datamodule)
    except RuntimeError as e:
        print("[TEST LOG] Checked this error: ", e)




def test_committor_with_derivatives():
    from mlcolvar.cvs.committor.utils import initialize_committor_masses
    from mlcolvar.data import DictModule, DictDataset
    from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives, compute_descriptors_derivatives
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
    dataset = DictDataset({"data": ref_pos, "weights": ref_weights, "labels": torch.arange((len(ref_pos)))})

    # initialize descriptors calculations: all pairwise distances
    ComputeDistances = PairwiseDistances(n_atoms=10, 
                                         PBC=False, 
                                         cell=[1, 1, 1], 
                                         scaled_coords=False)

    # create friction tensor
    masses = initialize_committor_masses(atom_types=[0,0,1,2,0,0,0,1,2,0], 
                                         masses=[ 12.011, 12.011, 15.999, 14.0067, 12.011, 12.011, 12.011, 15.999, 14.0067, 12.011])


    # --------------------------------- TRAIN MODELS ---------------------------------
    # Train the models: positions as input, desc as input with smartderivatives and passing derivatives
    for separate_boundary_dataset in [False, True]:
    
        # 1 ------------ Positions as input ------------
        # initialize datamodule
        torch.manual_seed(42)
        datamodule = DictModule(dataset, lengths=[1.0])
    
        # seed for reproducibility
        model = Committor(layers=[45, 20, 1],
                        atomic_masses=masses,
                        alpha=1, 
                        separate_boundary_dataset=separate_boundary_dataset)

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
        if separate_boundary_dataset:
            ref_output_check = torch.Tensor([[0.4759],
                                            [0.4765],
                                            [0.4828],
                                            [0.4786],
                                            [0.4725]])
        else:
            ref_output_check = torch.Tensor([[0.4756],
                                            [0.4762],
                                            [0.4825],
                                            [0.4783],
                                            [0.4723]])
            
        assert( (torch.allclose(ref_output, ref_output_check, atol=1e-3)))

        if not separate_boundary_dataset:
            # 2 ------------ Descriptors as input + explicit pass derivatives ------------
            # get descriptor and their derivatives
            pos, desc, d_desc_d_pos = compute_descriptors_derivatives(dataset=dataset,
                                                                    descriptor_function=ComputeDistances,
                                                                    n_atoms=n_atoms,
                                                                    separate_boundary_dataset=separate_boundary_dataset)

            dataset_desc = DictDataset({"data": desc, "weights": ref_weights, "labels": torch.arange((len(ref_pos)))}, create_ref_idx=True)

            # seed for reproducibility
            torch.manual_seed(42)
            datamodule = DictModule(dataset_desc, lengths=[1.0])
            
            model = Committor(layers=[45, 20, 1],
                            atomic_masses=masses,
                            alpha=1, 
                            separate_boundary_dataset=separate_boundary_dataset,
                            descriptors_derivatives=d_desc_d_pos)
            
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
            
            # this is to check other strategies
            ref_output = model(X)
            assert( (torch.allclose(ref_output, ref_output_check, atol=1e-3)))

            # test errors
            try:
                # separate boundary with explicit derivatives
                model = Committor(layers=[45, 20, 1],
                            atomic_masses=masses,
                            alpha=1, 
                            separate_boundary_dataset=True,
                            descriptors_derivatives=d_desc_d_pos)
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
                trainer.fit(model, datamodule)
            except ValueError as e:
                print("[TEST LOG] Checked this error: ", e)


        # 3 ------------ Descriptors as input + SmartDerivatives ------------
        # initialize smart derivatives, we do it explicitly to test different functionalities
        smart_derivatives = SmartDerivatives()
        smart_dataset = smart_derivatives.setup(dataset=dataset,
                                                descriptor_function=ComputeDistances,
                                                n_atoms=n_atoms,
                                                separate_boundary_dataset=separate_boundary_dataset)
        
        # seed for reproducibility
        torch.manual_seed(42)
        datamodule = DictModule(smart_dataset, lengths=[1.0])

        model = Committor(layers=[45, 20, 1],
                        atomic_masses=masses,
                        alpha=1, 
                        separate_boundary_dataset=separate_boundary_dataset,
                        descriptors_derivatives=smart_derivatives)
        
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
        
        # this is to check other strategies
        ref_output = model(X)
        assert( (torch.allclose(ref_output, ref_output_check, atol=1e-3)))

        # test errors
        try:
            # no ref_idx!
            wrong_dataset = DictDataset(data=smart_dataset['data'], labels=smart_dataset['labels'], weights=smart_dataset['weights'])
            wrong_datamodule = DictModule(wrong_dataset, lengths=[1.0])
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
            trainer.fit(model, wrong_datamodule)
        except ValueError as e:
            print("[TEST LOG] Checked this error: ", e)

if __name__ == "__main__":
    test_committor()
    test_committor_with_derivatives()