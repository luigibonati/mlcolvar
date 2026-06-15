import torch
import sys
from lightning import Trainer

from mlcolvar.data import DictModule
from mlcolvar.io import create_dataset_from_files, create_dataset_from_trajectories
from mlcolvar.core.nn.graph.schnet import SchNetModel
from mlcolvar.cvs import DeepTDA

# get arguments
mode = sys.argv[1]

torch.manual_seed(42)

if mode in ["gnn", "gnn-kbias", "gnn-lr", "gnn-lr-kbias"]:
    
    # check if long range
    if mode in ["gnn-lr", "gnn-lr-kbias"]:
        long_range = True
    elif mode in ["gnn", "gnn-kbias"]:
        long_range = False

    # we get the files from github
    filenames = ["https://github.com/EnricoTrizio/nacl_gnn_data/raw/refs/heads/main/UNBOUND/traj.xyz",     
                 "https://github.com/EnricoTrizio/nacl_gnn_data/raw/refs/heads/main/BOUND/traj.xyz"]

    topology = None

    # we only load a few points
    load_args = [{'start' : 0, 'stop' : 10, 'stride' : 1},
                {'start' : 0, 'stop' : 10, 'stride' : 1}]

    # load dataset
    dataset = create_dataset_from_trajectories(trajectories=filenames,
                                               topologies=topology, 
                                               cutoff=4.0,  
                                               buffer=3.0,
                                               system_selection='type Na or type Cl',
                                               environment_selection='type O',
                                               subsystem_selection='type Na or type Cl' if long_range else None,
                                               long_range_cutoff=10 if long_range else -1,
                                               load_args=load_args,
                                               lengths_conversion=10.0,
                                               )
    
    # initialize SchNet model
    model_arch = SchNetModel(n_out=1,
                             dataset_for_initialization=dataset,
                             pooling_operation="mean",
                             n_bases=8,
                             n_layers=2,
                             n_filters=8,
                             n_hidden_channels=8,
                             w_out_after_pool=True,
                             aggr='mean'
                            )

# frame in datamodule
datamodule = DictModule(dataset, lengths=[1])

# initialize model
model = DeepTDA(n_states=2,
                n_cvs=1,
                target_centers=[-7, 7],
                target_sigmas=[0.2, 0.2],
                model=model_arch)

# get trainer
trainer = Trainer(
    logger=False,
    accelerator='cpu',
    enable_checkpointing=False,
    max_epochs=5,
    enable_model_summary=False,
    limit_val_batches=0,    # this to skip validation
    num_sanity_val_steps=0  # this to skip validation
)

# fit model
trainer.fit(model, datamodule)

# trace to torchscript
traced_model = model.to_torchscript("model.pt", method="trace")
