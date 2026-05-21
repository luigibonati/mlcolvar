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

if mode in ['descriptors', 'descriptors-kbias']:

    # we get the files from github
    filenames = ["https://raw.githubusercontent.com/EnricoTrizio/alanine_gnn_committor_data/refs/heads/main/unbiased/A/COLVAR",     
                "https://raw.githubusercontent.com/EnricoTrizio/alanine_gnn_committor_data/refs/heads/main/unbiased/B/COLVAR"]

    # we only load a few points
    load_args = [{'start' : 0, 'stop' : 10, 'stride' : 1},
                {'start' : 0, 'stop' : 10, 'stride' : 1}]

    # load dataset
    dataset = create_dataset_from_files(file_names=filenames,
                                        filter_args={'regex':'x' }, # select distances between heavy atoms
                                        create_labels=True, 
                                        )

    model_arch = [45, 20, 10, 1]

elif mode in ["gnn", "gnn-kbias"]:
    # we get the files from github
    filenames = ["https://raw.githubusercontent.com/EnricoTrizio/alanine_gnn_committor_data/refs/heads/main/unbiased/A/traj_comp.xtc",     
                "https://raw.githubusercontent.com/EnricoTrizio/alanine_gnn_committor_data/refs/heads/main/unbiased/B/traj_comp.xtc"]

    topology = "https://raw.githubusercontent.com/EnricoTrizio/alanine_gnn_committor_data/refs/heads/main/unbiased/A/confAvac.gro"

    # we only load a few points
    load_args = [{'start' : 0, 'stop' : 10, 'stride' : 1},
                {'start' : 0, 'stop' : 10, 'stride' : 1}]

    # load dataset
    dataset = create_dataset_from_trajectories(trajectories=filenames,
                                               topologies=topology,
                                               cutoff=10.0,
                                               system_selection='all and not type H',
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
