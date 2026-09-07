import sys

import torch
from lightning import Trainer

from mlcolvar.data import DictModule
from mlcolvar.io import (
    create_dataset_from_files,
    create_dataset_from_trajectories,
)
from mlcolvar.core.nn.graph.schnet import SchNetModel
from mlcolvar.core.nn.utils import Custom_Sigmoid
from mlcolvar.cvs import DeepTDA


class DeepTDAForKBiasExport(DeepTDA):
    """DeepTDA adapter for exported Kolmogorov-bias models."""

    def __init__(
        self,
        *args,
        sigmoid_p: float = 3.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.sigmoid = Custom_Sigmoid(
            p=sigmoid_p,
        )

    def forward_nn(self, x):
        """Return the raw latent coordinate z."""
        return self.forward(x)


# get arguments
mode = sys.argv[1]

torch.manual_seed(42)


if mode in [
    "descriptors",
    "descriptors-kbias",
]:

    # we get the files from github
    filenames = [
        (
            "https://raw.githubusercontent.com/"
            "EnricoTrizio/alanine_gnn_committor_data/"
            "refs/heads/main/unbiased/A/COLVAR"
        ),
        (
            "https://raw.githubusercontent.com/"
            "EnricoTrizio/alanine_gnn_committor_data/"
            "refs/heads/main/unbiased/B/COLVAR"
        ),
    ]

    # we only load a few points
    load_args = [
        {
            "start": 0,
            "stop": 10,
            "stride": 1,
        },
        {
            "start": 0,
            "stop": 10,
            "stride": 1,
        },
    ]

    # load dataset
    dataset = create_dataset_from_files(
        file_names=filenames,
        filter_args={
            "regex": "x",
        },
        create_labels=True,
    )

    model_arch = [
        45,
        20,
        10,
        1,
    ]


elif mode in [
    "gnn",
    "gnn-kbias",
    "gnn-exported",
    "gnn-kbias-exported",
]:

    # we get the files from github
    filenames = [
        (
            "https://raw.githubusercontent.com/"
            "EnricoTrizio/alanine_gnn_committor_data/"
            "refs/heads/main/unbiased/A/traj_comp.xtc"
        ),
        (
            "https://raw.githubusercontent.com/"
            "EnricoTrizio/alanine_gnn_committor_data/"
            "refs/heads/main/unbiased/B/traj_comp.xtc"
        ),
    ]

    topology = (
        "https://raw.githubusercontent.com/"
        "EnricoTrizio/alanine_gnn_committor_data/"
        "refs/heads/main/unbiased/A/confAvac.gro"
    )

    # we only load a few points
    load_args = [
        {
            "start": 0,
            "stop": 10,
            "stride": 1,
        },
        {
            "start": 0,
            "stop": 10,
            "stride": 1,
        },
    ]

    # load dataset
    dataset = create_dataset_from_trajectories(
        trajectories=filenames,
        topologies=topology,
        cutoff=10.0,
        system_selection="all and not type H",
        load_args=load_args,
        lengths_conversion=10.0,
    )

    # initialize SchNet model
    model_arch = SchNetModel(
        n_out=1,
        dataset_for_initialization=dataset,
        pooling_operation="mean",
        n_bases=8,
        n_layers=2,
        n_filters=8,
        n_hidden_channels=8,
        w_out_after_pool=True,
        aggr="mean",
    )


else:
    raise ValueError(
        "Invalid mode. Use 'descriptors', "
        "'descriptors-kbias', 'gnn', "
        "'gnn-kbias', 'gnn-exported' or "
        "'gnn-kbias-exported'."
    )


# frame in datamodule
datamodule = DictModule(
    dataset,
    lengths=[1],
)


model_options = {
    "n_states": 2,
    "n_cvs": 1,
    "target_centers": [
        -7,
        7,
    ],
    "target_sigmas": [
        0.2,
        0.2,
    ],
    "model": model_arch,
}


# initialize model
if mode == "gnn-kbias-exported":
    model = DeepTDAForKBiasExport(
        **model_options,
        sigmoid_p=3.0,
    )
else:
    model = DeepTDA(
        **model_options,
    )


# get trainer
trainer = Trainer(
    logger=False,
    accelerator="cpu",
    enable_checkpointing=False,
    max_epochs=5,
    enable_model_summary=False,
    limit_val_batches=0,
    num_sanity_val_steps=0,
)


# fit model
trainer.fit(
    model,
    datamodule,
)


# export model
if mode in [
    "gnn-exported",
    "gnn-kbias-exported",
]:
    from mlcolvar.utils.export import export

    model.eval()
    model.cpu()

    example_graph = dataset[
        "data_list"
    ][0]

    if mode == "gnn-kbias-exported":
        k_bias_options = {
            "beta": 0.5,
            "lambd": 1.0,
        }
    else:
        k_bias_options = None

    export(
        model=model,
        example_inputs=example_graph,
        file_name="model.pt2",
        calculate_gradients=True,
        k_bias_options=k_bias_options,
        run_check=False,
    )

else:
    # trace to TorchScript
    traced_model = model.to_torchscript(
        "model.pt",
        method="trace",
    )