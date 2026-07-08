import sys

import torch

from mlcolvar.io import (
    create_dataset_from_files,
    create_dataset_from_trajectories,
    load_dataframe,
)


# get arguments
mode = sys.argv[1]


if mode in [
    "descriptors",
    "descriptors-kbias",
]:

    # load TorchScript model
    model = torch.jit.load(
        "model.pt"
    )

    # load the COLVAR from the driver
    filename = "COLVAR"

    # load dataset
    dataset, colvar = create_dataset_from_files(
        filename,
        filter_args={
            "regex": "x",
        },  # select distances between heavy atoms
        return_dataframe=True,
    )

    # get the CV values
    if mode == "descriptors":
        cv_from_plumed = torch.Tensor(
            colvar["model.node-0"].values
        )

    if mode == "descriptors-kbias":
        cv_from_plumed = torch.Tensor(
            colvar["model.z"].values
        )

    cv_from_python = model(
        dataset["data"]
    ).squeeze()


elif mode in [
    "gnn",
    "gnn-kbias",
    "gnn-exported",
]:

    filename = "COLVAR"
    trajectory = "traj_comp.xtc"
    topology = "ref.pdb"

    # load dataset
    dataset = create_dataset_from_trajectories(
        trajectories=trajectory,
        topologies=topology,
        cutoff=10.0,
        system_selection="all and not type H",
        lengths_conversion=10.0,
    )

    # load the COLVAR from the driver
    colvar = load_dataframe(
        filename
    )

    if mode == "gnn-exported":
        from mlcolvar.utils.export import (
            GraphAdapter,
            load_exported,
        )

        # load AOT-exported model
        model = load_exported(
            "model.pt2"
        )

        # exported PLUMED interface uses LABEL=gnn
        cv_from_plumed = torch.Tensor(
            colvar["gnn.node-0"].values
        )

        cv_values = []

        # AOT model was exported for a single graph, so evaluate
        # the trajectory one frame at a time.
        for graph in dataset["data_list"]:
            inputs = GraphAdapter.data_to_tuple(
                graph,
                device="cpu",
            )

            outputs = model(
                inputs
            )

            # The exported model returns:
            # CV, CV gradients, zero, zero.
            cv = outputs[0]

            cv_values.append(
                cv.squeeze()
            )

        cv_from_python = torch.stack(
            cv_values
        ).squeeze()

    else:
        # load the original TorchScript GNN model
        model = torch.jit.load(
            "model.pt"
        )

        if mode == "gnn":
            cv_from_plumed = torch.Tensor(
                colvar["model.node-0"].values
            )

        if mode == "gnn-kbias":
            cv_from_plumed = torch.Tensor(
                colvar["model.z"].values
            )

        cv_from_python = model(
            dataset.get_graph_inputs()
        ).squeeze()

    print(cv_from_plumed)
    print(cv_from_python)


else:
    raise ValueError(
        "Invalid mode. Use 'descriptors', "
        "'descriptors-kbias', 'gnn', "
        "'gnn-kbias' or 'gnn-exported'."
    )


# Ensure both outputs have the same one-dimensional shape.
cv_from_python = cv_from_python.reshape(-1)
cv_from_plumed = cv_from_plumed.reshape(-1)

print(
    "Maximum absolute error:",
    torch.max(
        torch.abs(
            cv_from_python
            - cv_from_plumed
        )
    ).item(),
)


# check that the results are consistent
check = torch.allclose(
    cv_from_python,
    cv_from_plumed,
    rtol=1e-2,
)

if check:
    sys.exit(0)
else:
    sys.exit(1)