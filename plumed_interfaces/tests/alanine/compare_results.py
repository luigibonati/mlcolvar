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
    "gnn-kbias-exported",
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

    if mode == "gnn-kbias-exported":
        from mlcolvar.utils.export import (
            GraphAdapter,
            load_exported,
        )

        # load AOT-exported Kolmogorov-bias model
        model = load_exported(
            "model.pt2"
        )

        z_from_plumed = torch.Tensor(
            colvar["gnn.z"].values
        )

        q_from_plumed = torch.Tensor(
            colvar["gnn.q"].values
        )

        kbias_from_plumed = torch.Tensor(
            colvar["gnn.kbias"].values
        )

        z_values = []
        q_values = []
        kbias_values = []

        # Evaluate one trajectory frame at a time.
        for graph in dataset["data_list"]:
            inputs = GraphAdapter.data_to_tuple(
                graph,
                device="cpu",
            )

            outputs = model(
                inputs
            )

            # outputs[0] has shape [1, 2]:
            # outputs[0][:, 0] = z
            # outputs[0][:, 1] = q
            z_values.append(
                outputs[0][0, 0]
                .detach()
                .cpu()
            )

            q_values.append(
                outputs[0][0, 1]
                .detach()
                .cpu()
            )

            # outputs[2] contains the Kolmogorov bias.
            kbias_values.append(
                outputs[2]
                .reshape(-1)[0]
                .detach()
                .cpu()
            )

        z_from_python = torch.stack(
            z_values
        )

        q_from_python = torch.stack(
            q_values
        )

        kbias_from_python = torch.stack(
            kbias_values
        )

        print("z from PLUMED:")
        print(z_from_plumed)

        print("z from Python:")
        print(z_from_python)

        print("q from PLUMED:")
        print(q_from_plumed)

        print("q from Python:")
        print(q_from_python)

        print("K-bias from PLUMED:")
        print(kbias_from_plumed)

        print("K-bias from Python:")
        print(kbias_from_python)

        print(
            "Maximum absolute z error:",
            torch.max(
                torch.abs(
                    z_from_python
                    - z_from_plumed
                )
            ).item(),
        )

        print(
            "Maximum absolute q error:",
            torch.max(
                torch.abs(
                    q_from_python
                    - q_from_plumed
                )
            ).item(),
        )

        print(
            "Maximum absolute K-bias error:",
            torch.max(
                torch.abs(
                    kbias_from_python
                    - kbias_from_plumed
                )
            ).item(),
        )

        check_z = torch.allclose(
            z_from_python,
            z_from_plumed,
            rtol=1e-2,
            atol=1e-5,
        )

        check_q = torch.allclose(
            q_from_python,
            q_from_plumed,
            rtol=1e-2,
            atol=1e-5,
        )

        check_kbias = torch.allclose(
            kbias_from_python,
            kbias_from_plumed,
            rtol=1e-2,
            atol=1e-5,
        )

        if check_z and check_q and check_kbias:
            sys.exit(0)
        else:
            sys.exit(1)


    elif mode == "gnn-exported":
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
        "'gnn-kbias', 'gnn-exported' or "
        "'gnn-kbias-exported'."
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
    atol=1e-5,
)

if check:
    sys.exit(0)
else:
    sys.exit(1)