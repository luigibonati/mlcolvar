import os
from typing import List, Union

from mlcolvar.io._utils import _download_temp_file
from mlcolvar.io.graphs.ase_ import create_pdb_from_xyz, load_traj_with_ase
from mlcolvar.io.graphs.mdtraj_ import load_traj_with_mdtraj


__all__ = []


def _load_trajectories(
    trajectories: Union[List[str], str],
    topologies: Union[List[str], str, None] = None,
    load_args: List[dict] = None,
    folder: str = None,
    delete_download: bool = True,
    backend: str = "mdtraj",
):
    """Load trajectory files using MDTraj or ASE.

    Parameters
    ----------
    trajectories : str or list[str]
        Path or paths to trajectory files.
    topologies : str or list[str], optional
        Topology file or files used by the MDTraj backend. A single topology
        can be shared across multiple trajectories.
    load_args : list[dict], optional
        Per-file loading options containing ``start``, ``stop``, and
        ``stride``.
    folder : str, optional
        Common directory containing trajectory and topology files.
    delete_download : bool, optional
        Whether temporary downloaded files are deleted after loading.
    backend : {"mdtraj", "ase"}, optional
        Backend used to load trajectory files.

    Returns
    -------
    list
        Loaded trajectory objects.
    """
    if backend not in {"mdtraj", "ase"}:
        raise ValueError(
            f"Unknown backend {backend!r}. Expected 'mdtraj' or 'ase'."
        )

    if isinstance(trajectories, str):
        trajectories = [trajectories]
    elif isinstance(trajectories, list):
        trajectories = list(trajectories)
    else:
        raise TypeError(
            "`trajectories` must be a string or a list of strings."
        )

    if not all(isinstance(trajectory, str) for trajectory in trajectories):
        raise TypeError(
            "All entries in `trajectories` must be strings."
        )

    if load_args is not None:
        if not isinstance(load_args, list) or len(load_args) != len(trajectories):
            raise ValueError(
                "`load_args` must contain one dictionary per trajectory."
            )

        if not all(isinstance(args, dict) for args in load_args):
            raise TypeError(
                "All entries in `load_args` must be dictionaries."
            )

    shared_topology = False

    if backend == "ase":
        if topologies is not None:
            raise ValueError(
                "`topologies` must be None when using the ASE backend."
            )

        topologies = ["" for _ in trajectories]

    else:
        if isinstance(topologies, str):
            topologies = [topologies for _ in trajectories]
            shared_topology = True

        elif topologies is None:
            topologies = ["" for _ in trajectories]

        elif isinstance(topologies, list):
            topologies = list(topologies)

            if len(topologies) == 0:
                topologies = ["" for _ in trajectories]

            elif len(topologies) == 1:
                topologies = topologies * len(trajectories)
                shared_topology = True

            elif len(topologies) != len(trajectories):
                raise ValueError(
                    "Provide either one topology or one topology "
                    "per trajectory."
                )

        else:
            raise TypeError(
                "`topologies` must be a string, a list of strings, or None."
            )

        if not all(isinstance(topology, str) for topology in topologies):
            raise TypeError(
                "All entries in `topologies` must be strings."
            )

    def is_url(path):
        return path.startswith(("http://", "https://"))

    if folder is not None:
        trajectories = [
            trajectory
            if is_url(trajectory)
            else os.path.join(folder, trajectory)
            for trajectory in trajectories
        ]

        topologies = [
            topology
            if not topology or is_url(topology)
            else os.path.join(folder, topology)
            for topology in topologies
        ]

    shared_temp_top = None
    shared_top_url = None

    if (
        backend == "mdtraj"
        and shared_topology
        and topologies[0]
        and is_url(topologies[0])
    ):
        shared_top_url = topologies[0]
        shared_temp_top, shared_top_path = _download_temp_file(
            file_url=shared_top_url,
            delete_download=delete_download,
            append_suffix=True,
            return_name=True,
        )
        topologies = [shared_top_path for _ in trajectories]

    loaded_trajectories = []

    try:
        for i, trajectory in enumerate(trajectories):
            topology = topologies[i]
            temp_traj = None
            temp_top = None
            url_traj = None
            url_top = None

            try:
                if is_url(trajectory):
                    url_traj = trajectory
                    temp_traj, trajectory = _download_temp_file(
                        file_url=url_traj,
                        delete_download=delete_download,
                        append_suffix=True,
                        return_name=True,
                    )

                if topology and is_url(topology):
                    url_top = topology
                    temp_top, topology = _download_temp_file(
                        file_url=url_top,
                        delete_download=delete_download,
                        append_suffix=True,
                        return_name=True,
                    )

                if backend == "mdtraj":
                    _, ext = os.path.splitext(trajectory)

                    if ext.lower() == ".xyz" and not topology:
                        pdb_file = trajectory.replace(ext, "_top.pdb")
                        topology = create_pdb_from_xyz(
                            trajectory,
                            pdb_file,
                        )

                args = load_args[i] if load_args is not None else {}
                start = args.get("start", 0)
                stop = args.get("stop")
                stride = args.get("stride", 1)

                if backend == "mdtraj":
                    traj = load_traj_with_mdtraj(
                        trajectory=trajectory,
                        topology=topology,
                        start=start,
                        stop=stop,
                        stride=stride,
                    )
                else:
                    traj = load_traj_with_ase(
                        trajectory=trajectory,
                        start=start,
                        stop=stop,
                        stride=stride,
                    )

                loaded_trajectories.append(traj)

            finally:
                if temp_traj is not None:
                    if delete_download:
                        temp_traj.close()
                    else:
                        print(
                            f"downloaded file ({url_traj}) "
                            f"saved as ({trajectory})."
                        )

                if temp_top is not None:
                    if delete_download:
                        temp_top.close()
                    else:
                        print(
                            f"downloaded file ({url_top}) "
                            f"saved as ({topology})."
                        )

    finally:
        if shared_temp_top is not None:
            if delete_download:
                shared_temp_top.close()
            else:
                print(
                    f"downloaded file ({shared_top_url}) "
                    f"saved as ({topologies[0]})."
                )

    return loaded_trajectories