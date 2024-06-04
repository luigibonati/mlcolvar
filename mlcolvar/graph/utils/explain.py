import torch
import numpy as np
from typing import List, Dict

from .progress import pbar
from mlcolvar.graph import data as gdata
from mlcolvar.graph import cvs as gcvs

"""
Metaphysics.
"""

__all__ = ['graph_node_sensitivity_analysis']


def graph_node_sensitivity_analysis(
    model: gcvs.GraphBaseCV,
    dataset: gdata.GraphDataSet,
    node_indices: List[int] = None,
    per_class: bool = False,
    device: str = 'cpu',
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Perform a sensitivity analysis by masking nodes. This allows us to measure
    which atom is most important to the CV.

    See also
    --------
    mlcolvar.utils.explain.sensitivity_analysis
        Perform the sensitivity analysis of a feedforward model.

    Parameters
    ----------
    model: mlcolvar.graph.cvs.GraphBaseCV
        Collective variable model.
    dataset: mlcovar.graph.data.GraphDataSet
        Dataset on which to compute the sensitivity analysis.
    node_indices: List[int]
        Indices of the nodes to analysis.
    per_class: bool
        If the dataset has labels, compute also the sensitivity per class.
    device: str
        Name of the device.
    show_progress: bool
        If show the progress bar.

    Returns
    -------
    results: dictionary
        Results of the sensitivity analysis, containing 'node_indices' and the
        'sensitivity', ordered according to the node indices.
    """
    if node_indices is None:
        n_nodes = [len(d.positions) for d in dataset]
        node_indices = list(range(max(n_nodes)))

    sensitivities = []
    sensitivities_component = []
    model = model.to(device)

    for node, i in enumerate(node_indices):

        node_results = []
        n_nodes = len(node_indices)
        if show_progress:
            items = pbar(
                dataset,
                frequency=0.0001,
                prefix='Node {:d}/{:d}'.format((i + 1), n_nodes)
            )
        else:
            items = dataset

        with torch.no_grad():
            for data in items:
                data = data.to(device)
                data = data.to_dict()
                batch_id = torch.tensor(
                    [0] * len(data['positions']), dtype=torch.long
                ).to(device)
                data['batch'] = batch_id

                cv_org = model(data)

                mask = data['edge_index'] != node
                mask = mask[0] & mask[1]
                shifts_masked = data['shifts'][mask.T]
                edge_index_masked = data['edge_index'][:, mask]
                data['shifts'] = shifts_masked
                data['edge_index'] = edge_index_masked

                cv_masked = model(data)

                delta = (cv_org - cv_masked).cpu().numpy().tolist()
                node_results.extend(delta)

        sensitivities_component.append(np.abs(node_results))
        sensitivities.append(np.mean(np.abs(node_results), axis=0))

    results = {}
    results['node_indices'] = node_indices
    results['sensitivities'] = np.array(sensitivities)
    results['sensitivities_components'] = np.array(
        sensitivities_component
    ).transpose(2, 1, 0)[0]

    return results
