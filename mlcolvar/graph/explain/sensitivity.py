import copy
import numpy as np
from typing import List, Dict

from mlcolvar.graph import cvs as gcvs
from mlcolvar.graph import data as gdata
from mlcolvar.graph import utils as gutils

from .utils import get_dataset_cv_values

"""
Sensitivity analysis.
"""

__all__ = ['graph_node_sensitivity']


def graph_node_sensitivity(
    model: gcvs.GraphBaseCV,
    dataset: gdata.GraphDataSet,
    node_indices: List[int] = None,
    device: str = 'cpu',
    batch_size: int = None,
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Perform a sensitivity analysis by masking nodes. This allows us to measure
    which atom is most important to the CV.

    Parameters
    ----------
    model: mlcolvar.graph.cvs.GraphBaseCV
        Collective variable model.
    dataset: mlcovar.graph.data.GraphDataSet
        Dataset on which to compute the sensitivity analysis.
    node_indices: List[int]
        Indices of the nodes to analysis.
    device: str
        Name of the device.
    batch_size:
        Batch size used for evaluating the CV.
    show_progress: bool
        If show the progress bar.

    Returns
    -------
    results: dictionary
        Results of the sensitivity analysis, containing 'node_indices',
        'sensitivities', and 'sensitivities_components', ordered according to
        the node indices.

    See also
    --------
    mlcolvar.utils.explain.sensitivity_analysis
        Perform the sensitivity analysis of a feedforward model.
    """
    if node_indices is None:
        n_nodes = [len(d.positions) for d in dataset]
        node_indices = list(range(max(n_nodes)))
    else:
        node_indices = [i for i in node_indices]
    n_nodes = len(node_indices)

    if show_progress:
        items = gutils.progress.pbar(
            node_indices, frequency=0.0001, prefix='Sensitivity'
        )
    else:
        items = node_indices

    sensitivities = []
    sensitivities_components = []
    dataset_clone = copy.deepcopy(dataset)
    model = model.to(device)

    cv_org = get_dataset_cv_values(
        model, dataset, batch_size, show_progress, 'Getting base data'
    )

    for node in items:

        for j in range(len(dataset_clone)):
            mask = dataset[j]['edge_index'] != node
            mask = mask[0] & mask[1]
            shifts_masked = dataset[j]['shifts'][mask.transpose(-1, 0)]
            edge_index_masked = dataset[j]['edge_index'][:, mask]
            dataset_clone[j]['shifts'] = shifts_masked
            dataset_clone[j]['edge_index'] = edge_index_masked

        cv_masked = get_dataset_cv_values(
            model, dataset_clone, batch_size, False,
        )

        delta = cv_org - cv_masked
        sensitivities_components.append(np.abs(delta))
        sensitivities.append(np.mean(np.abs(delta), axis=0))

    results = {}
    results['node_indices'] = node_indices
    results['sensitivities'] = np.array(sensitivities)
    results['sensitivities_components'] = np.array(
        sensitivities_components
    ).transpose(2, 1, 0)[0]

    return results