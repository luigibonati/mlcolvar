import copy
import torch
import numpy as np
from typing import List, Dict

from mlcolvar.graph import cvs as gcvs
from mlcolvar.graph import data as gdata
from mlcolvar.graph import utils as gutils

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
    device: str
        Name of the device.
    batch_size:
        Batch size used for evaluating the CV.
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
    n_nodes = len(node_indices)
    if show_progress:
        node_indices = gutils.progress.pbar(
            node_indices, frequency=0.0001, prefix='Sensitivity'
        )

    sensitivities = []
    sensitivities_component = []
    dataset_clone = copy.deepcopy(dataset)
    model = model.to(device)

    for node, i in enumerate(node_indices):

        node_results = []

        for j in range(len(dataset_clone)):
            mask = dataset[j]['edge_index'] != node
            mask = mask[0] & mask[1]
            shifts_masked = dataset[j]['shifts'][mask.T]
            edge_index_masked = dataset[j]['edge_index'][:, mask]
            dataset_clone[j]['shifts'] = shifts_masked
            dataset_clone[j]['edge_index'] = edge_index_masked

        datamodule_org = gdata.GraphDataModule(
            dataset,
            lengths=(1.0,),
            batch_size=batch_size,
            random_split=False,
            shuffle=False
        )
        datamodule_org.setup()

        datamodule_masked = gdata.GraphDataModule(
            dataset_clone,
            lengths=(1.0,),
            batch_size=batch_size,
            random_split=False,
            shuffle=False
        )
        datamodule_masked.setup()

        loaders = zip(
            datamodule_org.train_dataloader(),
            datamodule_masked.train_dataloader()
        )

        with torch.no_grad():
            for batchs in loaders:
                cv_org = model(batchs[0].to(device).to_dict())
                cv_masked = model(batchs[1].to(device).to_dict())

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
