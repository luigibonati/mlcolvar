import numpy as np
from typing import Dict
import torch

from mlcolvar.data import DictModule
from mlcolvar.utils.plot import pbar
from mlcolvar.core.nn import BaseGNN


__all__ = ['graph_node_sensitivity']


def graph_node_sensitivity(
    model,
    dataset,
    component: int = 0,
    device: str = 'cpu',
    batch_size: int = None,
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """Performs a sensitivity analysis on a GNN-based CV model by calculating 
    the CV gradient w.r.t. nodes' positions. 
    This allows us to measure which atom is most important to the CV model.

    Parameters
    ----------
    model: mlcolvar.cvs.BaseCV
        Collective variable model based on GNN
    dataset: mlcovar.data.DictDataset
        Graph-based dataset on which to compute the sensitivity analysis
    device: str
        Name of the device on which to perform the computation
    batch_size:
        Batch size used for evaluating the CV
    show_progress: bool
        If to show the progress bar

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
    # check model is GNN-based
    if not isinstance(model.nn, BaseGNN):
        raise ValueError (
                "The CV model is not based on GNN! Maybe you should use the feedforward sensitivity_analysis from  mlcolvar.utils.explain.sensitivity!"
            )

    model = model.to(device)

    gradients = get_dataset_cv_gradients(
        model=model,
        dataset=dataset,
        component=component,
        batch_size=batch_size,
        show_progress=show_progress,
        progress_prefix='Getting gradients'
    )
    sensitivities_components = np.linalg.norm(gradients, axis=-1)

    results = {}
    results['atoms_list'] = np.array(dataset.metadata['used_names'])
    results['node_labels'] = [str(a) for a in results['atoms_list']]
    results['node_labels_components'] = np.array([np.array(dataset.metadata['used_names'])[dataset[i]['data_list']['names_idx']] for i in range(len(dataset))])        
    results['sensitivities'] = sensitivities_components.mean(axis=0)
    results['sensitivities_components'] = sensitivities_components

    return results

def get_dataset_cv_values(
    model,
    dataset,
    batch_size: int = None,
    show_progress: bool = True,
    progress_prefix: str = 'Calculating CV values'
) -> np.ndarray:
    """Gets the values of a CV model on a given dataset. 
    The calculation will run on the device where the model is on.

    Parameters
    ----------
    model: mlcolvar.cvs.BaseCV
        Collective variable model
    dataset: mlcovar.data.DictDataset
        Dataset on which to compute the sensitivity analysis
    batch_size:
        Batch size used for evaluating the CV
    show_progress: bool
        If to show the progress bar
    """
    datamodule = DictModule(
        dataset=dataset,
        lengths=(1.0,),
        batch_size=batch_size,
        random_split=False,
        shuffle=False
    )
    datamodule.setup()

    cv_values = []
    device = next(model.parameters()).device

    if show_progress:
        items = pbar(
            datamodule.train_dataloader(),
            frequency=0.001,
            prefix=progress_prefix
        )
    else:
        items = datamodule.train_dataloader()

    with torch.no_grad():
        for batchs in items:
            outputs = model(batchs.to(device).to_dict())
            outputs = outputs.cpu().numpy()
            cv_values.append(outputs)

    return np.concatenate(cv_values)


def get_dataset_cv_gradients(
    model,
    dataset,
    component: int = 0,
    batch_size: int = None,
    show_progress: bool = True,
    progress_prefix: str = 'Calculating CV gradients'
) -> np.ndarray:
    """Get gradients of a GNN-based CV w.r.t. node positions in a given dataset. 
    The calculation will run on the device where the model is on.

    Parameters
    ----------
    model: mlcolvar.cvs.BaseCV
        Collective variable model based on GNN
    dataset: mlcovar.data.DictDataset
        Graph-based dataset on which to compute the sensitivity analysis
    component: int
        Component of the CV to analyse
    batch_size:
        Batch size used for evaluating the CV
    show_progress: bool
        If to show the progress bar
    """
    datamodule = DictModule(
        dataset=dataset,
        lengths=(1.0,),
        batch_size=batch_size,
        random_split=False,
        shuffle=False
    )
    datamodule.setup()

    cv_value_gradients = []
    device = next(model.parameters()).device

    if show_progress:
        items = pbar(
            datamodule.train_dataloader(),
            frequency=0.001,
            prefix=progress_prefix
        )
    else:
        items = datamodule.train_dataloader()

    for batchs in items:
        batch_dict = batchs['data_list'].to(device)
        batch_dict['positions'].requires_grad_(True)
        cv_values = model(batch_dict)
        cv_values = cv_values[:, component]
        grad_outputs = [torch.ones_like(cv_values, device=device)]
        gradients = torch.autograd.grad(
            outputs=[cv_values],
            inputs=[batch_dict['positions']],
            grad_outputs=grad_outputs,
            retain_graph=False,
            create_graph=False,
        )
        graph_sizes = batch_dict['ptr'][1:] - batch_dict['ptr'][:-1]
        
        # if we used the removed isolated atoms this will give an inhomogenous tensor!
        gradients = torch.split(
            gradients[0].detach(), graph_sizes.cpu().numpy().tolist()
        )
        
        # here we ensure that all the gradients have the correct shape 
        # and that each entry is at the correct index accordingly
        max_used_atoms = len(dataset.metadata['used_idx'])
        for i,g in enumerate(gradients):
            aux = torch.zeros((max_used_atoms, 3))
            # this populates the right entries according to the orignal indexing
            aux[batch_dict[i]['names_idx'], :] = g
            cv_value_gradients.extend(aux.unsqueeze(0).cpu().numpy())

    return np.array(cv_value_gradients)