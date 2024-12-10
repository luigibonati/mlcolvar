import torch
import numpy as np

from mlcolvar.graph import cvs as gcvs
from mlcolvar.graph import data as gdata
from mlcolvar.graph import utils as gutils

__all__ = ['get_dataset_cv_values', 'get_dataset_cv_gradients']

"""
Analysis utils.
"""


def get_dataset_cv_values(
    model: gcvs.GraphBaseCV,
    dataset: gdata.GraphDataSet,
    batch_size: int = None,
    show_progress: bool = True,
    progress_prefix: str = 'Calculating CV values'
) -> np.ndarray:
    """
    Get CV values of a given dataset. The calculation will run on the device
    where the model is on.

    Parameters
    ----------
    model: mlcolvar.graph.cvs.GraphBaseCV
        Collective variable model.
    dataset: mlcovar.graph.data.GraphDataSet
        Dataset on which to compute the sensitivity analysis.
    batch_size:
        Batch size used for evaluating the CV.
    show_progress: bool
        If show the progress bar.
    """
    datamodule = gdata.GraphDataModule(
        dataset,
        lengths=(1.0,),
        batch_size=batch_size,
        random_split=False,
        shuffle=False
    )
    datamodule.setup()

    cv_values = []
    device = next(model.parameters()).device

    if show_progress:
        items = gutils.progress.pbar(
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
    model: gcvs.GraphBaseCV,
    dataset: gdata.GraphDataSet,
    component: int = 0,
    batch_size: int = None,
    show_progress: bool = True,
    progress_prefix: str = 'Calculating CV gradients'
) -> np.ndarray:
    """
    Get gradients of the CV w.r.t. node positions in a given dataset. The
    calculation will run on the device where the model is on.

    Parameters
    ----------
    model: mlcolvar.graph.cvs.GraphBaseCV
        Collective variable model.
    dataset: mlcovar.graph.data.GraphDataSet
        Dataset on which to compute the sensitivity analysis.
    component: int
        Component of the CV to analysis.
    batch_size:
        Batch size used for evaluating the CV.
    show_progress: bool
        If show the progress bar.
    """
    datamodule = gdata.GraphDataModule(
        dataset,
        lengths=(1.0,),
        batch_size=batch_size,
        random_split=False,
        shuffle=False
    )
    datamodule.setup()

    cv_value_gradients = []
    device = next(model.parameters()).device

    if show_progress:
        items = gutils.progress.pbar(
            datamodule.train_dataloader(),
            frequency=0.001,
            prefix=progress_prefix
        )
    else:
        items = datamodule.train_dataloader()

    for batchs in items:
        batch_dict = batchs.to(device).to_dict()
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
        gradients = torch.split(
            gradients[0].detach(), graph_sizes.cpu().numpy().tolist()
        )
        gradients = [g.cpu().numpy() for g in gradients]
        cv_value_gradients.extend(gradients)

    return np.array(cv_value_gradients)
