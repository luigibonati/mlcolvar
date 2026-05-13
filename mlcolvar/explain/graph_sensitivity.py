import numpy as np
from typing import Dict
import torch
import copy

from mlcolvar.data import DictModule
from mlcolvar.utils.plot import pbar
from mlcolvar.core.nn import BaseGNN
from mlcolvar.data.utils import save_dataset_configurations_as_extyz


__all__ = ['graph_node_sensitivity']


def graph_node_sensitivity(model,
                           dataset,
                           component: int = 0,
                           device: str = 'cpu',
                           batch_size: int = None,
                           show_progress: bool = True,
                           extxyz_filename: str = None,
                          ) -> Dict[str, np.ndarray]:
    """Performs a sensitivity analysis on a GNN-based CV model using
    partial derivatives w.r.t. nodes' positions. 
    This allows us to measure which atom is most important to the CV model.

    If system/environment atoms are defined in the input dataset, the average node-sensitivities are returned only
    for the system atom, while a aggregated sensitivities (mean and sum) are returned for environment instead.
    
    Optionally, the results can be printed to an extxyz, where each atom is associated to its sensitivity score
    so that they can be visualized in a molecular viewer.

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
    extxyz_filename: str
        If provided, a extxyz file with this name is printed with the positions of the atoms in the graph dataset 
        with the corresponding sensitivities so that they can be visualized in a molecular viewer.

    Returns
    -------
    results: dictionary
        Results of the sensitivity analysis, containing:
          - 'node_labels': names associated to the nodes of the graphs. If truncated graphs are used, only the system atoms
                           are labeled, while the contribution from the environment atoms is aggregated with mean and sum.
          - 'avg_sensitivities': averaged sensititivities over the given dataset. If truncated graphs are used, environment 
                                 values are aggregated with mean and sum. The results are ordered consistently with the node
                                 labels, the enviroment values are the last two.
          - 'raw_sensitivities': raw sensitivities per-frame, including the sensitivities relative to each atom. If truncated
                                 graphs are used, the sensitivities wrt different environment atoms (whose number may change
                                 for different frame) are returned. The results are ordered as the positions in the dataset.

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

    # make user aware of behaviour for truncated graphs
    if dataset.metadata['is_truncated_graph']:
        print("[NOTE] The input dataset contains truncated graphs for which system and environment selections are provided.\nThe average node-sensitivities" +
             "will be returned only for the system atom, while an aggregated sensitivty is returned for environment instead")

    model = model.to(device)

    # get gradients of cv model on dataset    
    gradients_norm, gradients_components = get_dataset_cv_gradients(model=model,
                                                                    dataset=dataset,
                                                                    component=component,
                                                                    batch_size=batch_size,
                                                                    show_progress=show_progress,
                                                                    progress_prefix='Getting gradients'
                                                                    )
    
    # create a nice dataset with the results
    results = {}
    results['atoms_list'] = [a for a in dataset.metadata['system_atoms_names']]

    # append environment result entries if necessary
    if dataset.metadata['is_truncated_graph']:
        results['atoms_list'].extend(['environment_atoms_mean', 'environment_atoms_sum'])
        
    results['node_labels'] = [str(a) for a in results['atoms_list']]
    results['avg_sensitivities'] = gradients_norm
    results['raw_sensitivities'] = gradients_components
    
    # save an extxyz file with the atoms in the graph and their sensitivity score
    if extxyz_filename is not None:
        updated_dataset = copy.deepcopy(dataset)
        for i, d in enumerate(updated_dataset['data_list']):
            d['sensitivities'] = gradients_components[i]
        save_dataset_configurations_as_extyz(dataset=updated_dataset,
                                             file_name=extxyz_filename,
                                             extra_keys=['sensitivities'])
    
    return results


def get_dataset_cv_values(model,
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
            outputs = model(batchs['data_list'].to(device).to_dict())
            outputs = outputs.cpu().numpy()
            cv_values.append(outputs)

    return np.concatenate(cv_values)


def get_dataset_cv_gradients(model,
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

    # get device
    device = next(model.parameters()).device

    # create a datamodule to initialize the dataloader
    datamodule = DictModule(dataset=dataset,
                            lengths=(1.0,),
                            batch_size=batch_size,
                            random_split=False,
                            shuffle=False
                            )
    datamodule.setup()

    if show_progress:
        items = pbar(
            datamodule.train_dataloader(),
            frequency=0.001,
            prefix=progress_prefix
        )
    else:
        items = datamodule.train_dataloader()


    cv_value_gradients = []
    cv_value_gradients_components = []
    # iterate over the batches
    for batchs in items:
        # get data
        batch_dict = batchs['data_list'].to(device)
        batch_dict['positions'].requires_grad_(True)
        
        # get desired cv component
        cv_values = model(batch_dict)
        cv_values = cv_values[:, component]
        
        # compute gradients
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
        
        if dataset.metadata['is_truncated_graph']:            
            # here we separate the system (constant in number) and environment (possibly changing) gradients 
            # system grads are treated individually, env grads are aggragated with sum and mean
            for i,g in enumerate(gradients):
                # slice the gradients to get the contribution from system and environment atoms
                system_atoms_grads = torch.linalg.vector_norm( g[batch_dict[i]['system_masks'].squeeze()], dim=-1)
                env_atoms_grads    = torch.linalg.vector_norm( g[torch.logical_not(batch_dict[i]['system_masks'].squeeze())], dim=-1)

                # add the values to the result lists
                # one with the components of the gradients of all atoms, ordered as the positions in the dataset
                cv_value_gradients_components.append(torch.linalg.vector_norm( g, dim=-1).unsqueeze(-1))    

                # one with the norm of the gradients for each system atom and aggregated env atoms
                cv_value_gradients.extend(torch.concat([system_atoms_grads,
                                                        env_atoms_grads.mean(dim=-1, keepdim=True), 
                                                        env_atoms_grads.sum(dim=-1, keepdim=True)
                                                        ]).unsqueeze(0).cpu().numpy()
                                        )
        else:
            # here we ensure that all the gradients have the correct shape 
            # and that each entry is at the correct index accordingly
            max_used_atoms = len(dataset.metadata['system_idx'])
            for i,g in enumerate(gradients):
                aux = torch.zeros((max_used_atoms, 3))
                # this populates the right entries according to the orignal indexing
                aux[batch_dict[i]['system_names_idx'], :] = g
                aux = aux.unsqueeze(0)
                cv_value_gradients.extend(torch.linalg.vector_norm(aux, dim=-1).cpu().numpy())
                cv_value_gradients_components.extend(aux.cpu().numpy())
                
        return cv_value_gradients, cv_value_gradients_components



def test_get_cv_values_graph():
    import lightning
    from mlcolvar.cvs import DeepTDA
    from mlcolvar.core.nn.graph import SchNetModel
    from mlcolvar.data.graph.utils import create_test_graph_input

    # create data, we need the dataset for sensitivity analysis later
    dataset = create_test_graph_input(output_type='dataset', n_samples=50, n_states=2, n_atoms=3)
    datamodule = DictModule(dataset=dataset, lengths=[0.8, 0.2], shuffle=[1, 0])

    # create model
    gnn_model = SchNetModel(n_out=1, cutoff=0.1, atomic_numbers=[8, 1])
    model = DeepTDA(
        n_states=2,
        n_cvs=1,
        target_centers=[-5, 5],
        target_sigmas=[0.2, 0.2],
        model=gnn_model
    )

    # train model
    trainer = lightning.Trainer(
        accelerator="cpu", max_epochs=2, logger=False, enable_checkpointing=False, enable_model_summary=False
    )
    trainer.fit(model, datamodule)

    # do analysis
    cv_values = get_dataset_cv_values(model=model, dataset=dataset, batch_size=0)

    # print results
    print(cv_values)

    assert (torch.allclose(model(dataset.get_graph_inputs()), torch.Tensor(cv_values)))



def test_graph_sensitivity():
    import lightning
    from mlcolvar.cvs import DeepTDA
    from mlcolvar.core.nn.graph import SchNetModel
    from mlcolvar.data.graph.utils import create_test_graph_input

    for environment in [False, True]:
        # create data, we need the dataset for sensitivity analysis later
        dataset = create_test_graph_input(output_type='dataset', n_samples=100, n_states=2, n_atoms=3, environment=environment)
        datamodule = DictModule(dataset=dataset, lengths=[0.8, 0.2], shuffle=[1, 0])

        # create model
        gnn_model = SchNetModel(n_out=1, cutoff=0.1, atomic_numbers=[8, 1])
        model = DeepTDA(
            n_states=2,
            n_cvs=1,
            target_centers=[-5, 5],
            target_sigmas=[0.2, 0.2],
            model=gnn_model
        )

        # train model
        trainer = lightning.Trainer(
            accelerator="cpu", max_epochs=2, logger=False, enable_checkpointing=False, enable_model_summary=False
        )
        trainer.fit(model, datamodule)

        # do analysis
        test_sensitivity = graph_node_sensitivity(model=model,
                                        dataset=dataset,
                                        batch_size=0)

        # print results
        print(test_sensitivity)
