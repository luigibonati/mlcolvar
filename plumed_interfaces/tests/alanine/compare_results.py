import torch
import sys
from mlcolvar.io import create_dataset_from_files, create_dataset_from_trajectories, load_dataframe

# get arguments
mode = sys.argv[1]

# load model
model = torch.jit.load("model.pt")

if mode in ['descriptors', 'descriptors-kbias']:
    
    # load the COLVAR from the driver
    filename = 'COLVAR'         

    # load dataset
    dataset, colvar = create_dataset_from_files(filename,
                                                filter_args={'regex':'x' }, # select distances between heavy atoms
                                                return_dataframe=True, 
                                                )

    # get the cv values
    if mode == "descriptors":
        cv_from_plumed = torch.Tensor(colvar['model.node-0'].values)
    if mode == 'descriptors-kbias':
        cv_from_plumed = torch.Tensor(colvar['model.z'].values)
    
    cv_from_python = model(dataset['data']).squeeze()

elif mode in ["gnn", "gnn-kbias"]: 
    filename = "COLVAR"
    trajectory = "traj_comp.xtc"
    topology = "ref.pdb"

    # load dataset
    dataset = create_dataset_from_trajectories(trajectories=trajectory,
                                               topologies=topology, # with xyz use none                 
                                               cutoff=10.0,
                                               system_selection='all and not type H',
                                               lengths_conversion=10.0,
                                               )
    
    # load the COLVAR from the driver
    colvar = load_dataframe(filename)

    # get the cv values
    if mode == "gnn":
        cv_from_plumed = torch.Tensor(colvar['model.node-0'].values)
    if mode == 'gnn-kbias':
        cv_from_plumed = torch.Tensor(colvar['model.z'].values)
    
    cv_from_python = model(dataset.get_graph_inputs()).squeeze()
    
    print(cv_from_plumed)
    print(cv_from_python)


# check that the results are consistent
check = torch.allclose(cv_from_python, 
                       cv_from_plumed, 
                       rtol=1e-2)

if check:
    sys.exit(0)
else:
    sys.exit(1)
    

