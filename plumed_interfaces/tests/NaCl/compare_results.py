import torch
import sys
from mlcolvar.io import create_dataset_from_files, create_dataset_from_trajectories, load_dataframe

# get arguments
mode = sys.argv[1]

# load model
model = torch.jit.load("model.pt")

if mode in ["gnn", "gnn-kbias", "gnn-lr", "gnn-lr-kbias"]:
    
    # check if long range
    if mode in ["gnn-lr", "gnn-lr-kbias"]:
        long_range = True
    elif mode in ["gnn", "gnn-kbias"]:
        long_range = False

    filename = "COLVAR"
    trajectory = "traj.xtc"
    topology = "NaCl.pdb"

    # load dataset
    # load dataset
    dataset = create_dataset_from_trajectories(trajectories=trajectory,
                                               topologies=topology, 
                                               cutoff=4.0,  
                                               buffer=3.0,
                                               system_selection='type Na or type Cl',
                                               environment_selection='type O',
                                               subsystem_selection='type Na or type Cl' if long_range else None,
                                               long_range_cutoff=10 if long_range else -1,
                                               lengths_conversion=10.0,
                                               )
    
    # load the COLVAR from the driver
    colvar = load_dataframe(filename)

    # get the cv values
    if mode in ["gnn", "gnn-lr"]:
        cv_from_plumed = torch.Tensor(colvar['model.node-0'].values)
    if mode in ["gnn-kbias", "gnn-lr-kbias"]:
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
    

