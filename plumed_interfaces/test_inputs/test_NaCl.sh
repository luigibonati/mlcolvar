#!/bin/bash

# retrieve mode (gnn, gnn-kbias)
mode=$1
export OMP_NUM_THREADS=8


# source programs
source /home/etrizio@iit.local/Bin/lammps-stable_22Jul2025_update1_all/sourceme.sh
source /home/etrizio@iit.local/Bin/dev/plumed2-2.10b/sourceme.sh

# define python path with mdraj
PYTHON_PATH="/home/etrizio@iit.local/Bin/miniconda3/envs/graph_mlcolvar_test_2.5/bin/python"



FOLDER_NAME="test_run_NaCl" 
rm -r $FOLDER_NAME
echo folder $FOLDER_NAME

# copy template folder and move inside
cp -r ../plumed_interfaces/test_inputs/NaCl/gnn_based $FOLDER_NAME
cd $FOLDER_NAME

if [ $mode == "gnn" ]; then
    # use standard interface and input file
    cp ../../plumed_interfaces/PytorchModelGNN.cpp .
    mv plumed_PytorchModelGNN.dat plumed.dat

elif [ $mode == "gnn-kbias" ]; then
    # use kbias interface and input file
    cp ../../plumed_interfaces/PytorchKolmogorovBiasGNN.cpp .
    mv plumed_PytorchKolmogorovBiasGNN.dat plumed.dat

else
    echo "Invalid mode. Use 'gnn' or 'gnn-kbias'."
    exit 1
fi

# update phyton path
sed -i "s|PYTHON_BIN=/path/to/python/with/mdtraj|PYTHON_BIN=$PYTHON_PATH|g" plumed.dat

# run simulation
lmp -i input.lmp -l log.lammps &

# return to original folder
cd ..
cd ..