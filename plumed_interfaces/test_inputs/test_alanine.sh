#!/bin/bash

# retrieve mode (descriptors, descriptors-kbias, gnn, gnn-kbias)
mode=$1
export OMP_NUM_THREADS=2
NSTEPS=$[500*20] #last is ps


# source programs
source ~/Bin/gromacs-2024.5/sourceme.sh
source /home/etrizio@iit.local/Bin/dev/plumed2-2.10b/sourceme.sh


# define python path with mdraj
PYTHON_PATH="/home/etrizio@iit.local/Bin/miniconda3/envs/graph_mlcolvar_test_2.5/bin/python"



FOLDER_NAME="test_run_alanine" 
rm -r $FOLDER_NAME
echo folder $FOLDER_NAME

# copy template folder and move inside
if [ "$mode" = "descriptors" ] || [ "$mode" = "descriptors-kbias" ]; then
    cp -r ../plumed_interfaces/test_inputs/alanine/descriptor_based $FOLDER_NAME
elif [ "$mode" = "gnn" ] || [ "$mode" = "gnn-kbias" ]; then
    cp -r ../plumed_interfaces/test_inputs/alanine/gnn_based $FOLDER_NAME
else
    echo "Invalid mode. Use 'descriptors', 'descriptors-kbias', 'gnn' or 'gnn-kbias'."
    exit 1
fi
cd $FOLDER_NAME


if [ $mode = "descriptors" ]; then
    # use standard interface and input file
    cp ../../plumed_interfaces/PytorchModel.cpp .
    mv plumed_PytorchModel.dat plumed.dat

elif [ $mode = "descriptors-kbias" ]; then
    # use kbias interface and input file
    cp ../../plumed_interfaces/PytorchKolmogorovBias.cpp .
    mv plumed_PytorchKolmogorovBias.dat plumed.dat

elif [ $mode = "gnn" ]; then
    # use standard interface and input file
    cp ../../plumed_interfaces/PytorchModelGNN.cpp .
    mv plumed_PytorchModelGNN.dat plumed.dat

elif [ $mode = "gnn-kbias" ]; then
    # use kbias interface and input file
    cp ../../plumed_interfaces/PytorchKolmogorovBiasGNN.cpp .
    mv plumed_PytorchKolmogorovBiasGNN.dat plumed.dat
fi

# update phyton path
sed -i "s|PYTHON_BIN=/path/to/python/with/mdtraj|PYTHON_BIN=$PYTHON_PATH|g" plumed.dat

# run simulation
gmx mdrun -s stateA.tpr -nsteps $NSTEPS -cpi state.cpt -plumed plumed.dat -gpu_id 0 -ntmpi 1 -pin on &


# return to original folder
cd ..
cd ..