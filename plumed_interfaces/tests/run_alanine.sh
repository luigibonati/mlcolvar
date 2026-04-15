#!/bin/bash

# retrieve mode (descriptors, descriptors-kbias, gnn, gnn-kbias)
mode=$1

# =====================================================================================
# ======================================= SETUP =======================================
# =====================================================================================

# run parameters
export OMP_NUM_THREADS=2
NSTEPS=$[500*20] #last is ps

# define paths to sourceme.sh files for gromacs and plumed
#these need to be edited by the user before running the script
GROMACS_SOURCE="/path/to/gromacs/sourceme.sh"
PLUMED_SOURCE="/path/to/plumed/sourceme.sh"

# define python path with mdraj, only for mode gnn and gnn-kbias
PYTHON_PATH="/path/to/python/with/mdtraj"

# =====================================================================================
# ======================================= CHECKS ======================================
# =====================================================================================

# try to source gromacs and plumed, if not found print error message and exit
if ! source $GROMACS_SOURCE 2>/dev/null; then
    echo "GROMACS sourceme.sh file could not be found. Please edit the script to source GROMACS before running it."
    exit 1
fi
if ! source $PLUMED_SOURCE 2>/dev/null; then
    echo "PLUMED sourceme.sh file could not be found. Please edit the script to source PLUMED before running it."
    exit 1
fi


# check that gromacs and plumed are sourced and python path is set if needed
if ! command -v gmx &> /dev/null; then
    echo "GROMACS executable not working, please check!"
    exit 1
fi  
if ! command -v plumed &> /dev/null; then
    echo "PLUMED executable not working, please check!"
    exit 1
fi
if [ "$mode" = "gnn" ] || [ "$mode" = "gnn-kbias" ]; then
    if ! command -v $PYTHON_PATH &> /dev/null; then
        echo "Python could not be found. Please edit the script to set the PYTHON_PATH variable to a python executable with mdtraj installed."
        exit 1
    fi
    if ! $PYTHON_PATH -c "import mdtraj" 2>/dev/null; then
        echo "mdtraj is not installed in the Python environment."
        exit 1
    fi
fi

# =====================================================================================
# ====================================== PREPARE ======================================
# =====================================================================================

# create run folder
FOLDER_NAME="test_run_alanine" 
rm -r $FOLDER_NAME
echo folder $FOLDER_NAME

# copy template folder and move inside
if [ "$mode" = "descriptors" ] || [ "$mode" = "descriptors-kbias" ]; then
    cp -r ../plumed_interfaces/tests/alanine/descriptor_based $FOLDER_NAME
elif [ "$mode" = "gnn" ] || [ "$mode" = "gnn-kbias" ]; then
    cp -r ../plumed_interfaces/tests/alanine/gnn_based $FOLDER_NAME
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

# remove useless input files
rm plumed_*

# update python path
sed -i "s|PYTHON_BIN=/path/to/python/with/mdtraj|PYTHON_BIN=$PYTHON_PATH|g" plumed.dat

# =====================================================================================
# ======================================== RUN ========================================
# =====================================================================================

# run simulation
gmx mdrun -s stateA.tpr -nsteps $NSTEPS -cpi state.cpt -plumed plumed.dat -gpu_id 0 -ntmpi 1 -pin on &

# return to original folder
cd ..
cd ..