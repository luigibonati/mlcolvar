#!/bin/bash

# retrieve mode (gnn, gnn-kbias)
mode=$1


# =====================================================================================
# ======================================= SETUP =======================================
# =====================================================================================

# run parameters
export OMP_NUM_THREADS=8

# define paths to sourceme.sh files for lammps and plumed
#these need to be edited by the user before running the script
LAMMPS_SOURCE="/path/to/lammps/sourceme.sh"
PLUMED_SOURCE="/path/to/plumed/sourceme.sh"

# define python path with mdraj
PYTHON_PATH="/path/to/python/with/mdtraj"

# =====================================================================================
# ======================================= CHECKS ======================================
# =====================================================================================

# try to source lammps and plumed, if not found print error message and exit
if ! source $LAMMPS_SOURCE 2>/dev/null; then
    echo "LAMMPS sourceme.sh file could not be found. Please edit the script to source LAMMPS before running it."
    exit 1
fi
if ! source $PLUMED_SOURCE 2>/dev/null; then
    echo "PLUMED sourceme.sh file could not be found. Please edit the script to source PLUMED before running it."
    exit 1
fi


# check that lammps and plumed are sourced and python path is set if needed
if ! command -v lmp &> /dev/null; then
    echo "LAMMPS executable not working, please check!"
    exit 1
fi  
if ! command -v plumed &> /dev/null; then
    echo "PLUMED executable not working, please check!"
    exit 1
fi

if ! command -v $PYTHON_PATH &> /dev/null; then
    echo "Python could not be found. Please edit the script to set the PYTHON_PATH variable to a python executable with mdtraj installed."
    exit 1
fi
if ! $PYTHON_PATH -c "import mdtraj" 2>/dev/null; then
    echo "mdtraj is not installed in the Python environment."
    exit 1
fi

# =====================================================================================
# ====================================== PREPARE ======================================
# =====================================================================================

FOLDER_NAME="test_run_NaCl" 
rm -r $FOLDER_NAME
echo folder $FOLDER_NAME

# copy template folder and move inside
cp -r ../plumed_interfaces/tests/NaCl/gnn_based $FOLDER_NAME
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

# remove useless input files
rm plumed_*

# update python path
sed -i "s|PYTHON_BIN=/path/to/python/with/mdtraj|PYTHON_BIN=$PYTHON_PATH|g" plumed.dat

# =====================================================================================
# ======================================== RUN ========================================
# =====================================================================================

# run simulation
lmp -i input.lmp -l log.lammps &

# return to original folder
cd ..
cd ..