#!/bin/bash

# retrieve mode:
# descriptors, descriptors-kbias, gnn, gnn-kbias, gnn-exported
mode=$1

# =====================================================================================
# ======================================= SETUP =======================================
# =====================================================================================

# define path to sourceme.sh files for plumed
# this needs to be edited by the user before running the script
PLUMED_SOURCE="/path/to/plumed/sourceme.sh"

# define python path with mdtraj, only for GNN-based modes
PYTHON_PATH="/path/to/python/with/mdtraj"

# =====================================================================================
# ======================================= CHECKS ======================================
# =====================================================================================

# try to source plumed, if not found print error message and exit
if ! source "$PLUMED_SOURCE" 2>/dev/null; then
    echo "PLUMED sourceme.sh file could not be found. Please edit the script to source PLUMED before running it."
    exit 1
fi

# check that plumed is sourced
if ! command -v plumed &> /dev/null; then
    echo "PLUMED executable not working, please check!"
    exit 1
fi

# check python and mdtraj for GNN-based modes
if [ "$mode" = "gnn" ] \
    || [ "$mode" = "gnn-kbias" ] \
    || [ "$mode" = "gnn-exported" ]; then

    if [ ! -x "$PYTHON_PATH" ]; then
        echo "Python could not be found. Please edit the script to set the PYTHON_PATH variable to a Python executable with mdtraj installed."
        exit 1
    fi

    if ! "$PYTHON_PATH" -c "import mdtraj" 2>/dev/null; then
        echo "mdtraj is not installed in the Python environment."
        exit 1
    fi
fi

# add PyTorch shared libraries to the runtime search path
# required when loading AOTInductor packages such as model.pt2
if [ "$mode" = "gnn-exported" ]; then

    TORCH_LIB_DIR=$(
        "$PYTHON_PATH" -c \
        "from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / 'lib')"
    )

    if [ ! -f "$TORCH_LIB_DIR/libtorch.so" ]; then
        echo "libtorch.so could not be found in: $TORCH_LIB_DIR"
        exit 1
    fi

    export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$TORCH_LIB_DIR:${LIBRARY_PATH:-}"

    echo "PyTorch library directory: $TORCH_LIB_DIR"
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
fi

# =====================================================================================
# ====================================== PREPARE ======================================
# =====================================================================================

# create run folder
FOLDER_NAME="test_driver_alanine"
rm -rf "$FOLDER_NAME"
echo "folder $FOLDER_NAME"

# copy template folder
if [ "$mode" = "descriptors" ] \
    || [ "$mode" = "descriptors-kbias" ]; then

    cp -r \
        ../plumed_interfaces/tests/alanine/descriptor_based_inputs \
        "$FOLDER_NAME"

elif [ "$mode" = "gnn" ] \
    || [ "$mode" = "gnn-kbias" ] \
    || [ "$mode" = "gnn-exported" ]; then

    cp -r \
        ../plumed_interfaces/tests/alanine/gnn_based_inputs \
        "$FOLDER_NAME"

else
    echo "Invalid mode. Use 'descriptors', 'descriptors-kbias', 'gnn', 'gnn-kbias' or 'gnn-exported'."
    exit 1
fi

cd "$FOLDER_NAME" || exit 1

cp ../../plumed_interfaces/tests/alanine/driver_data/* .


if [ "$mode" = "descriptors" ]; then

    cp ../../plumed_interfaces/PytorchModel.cpp .
    mv plumed_PytorchModel.dat plumed.dat

elif [ "$mode" = "descriptors-kbias" ]; then

    cp ../../plumed_interfaces/PytorchKolmogorovBias.cpp .
    mv plumed_PytorchKolmogorovBias.dat plumed.dat

elif [ "$mode" = "gnn" ]; then

    cp ../../plumed_interfaces/PytorchModelGNN.cpp .
    mv plumed_PytorchModelGNN.dat plumed.dat

elif [ "$mode" = "gnn-kbias" ]; then

    cp ../../plumed_interfaces/PytorchKolmogorovBiasGNN.cpp .
    mv plumed_PytorchKolmogorovBiasGNN.dat plumed.dat

elif [ "$mode" = "gnn-exported" ]; then

    cp ../../plumed_interfaces/PytorchModelGNNExported.cpp .
    mv plumed_PytorchModelGNNExported.dat plumed.dat

fi

# remove unused input files
rm -f plumed_*

# remove models possibly copied from the template directory
rm -f model.pt model.pt2

# update python path
sed -i \
    "s|PYTHON_BIN=/path/to/python/with/mdtraj|PYTHON_BIN=$PYTHON_PATH|g" \
    plumed.dat

# remove bias commands from plumed.dat
sed -i '/^[[:space:]]*opes:/d' plumed.dat
sed -i '/^[[:space:]]*BIASVALUE/d' plumed.dat

# change printing stride
sed -i "s|STRIDE=500|STRIDE=1|g" plumed.dat

# =====================================================================================
# ======================================== RUN ========================================
# =====================================================================================

# train and export model
if ! "$PYTHON_PATH" \
    ../../plumed_interfaces/tests/alanine/train_cv.py \
    "$mode"; then

    echo "[TEST FAILED] Model training or export failed."
    exit 1
fi

# run PLUMED driver
if ! plumed driver \
    < plumed.dat \
    --timestep 1 \
    --ixtc traj_comp.xtc; then

    echo "[TEST FAILED] PLUMED driver failed."
    exit 1
fi

# compare PLUMED and Python outputs
echo ""
echo "Comparing generated COLVAR with Python model output..."

head -n 30 COLVAR

if "$PYTHON_PATH" \
    ../../plumed_interfaces/tests/alanine/compare_results.py \
    "$mode"; then

    echo "[TEST PASSED] Generated COLVAR file matches the Python model output (relative numerical tolerance 1e-2)"
    exit 0

else

    echo "[TEST FAILED] Generated COLVAR file differs from the Python model output (relative numerical tolerance 1e-2)"
    cd ../..
    exit 1

fi