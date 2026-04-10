# PLUMED Interfaces

This folder contains most updated PLUMED interface source files and test input sets for the `mlcolvar` PLUMED integration.

## Interface files

- `PytorchModel.cpp`
  - Implements the `PYTORCH_MODEL` function for the use of standard descriptor-based PyTorch models as CVs. This is also available in PLUMED official releases.
- `PytorchKolmogorovBias.cpp`
  - Implements the `PYTORCH_KOLMOGOROV_BIAS` function for computing transition-state-oriented Kolmogorov bias from PyTorch descriptor-based model of the committor whicha can also be used as CVs.
- `PytorchModelGNN.cpp`
  - Implements the `PYTORCH_GNN` colvar for the use of graph neural network models as CVs. The construction of the graph is done in PLUMED.
- `PytorchKolmogorovBiasGNN.cpp`
  - Implements the `PYTORCH_KOLMOGOROV_BIAS_GNN` colvar for computing transition-state-oriented Kolmogorov bias from raph neural network models of the committor whicha can also be used as CVs.

## Test inputs

The `test_inputs` folder contains example folders and run scripts for two systems:

- `test_inputs/test_alanine.sh`
  - Launches alanine test runs.
  - Supported modes: `descriptors`, `descriptors-kbias`, `gnn`, `gnn-kbias`.
  - Chooses the corresponding test template from `test_inputs/alanine/descriptor_based` or `test_inputs/alanine/gnn_based`.
- `test_inputs/test_NaCl.sh`
  - Launches the NaCl test run.
  - Supported modes: `gnn`, `gnn-kbias`.
  - Uses the `test_inputs/NaCl/gnn_based` template.

### Alanine test templates

- `test_inputs/alanine/descriptor_based`
  - `plumed_PytorchModel.dat`
  - `plumed_PytorchKolmogorovBias.dat`
  - `model_z.pt`
  - `stateA.tpr`
- `test_inputs/alanine/gnn_based`
  - `plumed_PytorchModelGNN.dat`
  - `plumed_PytorchKolmogorovBiasGNN.dat`
  - `model_q.pt`
  - `model_z.pt`
  - `ref.pdb`
  - `stateA.tpr`
  - `topolvac.top`
  - `confAvac.gro`
  - `gromppvac.mdp`
  - `mdout.mdp`

### NaCl test templates

- `test_inputs/NaCl/gnn_based`
  - `plumed_PytorchModelGNN.dat`
  - `plumed_PytorchKolmogorovBiasGNN.dat`
  - `plumed_unbiased.dat`
  - `model_z.pt`
  - `NaCl.pdb`
  - `NaCl_216wat_bound.data`
  - `NaCl_216wat_unbound.data`
  - `input.lmp`

## How to use the tests

1. Make sure PLUMED is installed and built with the required `libtorch` support.
2. \[For gnn-based\] Install/activate/locate the Python environment that provides `mdtraj`.
3. Adjust the environment-specific paths in the test scripts:
   - `test_inputs/test_alanine.sh`
   - `test_inputs/test_NaCl.sh`

   These scripts currently source local installation scripts and set a hard-coded Python path.

4. Run the desired test script from the repository root:

```bash
bash plumed_interfaces/test_inputs/test_alanine.sh descriptors
bash plumed_interfaces/test_inputs/test_alanine.sh descriptors-kbias
bash plumed_interfaces/test_inputs/test_alanine.sh gnn
bash plumed_interfaces/test_inputs/test_alanine.sh gnn-kbias

bash plumed_interfaces/test_inputs/test_NaCl.sh gnn
bash plumed_interfaces/test_inputs/test_NaCl.sh gnn-kbias
```

## What happens during a test run

- The test script copies the selected template folder to a working folder.
- It copies one of the PLUMED interface source files from `plumed_interfaces/` into that working folder.
- It renames the appropriate `plumed_*.dat` file to `plumed.dat`.
- The script updates the `PYTHON_BIN` entry in the PLUMED input file to point to the configured Python interpreter \[ For gnn-based \].
- Finally, it launches the MD engine (`gmx mdrun` for alanine, `lmp` for NaCl).

## Notes

- The test scripts use environment-specific paths and executable names. Update these paths before running the tests.
- The test inputs themselves include pre-trained model files and PLUMED input files for the available modes.
- The `LOAD FILE=...` directive in the PLUMED input files references the corresponding `*.cpp` interface source file.

## Recommended next step

If you want to add a new test or a new PLUMED interface variant, add a new `*.cpp` source file here, then create a matching `plumed_*.dat` input and a new mode in the appropriate test script.
