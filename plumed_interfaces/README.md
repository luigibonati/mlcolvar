# PLUMED Interfaces

This folder contains the most updated source files of the PLUMED interface for the use of machine learning collective variables.
These interfaces are primarily intended (but not limited) to be used with CV models trained through `mlcolvar` and can be loaded into a libtorch-aware PLUMED using the `LOAD` command at runtime.

This folder also provides a few sets of test inputs for alanine dipeptide and the dissociation of NaCl that can be used to test the interfaces or possible modifications.

## Interface files

#### Machine learning collective variables

- `PytorchModel.cpp`
  - Implements the `PYTORCH_MODEL` function for the use of standard descriptor-based PyTorch models as CVs. This is also available in PLUMED official releases.
  Note that to avoid conflicts with default PLUMED installation, the action defined in the source file provided here is named `PYTORCH_MODEL_RUNTIME`.

- `PytorchModelGNN.cpp`
  - Implements the `PYTORCH_GNN` colvar for the use of gnn-based models as CVs. The construction of the input graph is done in PLUMED using its built-in neighbor-list.

#### Kolmogorov transition-state oriented bias from ML-models of the committor function

- `PytorchKolmogorovBias.cpp`
  - Implements the `PYTORCH_KOLMOGOROV_BIAS` function for computing transition-state-oriented Kolmogorov bias from PyTorch descriptor-based model of the committor which can also be used as CVs.

- `PytorchKolmogorovBiasGNN.cpp`
  - Implements the `PYTORCH_KOLMOGOROV_BIAS_GNN` colvar for computing transition-state-oriented Kolmogorov bias from gnn-based models of the committor which can also be used as CVs.

## Test inputs

The `test_inputs` folder contains example folders for two simple systems (alanine and NaCl) that can be used to test the available interfaces and possible modifications.
Two run scripts are provided for the two systems as example:

- `test_inputs/test_alanine.sh`
  - Launches alanine test runs.
  - Supported modes: `descriptors`, `descriptors-kbias`, `gnn`, `gnn-kbias`.
  - Chooses the corresponding test template from `test_inputs/alanine/descriptor_based` or `test_inputs/alanine/gnn_based`.
- `test_inputs/test_NaCl.sh`
  - Launches the NaCl test run.
  - Supported modes: `gnn`, `gnn-kbias`.
  - Uses the `test_inputs/NaCl/gnn_based` template.

### Notes

- The test scripts use environment-specific paths and executable names. Edit these paths before running the tests, otherwise an error will be thrown.
- The test inputs themselves include pre-trained model files and PLUMED input files for the available modes.
- The `LOAD FILE=...` directive in the PLUMED input files references the corresponding `*.cpp` interface source file.
- The scripts are designed to be run from a folder within the mlcolvar root folder (e.g., mlcolvar/aux, which is ignored by git).

### How to use the tests

1. Make sure PLUMED is installed and built with the required `libtorch` support.
2. \[For gnn-based\] Install/activate/locate the Python environment that provides `mdtraj`.
3. Adjust the environment-specific paths in the test scripts:
   - `test_inputs/test_alanine.sh`
   - `test_inputs/test_NaCl.sh`

4. Run the desired test script from a folder within the mlcolvar root folder (e.g., mlcolvar/aux, which is ignored by git):

```bash
cd aux
bash ../plumed_interfaces/test_inputs/test_alanine.sh descriptors
bash ../plumed_interfaces/test_inputs/test_alanine.sh descriptors-kbias
bash ../plumed_interfaces/test_inputs/test_alanine.sh gnn
bash ../plumed_interfaces/test_inputs/test_alanine.sh gnn-kbias

bash ../plumed_interfaces/test_inputs/test_NaCl.sh gnn
bash ../plumed_interfaces/test_inputs/test_NaCl.sh gnn-kbias
```

#### Alanine test templates

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

#### NaCl test templates

- `test_inputs/NaCl/gnn_based`
  - `plumed_PytorchModelGNN.dat`
  - `plumed_PytorchKolmogorovBiasGNN.dat`
  - `plumed_unbiased.dat`
  - `model_z.pt`
  - `NaCl.pdb`
  - `NaCl_216wat_bound.data`
  - `NaCl_216wat_unbound.data`
  - `input.lmp`