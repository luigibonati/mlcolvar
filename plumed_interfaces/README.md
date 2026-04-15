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

The `tests` folder contains example folders for two simple systems (alanine and NaCl) that can be used to test the available interfaces and possible modifications.
Two types of scripts are provided for the two systems:

- `driver_alanine.sh` / `driver_NaCl.sh`: run a PLUMED driver to post-process a tracjetory. The output COLVAR file is also compared with a reference and the result is printed. These are used for testing the interfaces in GitHub.
- `run_alanine.sh` / `run_NaCl.sh`: run a simualtion with PLUMED generate a (biased)tracjetory. These are not used for testing the interfaces in GitHub.

Test scripts **expect an input string** that specifies the type of interface that needs to be tested.
- Alanine tests support all the modes:`descriptors`, `descriptors-kbias`, `gnn`, `gnn-kbias`.
- NaCl tests support the gnn-based modes only: `gnn`, `gnn-kbias`.


### Notes

- The test scripts use environment-specific paths and executable names. Edit these paths before running the tests, otherwise an error will be thrown.
- The test inputs themselves include pre-trained model files and PLUMED input files for the available modes.
- The `LOAD FILE=...` directive in the PLUMED input files references the corresponding `*.cpp` interface source file.
- The scripts are designed to be run from a folder within the mlcolvar root folder (e.g., mlcolvar/aux, which is ignored by git).

### How to use the tests

1. Make sure PLUMED is installed and built with the required `libtorch` support.
2. \[For gnn-based\] Install/activate/locate the Python environment that provides `mdtraj`.
3. Adjust the environment-specific paths in the needed test script(s) among:
   - `tests/driver_alanine.sh`
   - `tests/run_alanine.sh`
   - `tests/driver_NaCl.sh`
   - `tests/run_NaCl.sh`

4. Run the desired test script from a folder within the mlcolvar root folder (e.g., mlcolvar/aux, which is ignored by git):

```bash
# For example, to run tests with PLUMED driver

cd aux
bash ../plumed_interfaces/tests/driver_alanine.sh descriptors
bash ../plumed_interfaces/tests/driver_alanine.sh descriptors-kbias
bash ../plumed_interfaces/tests/driver_alanine.sh gnn
bash ../plumed_interfaces/tests/driver_alanine.sh gnn-kbias

bash ../plumed_interfaces/tests/driver_NaCl.sh gnn
bash ../plumed_interfaces/tests/driver_NaCl.sh gnn-kbias
```

#### Alanine test templates

- `tests/alanine/descriptor_based`
  - `plumed_PytorchModel.dat`
  - `plumed_PytorchKolmogorovBias.dat`
  - `model.pt`
  - `stateA.tpr`
- `tests/alanine/gnn_based`
  - `plumed_PytorchModelGNN.dat`
  - `plumed_PytorchKolmogorovBiasGNN.dat`
  - `model_q.pt`
  - `model.pt`
  - `ref.pdb`
  - `stateA.tpr`
  - `topolvac.top`
  - `confAvac.gro`
  - `gromppvac.mdp`
  - `mdout.mdp`

#### NaCl test templates

- `tests/NaCl/gnn_based`
  - `plumed_PytorchModelGNN.dat`
  - `plumed_PytorchKolmogorovBiasGNN.dat`
  - `plumed_unbiased.dat`
  - `model.pt`
  - `NaCl.pdb`
  - `NaCl_216wat_bound.data`
  - `NaCl_216wat_unbound.data`
  - `input.lmp`