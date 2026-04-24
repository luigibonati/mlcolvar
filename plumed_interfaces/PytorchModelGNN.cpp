/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (c) 2022-2024 of Luigi Bonati and Enrico Trizio.

The pytorch module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The pytorch module is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#ifdef __PLUMED_HAS_LIBTORCH

#include <cmath>
#include <memory>
#include <fstream>
#include <type_traits>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/jit_log.h>

#include "core/PlumedMain.h"
#include "config/Config.h"
#include "colvar/Colvar.h"
#include "colvar/ActionRegister.h"
#include "tools/NeighborList.h"
#include "tools/Communicator.h"
#include "tools/OpenMP.h"
#include "tools/File.h"
#include "tools/PDB.h"

// NOTE: Freezing a ScriptModule (torch::jit::freeze) works only in >=1.11
// For 1.8 <= versions <=1.10 we need a hack
// (see https://discuss.pytorch.org/t/how-to-check-libtorch-version/77709/4 and also
// https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp#L479)
// adapted from NequIP https://github.com/mir-group/nequip
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
#define DO_TORCH_FREEZE_HACK
// For the hack, need more headers:
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#endif

using namespace std;

namespace PLMD {

class NeighborList;

namespace colvar {

namespace pytorch_gnn {

template <typename Main, typename Atoms>
auto getUsingNaturalUnits(Main& main, Atoms&, int)
  -> decltype(main.usingNaturalUnits()) {
  return main.usingNaturalUnits();
}

template <typename Main, typename Atoms>
auto getUsingNaturalUnits(Main&, Atoms& atoms, long)
  -> decltype(atoms.usingNaturalUnits()) {
  return atoms.usingNaturalUnits();
}

template <typename Main, typename Atoms>
auto getLengthUnit(Main& main, Atoms&, int)
  -> decltype(main.getUnits().getLength()) {
  return main.getUnits().getLength();
}

template <typename Main, typename Atoms>
auto getLengthUnit(Main&, Atoms& atoms, long)
  -> decltype(atoms.getUnits().getLength()) {
  return atoms.getUnits().getLength();
}

//+PLUMEDOC PYTORCH_GNN PYTORCH_GNN
/*
Load a Graph Neural Network (GNN) model compiled with TorchScript.

This module uses a fixed length unit of _Angstrom_. Thus, the GNN model read by
this module should be trained under the same unit convention. Besides, the
module constructs node attributes w.r.t to the atomic types. As a result, this
module require a PDB file which records names of _ALL_ atoms in the system
(the STRUCTURE keyword). Note that the atom names in this PDB file could
_ONLY_ be element symbols, e.g.:
\auxfile{plumed_topo.pdb}
ATOM      1  H   ACE A   1      15.100  12.940  29.390  1.00  0.00           H
ATOM      2  C   ACE A   1      14.970  13.860  29.960  1.00  0.00           C
ATOM      3  H   ACE A   1      15.720  13.820  30.760  1.00  0.00           H
ATOM      4  H   ACE A   1      13.980  13.920  30.410  1.00  0.00           H
ATOM      5  C   ACE A   1      15.300  15.070  29.100  1.00  0.00           C
\endauxfile

The module constructs graph edges between neighbors inside the selected atom
group, using the cutoff value recorded in the model file. By default, such an
atom group is defined by the single SYSTEM_SELECTION keyword. Under this case, the node
number of the input graph in each MD step is fixed, and the number of edges
will change according to the relative postions of the atoms.
However, if the ENVIRONMENT_SELECTION parameter is given, the atom group mentioned above will
contain all atoms in SYSTEM_SELECTION, _AND_ atoms in ENVIRONMENT_SELECTION which are within a radius of
_ANY_ atom in SYSTEM_SELECTION. Such a radius of selecting atoms from ENVIRONMENT_SELECTION equals to the
cutoff radius recorded in the model file _plus_ the buffer size controlled by
the BUFFER keyword. Thus, when ENVIRONMENT_SELECTION is given, the node number of the input
graph could fluctuate in different MD steps.

This module also support committor calculations. When the input PyTorch model
is a committor model, the outputs will assign the zeta value to the first
output (node-0) and the q value to the second output (node-1). If the KBIAS
keyword is also given, the module will also calculate the committor bias and
assign it with a label of kbias. These information will be shown in the log as
well.

Note that this function requires \ref installation-libtorch LibTorch C++ library.
Check the instructions in the \ref PYTORCH page to enable the module.
Specifically, we encourage the user to install the GPU-enabled version of
LibTorch, when dealing with large input graphs.

\par Examples
The following example instructs plumed to evaluate the GNN model using the atoms 1-10. The neighbor list for determining the edges will be updated every 100 steps.
\plumedfile
PYTORCH_GNN ...
  SYSTEM_SELECTION=1-10
  MODEL=model.ptc
  STRUCTURE=plumed_topo.pdb
  NL_STRIDE=100
  LABEL=gnn
... PYTORCH_GNN
\endplumedfile

The following example instructs plumed to do the same calculation as the above example, but will evaluate the model on CUDA using double precision, and add an OPES bias potential on the CV.
\plumedfile
PYTORCH_GNN ...
  SYSTEM_SELECTION=1-10
  MODEL=model.ptc
  STRUCTURE=plumed_topo.pdb
  NL_STRIDE=100
  CUDA
  FLOAT64
  LABEL=gnn
... PYTORCH_GNN

OPES_METAD ...
  LABEL=opes
  ARG=gnn.node-0
  FILE=KERNELS
  PACE=500
  TEMP=300
  BARRIER=35
... OPES_METAD
\endplumedfile


The following example instructs plumed to evaluate the GNN model using the atoms 1-10 as center atoms, and atoms 11-100 as the environment atoms. The buffer size used for selecting active atoms from the environment atoms is 2 PLUMED unit. The neighbor list for determining the edges will be updated every 2 steps.
\plumedfile
PYTORCH_GNN ...
  SYSTEM_SELECTION=1-10
  ENVIRONMENT_SELECTION=11-100
  MODEL=model.ptc
  STRUCTURE=plumed_topo.pdb
  NL_STRIDE=2
  BUFFER=2.0
  LABEL=gnn
... PYTORCH_GNN
\endplumedfile

*/
//+ENDPLUMEDOC


class PytorchGNN: public Colvar
{
  int n_out = 0;
  bool pbc = true;
  bool serial = false;
  bool firsttime = true;
  bool invalidate_list = true;
  bool bailout_fusion = false;
  double r_max = 0.0; // In PLUMED length unit
  double buffer = 0.0; // In PLUMED length unit
  std::string model_file_name;
  std::string structure_file_name;
  std::vector<int> system_node_types;
  std::vector<int> model_atomic_numbers;
  std::vector<AtomNumber> atom_list_a;
  std::vector<AtomNumber> atom_list_b;
  std::vector<int> atom_list_active; // local_ids
  std::unique_ptr<NeighborList> neighbor_list;
  torch::jit::script::Module model;
  torch::ScalarType torch_float_dtype = torch::kFloat32;
  torch::Device device = c10::Device(torch::kCPU);
  const std::array<std::string, 118> periodic_table = {
     "h", "he",
     "li", "be",                                                              "b",  "c",  "n",  "o",  "f", "ne",
     "na", "mg",                                                             "al", "si",  "p",  "s", "cl", "ar",
     "k",  "ca", "sc", "ti",  "v", "cr", "mn", "fe", "co", "ni", "cu", "zn", "ga", "ge", "as", "se", "br", "kr",
     "rb", "sr",  "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in", "sn", "sb", "te",  "i", "xe",
     "cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb", "lu",
                       "hf", "ta",  "w", "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi", "po", "at", "rn",
     "fr", "ra", "ac", "th", "pa",  "u", "np", "pu", "am", "cm", "bk", "cf", "es", "fm", "md", "no", "lr",
                       "rf", "db", "sg", "bh", "hs", "mt", "ds", "rg", "cn", "nh", "fl", "mc", "lv", "ts", "og"
  }; // TODO: add ghost atoms
  std::string model_summary(
    std::string model_name, torch::jit::Module module, int level_max, int level
  );
  int atomic_number_from_name(std::string name);
  bool groups_have_intersection(void);
  void find_active_atoms(int n_threads);

public:
  explicit PytorchGNN(const ActionOptions&);
  ~PytorchGNN();
  static void registerKeywords(Keywords& keys);
  void calculate() override;
  void prepare() override;
}; // class PytorchGNN

PLUMED_REGISTER_ACTION(PytorchGNN, "PYTORCH_GNN")

void PytorchGNN::registerKeywords(Keywords& keys)
{
  Colvar::registerKeywords(keys);

  keys.add(
    "atoms",
    "SYSTEM_SELECTION",
    "First list of atoms (corresponding to the `system_selection` in mlcolvar)"
  );

  keys.add(
    "atoms",
    "ENVIRONMENT_SELECTION",
    "Second list of atoms (corresponding to the `environment_selection` in mlcolvar`)"
  );

  keys.add(
    "compulsory",
    "MODEL",
    "Filename of the PyTorch compiled model"
  );

  keys.add(
    "compulsory",
    "STRUCTURE",
    "PDB file name that contains the whole simulated system, with correct atom names and orders"
  );

  keys.add(
    "optional",
    "NL_STRIDE",
    "The frequency with which we are updating the atoms in the neighbor list"
  );

  keys.add(
    "optional",
    "BUFFER",
    "Buffer size used in finding active environment atoms"
  );

  keys.addFlag(
    "CUDA",
    false,
    "Perform the calculation on CUDA"
  );

  keys.addFlag(
    "SERIAL",
    false,
    "Perform the calculation in serial - for debug purpose"
  );

  keys.addFlag(
    "BAILOUTFUSION",
    false,
    "Use a faster LibTorch fusion strategy (experimental), by default false. Set it to false for debugging and older version compatibility."
  );

  keys.addFlag(
    "FLOAT64",
    false,
    "Evaluate the model in double precision"
  );

  keys.addOutputComponent(
    "node",
    "default",
    "Model outputs"
  );

}

PytorchGNN::PytorchGNN(const ActionOptions& ao):
  PLUMED_COLVAR_INIT(ao)
{// print libtorch version
  std::stringstream ss;
  ss << TORCH_VERSION_MAJOR << "." \
     << TORCH_VERSION_MINOR << "." \
     << TORCH_VERSION_PATCH;
  std::string version;
  ss >> version; // extract into the string.
  std::string version_info = "  LibTorch version: " + version + "\n";
  log.printf(version_info.data());

  // parse input
  parseAtomList("SYSTEM_SELECTION", atom_list_a);
  parseAtomList("ENVIRONMENT_SELECTION", atom_list_b);

  parse("MODEL", model_file_name);

  parse("STRUCTURE", structure_file_name);

  int neighbor_list_stride = 1;
  parse("NL_STRIDE", neighbor_list_stride);
  if (neighbor_list_stride <= 0)
    plumed_merror("NL_STRIDE should be positive!");

  parse("BUFFER", buffer);
  if (buffer > 0 && atom_list_b.size() == 0)
    plumed_merror("No ENVIRONMENT_SELECTION given! Cannot define the BUFFER key!");

  bool use_cuda = false;
  bool required_cuda = false;
  parseFlag("CUDA", required_cuda);

  parseFlag("SERIAL", serial);
  if (required_cuda and serial)
    plumed_merror("Can not enable CUDA with SERIAL at the same time!");

  bool use_float64 = false;
  parseFlag("FLOAT64", use_float64);

  bool nopbc = !pbc;
  parseFlag("NOPBC", nopbc);
  pbc = !nopbc;

  parseFlag("BAILOUTFUSION", bailout_fusion);

  checkRead();

  // check groups
  if (atom_list_b.size() > 0) {
    if (groups_have_intersection())
      plumed_merror("SYSTEM_SELECTION can't intersect with ENVIRONMENT_SELECTION!");
    atom_list_active.resize(atom_list_a.size() + atom_list_b.size());
    atom_list_active.clear();
  } else {
    atom_list_active.resize(atom_list_a.size());
    atom_list_active.clear();
    find_active_atoms(1);
  }

  // check precision to be used
  if (use_float64)
    torch_float_dtype = torch::kFloat64;

  // check whether to use CUDA
  if (required_cuda && torch::cuda::is_available()) {
    device = c10::Device(torch::kCUDA);
    use_cuda = true;
  } else if (required_cuda) {
    use_cuda = false;
  }

  // check structure file
  PDB pdb;
  FILE *fp = fopen(structure_file_name.c_str(), "r");
  if (fp != NULL) {
    pdb.readFromFilepointer(
      fp,
      getUsingNaturalUnits(plumed, plumed.getAtoms(), 0),
      0.1 / getLengthUnit(plumed, plumed.getAtoms(), 0)
    );
    fclose(fp);
  } else {
    plumed_merror("Can not open PDB file: '" + structure_file_name + "'");
  }

  // deserialize the model from file
  try {
    model = torch::jit::load(model_file_name, device);
  } catch (const c10::Error& e) {
    plumed_merror(
      "Can't load model file: '" + model_file_name + "'. Reason: " + e.what()
    );
  }

  // disable parameter grads
  for (auto p: model.parameters())
    p.requires_grad_(false);

  // set up model precision
  model.to(torch_float_dtype);

  // summary
  std::string model_architecture = model_summary("CV", model, 3, 0);

  // get CV length
  if (!model.hasattr("n_out"))
    plumed_merror(
      "Can not find model attribute 'n_out'! This has to be set during the compilation of the model!"
    );
  if (model.hasattr("n_out"))
    n_out = model.attr("n_out").toTensor().item<int>();
  
  // get cutoff radius
  if (!model.hasattr("r_max") && !model.hasattr("cutoff") )
    plumed_merror(
      "Can not find model attribute: 'r_max' or 'cutoff'! One of these attributes has to be set during the compilation of the model!"
    );
  else if (model.hasattr("r_max") && model.hasattr("cutoff") )
    plumed_merror(
      "Both model attribute: 'r_max' and 'cutoff' are defined!"
    );

  // TODO: now, the `r_max` parameter in the model file is defined in unit of Angstrom.
  // We should warn the users about this default
  if (model.hasattr("cutoff"))
    r_max = model.attr("cutoff").toTensor().item<double>();
  else
    r_max = model.attr("r_max").toTensor().item<double>();
  r_max = r_max / getLengthUnit(plumed, plumed.getAtoms(), 0) * 0.1;

  // get atomic numbers
  if (!model.hasattr("atomic_numbers"))
    plumed_merror(
      "Can't find model attribute: 'atomic_numbers'! This attribute has to be set during the compilation of the model!"
    );
  auto atomic_numbers = model.attr("atomic_numbers").toTensor();
  for (int64_t i = 0; i < atomic_numbers.size(0); i++)
    model_atomic_numbers.push_back(atomic_numbers[i].item<int64_t>());

// https://stackoverflow.com/questions/77102532/libtorch-performance-issue-when-using-multiple-gpus-in-multiple-threads
  if (bailout_fusion) {
    torch::jit::FusionStrategy bailout = {
      {torch::jit::FusionBehavior::STATIC,  0},
      {torch::jit::FusionBehavior::DYNAMIC, 0},
    };
    torch::jit::setFusionStrategy(bailout);
  }

  // optimize model
  model.eval();
#ifdef DO_TORCH_FREEZE_HACK
  // NOTE: do the hack
  // copied from the implementation of torch::jit::freeze,
  // except without the broken check
  // see https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp
  bool optimize_numerics = true;  // the default
  // the {} is preserved_attrs
  auto out_mod = torch::jit::freeze_module(model, {});
  // see 1.11 bugfix in https://github.com/pytorch/pytorch/pull/71436
  auto graph = out_mod.get_method("forward").graph();
  OptimizeFrozenGraph(graph, optimize_numerics);
  model = out_mod;
#else
  // do it normally
  model = torch::jit::freeze(model);
#endif

  // optimize model for inference
  if (TORCH_VERSION_MAJOR == 2 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 10)) {
    model = torch::jit::optimize_for_inference(model);
  }

  // send the model to device
  model.to(device);

  // create system atomic numbers
  std::vector<int> atom_is_required(pdb.getAtomNumbers().size());
  for (size_t i = 0; i < atom_list_a.size(); i++) {
    int index = atom_list_a[i].index();
    atom_is_required[index] = 1;
  }
  for (size_t i = 0; i < atom_list_b.size(); i++) {
    int index = atom_list_b[i].index();
    atom_is_required[index] = 1;
  }

  for (size_t i = 0; i < pdb.getAtomNumbers().size(); i++) {
    AtomNumber index = pdb.getAtomNumbers()[i];
    std::string name = pdb.getAtomName(index);
    int number = atomic_number_from_name(name);
    auto iter = std::find(
      model_atomic_numbers.begin(),
      model_atomic_numbers.end(),
      number
    );
    if (iter == model_atomic_numbers.end()) {
      if (atom_is_required[i])
        plumed_merror(
          "Element '" + name + "' does not present in model " + model_file_name
        );
      else
        system_node_types.push_back(-1);
    } else {
      int node_type = std::distance(model_atomic_numbers.begin(), iter);
      system_node_types.push_back(node_type);
    }
  }

  // create components
  for (int i = 0; i < n_out; i++) {
    string name_comp = "node-" + std::to_string(i);
    addComponentWithDerivatives(name_comp);
    componentIsNotPeriodic(name_comp);
  }

  // initialize the neighbor list
  if (atom_list_b.size() > 0)
    neighbor_list = Tools::make_unique<NeighborList>(
      atom_list_a,
      atom_list_b,
      serial,
      false,
      pbc,
      getPbc(),
      comm,
      r_max + buffer,
      neighbor_list_stride
    );
  else
    neighbor_list = Tools::make_unique<NeighborList>(
      atom_list_a,
      serial,
      pbc,
      getPbc(),
      comm,
      r_max + buffer,
      neighbor_list_stride
    );
  requestAtoms(neighbor_list->getFullAtomList());

  // print log
  std::string thename = getLabel();
  if(atom_list_b.size() > 0) {
    log.printf(
      "  Will build graphs using %u system and %u environment atoms\n",
      static_cast<unsigned>(atom_list_a.size()),
      static_cast<unsigned>(atom_list_b.size())
    );
    log.printf("  System atom list (SYSTEM_SELECTION):\n");
    for (unsigned int i = 0; i < atom_list_a.size(); i++) {
      if (((i + 1) % 10) == 0)
        log.printf("\n");
      log.printf("  %d", atom_list_a[i].serial());
    }
    log.printf("\n");
    log.printf("  Environment atom list (ENVIRONMENT_SELECTION):\n");
    for (unsigned int i = 0; i < atom_list_b.size(); i++) {
      if (((i + 1) % 10) == 0)
        log.printf("\n");
      log.printf("  %d", atom_list_b[i].serial());
    }
    log.printf("\n");
  } else {
    log.printf(
      "  Will build graphs using %u atoms\n",
      static_cast<unsigned>(atom_list_a.size())
    );
    log.printf("  Atom list:\n");
    for (unsigned int i = 0; i < atom_list_a.size(); i++) {
      if (((i + 1) % 10) == 0)
        log.printf("\n");
      log.printf("  %d", atom_list_a[i].serial());
    }
    log.printf("\n");
  }
  log << "  Model atomic numbers: " << model_atomic_numbers;
  log.printf("\n");
  log.printf("  Boundary conditions: ");
  if (pbc)
    log.printf("periodic\n");
  else
    log.printf("non-periodic\n");
  log.printf("  Neighbor List update stride: %d\n", neighbor_list_stride);
  log.printf("  Graph cutoff radius: %f (PLUMED length unit)\n", r_max);
  if(atom_list_b.size() > 0)
    log.printf("  Environment buffer size: %f (PLUMED length unit)\n", buffer);
  log.printf("  Number of outputs: %d \n", n_out);
  log.printf("  Will run on device: ");
  if (use_cuda)
    log.printf("CUDA\n");
  else if (required_cuda)
    log.printf("CPU (CUDA device not found/LibTorch does not support CUDA)\n");
  else
    log.printf("CPU (as required)\n");
  log << "  Model architecture: \n";
  log << model_architecture;
  log.printf("  Bibliography: ");
  log << plumed.cite("Bonati, Trizio, Rizzi and Parrinello, J. Chem. Phys. 159, 014801 (2023)");
  log << plumed.cite("Zhang et al., J. Chem. Theory Comput. 20, 24, 10787–10797 (2024)");
  log.printf("\n");
}

PytorchGNN::~PytorchGNN()
{
  return;
}

void PytorchGNN::prepare()
{
  if (neighbor_list->getStride() > 0) {
    if (firsttime || ((getStep() % neighbor_list->getStride()) == 0)) {
      requestAtoms(neighbor_list->getFullAtomList());
      invalidate_list = true;
      firsttime = false;
    } else {
      requestAtoms(neighbor_list->getReducedAtomList());
      invalidate_list = false;
      if (getExchangeStep())
        plumed_merror(
          "Neighbor lists should be updated on exchange steps - choose a NL_STRIDE which divides the exchange stride!"
        );
    }
    if (getExchangeStep())
      firsttime = true;
  }
}


void PytorchGNN::calculate()
{
  // get some common data
  auto pbc_tools = getPbc();
  int n_atoms = getNumberOfAtoms();
  std::vector<PLMD::Vector> x_local = getPositions();

  // threads
  int n_threads = OpenMP::getNumThreads();
  if (!serial)
    n_threads = std::min(n_threads, n_atoms);
  else
    n_threads = 1;

  // perform the size check
  if (system_node_types.size() != (size_t)plumed.getAtoms().getNatoms()) // TODO: this is now deprecated from plumed 2.10
    plumed_merror(
      "Structure file '" +
      structure_file_name +
      "' has different number of atoms with the simulated system!"
    );

  // update the neighbor list and number of atoms
  if (neighbor_list->getStride() > 0 && invalidate_list)
    neighbor_list->update(x_local);
  if (atom_list_b.size() > 0)
    find_active_atoms(n_threads);
  n_atoms = (int)atom_list_active.size();
  n_threads = std::min(n_threads, n_atoms);

  // get the unit
  double to_ang = 10 * getLengthUnit(plumed, plumed.getAtoms(), 0);

  // get the positions
  // TODO: now, the positions used by the model file is in unit of Angstrom.
  // We should warn the users about this default
  std::vector<float> positions_vector(n_atoms * 3);
  #pragma omp parallel for num_threads(n_threads)
  for (int i = 0; i < n_atoms; i++) {
    int index = atom_list_active[i]; 
    positions_vector[i * 3 + 0] = x_local[index][0] * to_ang;
    positions_vector[i * 3 + 1] = x_local[index][1] * to_ang;
    positions_vector[i * 3 + 2] = x_local[index][2] * to_ang;
  }  

  torch::Tensor positions = torch::from_blob(  
    positions_vector.data(),
    n_atoms * 3,
    torch::TensorOptions().dtype(torch::kFloat32)
  );
  positions = positions.to(device).to(torch_float_dtype);
  positions = positions.reshape({n_atoms, 3});

  // cell
  // TODO: now, the box data used by the model file is in unit of Angstrom.
  // We should warn the users about this default
  PLMD::Tensor box = getBox();
  std::vector<float> cell_vector(9);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      cell_vector[i * 3 + j] = box[i][j] * to_ang;
  }
  torch::Tensor cell = torch::from_blob(
    cell_vector.data(),
    9,
    torch::TensorOptions().dtype(torch::kFloat32)
  );
  cell = cell.to(device).to(torch_float_dtype);
  cell = cell.reshape({3, 3});

  // build node attributes
  // TODO: now, the node attributes are in MACE's format.
  // We should try to give more options, or warn the users about this default
  int n_node_feats = (int)model_atomic_numbers.size();
  std::vector<float> node_attrs_vector(n_node_feats * n_atoms);
  #pragma omp parallel for num_threads(n_threads)
  for (int i = 0; i < n_atoms; i++) {
    int index = atom_list_active[i];
    int node_type = system_node_types[getAbsoluteIndex(index).index()];
    node_attrs_vector[i * n_node_feats + node_type] = 1.0;
  }
  torch::Tensor node_attrs = torch::from_blob(
    node_attrs_vector.data(),
    n_node_feats * n_atoms,
    torch::TensorOptions().dtype(torch::kFloat32)
  );
  node_attrs = node_attrs.to(device).to(torch_float_dtype);
  node_attrs = node_attrs.reshape({n_atoms, n_node_feats});

  // build edges
  int n_edges = 0;
  torch::Tensor edge_index;

  if (atom_list_b.size() > 0) {
    n_edges = n_atoms * (n_atoms - 1);
    std::vector<float> distance_vector(n_edges);
    std::vector<std::vector<int64_t>> edge_index_vector;
    edge_index_vector.resize(2, std::vector<int64_t>(n_edges));

    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < n_atoms; i++) {
      int count = 0;
      for (int j = 0; j < n_atoms; j++) {
        if (i != j) {
          edge_index_vector[0][i * (n_atoms - 1) + count] = i;
          edge_index_vector[1][i * (n_atoms - 1) + count] = j;
          count++;
        }
      }
    }

    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < n_edges; i++) {
      distance_vector[i] = pbc_tools.distance(
        true,
        x_local[atom_list_active[edge_index_vector[0][i]]],
        x_local[atom_list_active[edge_index_vector[1][i]]]
      );
    }

    torch::Tensor distances = torch::from_blob(
      distance_vector.data(),
      n_edges,
      torch::TensorOptions().dtype(torch::kFloat32)
    );
    torch::Tensor senders = torch::from_blob(
      edge_index_vector[0].data(),
      n_edges,
      torch::TensorOptions().dtype(torch::kInt64)
    );
    torch::Tensor receivers = torch::from_blob(
      edge_index_vector[1].data(),
      n_edges,
      torch::TensorOptions().dtype(torch::kInt64)
    );

    const torch::Tensor mask = distances <= r_max;
    senders = senders.index({mask});
    receivers = receivers.index({mask});
    edge_index = torch::vstack({senders, receivers});
    n_edges = (int)edge_index.size(1);
  } else {
    n_edges = (int)neighbor_list->size() * 2;
    int n_pairs = (int)neighbor_list->size();
    std::vector<std::vector<int64_t>> edge_index_vector;
    edge_index_vector.resize(2, std::vector<int64_t>(n_edges));

    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < n_pairs; i++) {
        auto pair = neighbor_list->getClosePair(i);
        edge_index_vector[0][i] = pair.first;
        edge_index_vector[1][i] = pair.second;
        edge_index_vector[0][n_pairs + i] = pair.second;
        edge_index_vector[1][n_pairs + i] = pair.first;
    }

    torch::Tensor senders = torch::from_blob(
      edge_index_vector[0].data(),
      n_edges,
      torch::TensorOptions().dtype(torch::kInt64)
    );
    torch::Tensor receivers = torch::from_blob(
      edge_index_vector[1].data(),
      n_edges,
      torch::TensorOptions().dtype(torch::kInt64)
    );
    edge_index = torch::vstack({senders, receivers});
  }

  edge_index = edge_index.to(device);

  // edge shifts
  torch::Tensor shifts;
  torch::Tensor unit_shifts;

  if (pbc) {
    if (pbc_tools.isOrthorombic()) {
      auto deltas = (
        torch::index_select(positions, 0, edge_index[0])
        - torch::index_select(positions, 0, edge_index[1])
      );
      unit_shifts = torch::round(deltas / torch::diagonal(cell, 0));
      shifts = unit_shifts * torch::diagonal(cell, 0);
    } else {
      auto cell_inv = torch::linalg_pinv(cell.transpose(1, 0));
      auto positions_s = torch::matmul(
        cell_inv, positions.transpose(1, 0)
      );
      auto deltas = (
        torch::index_select(positions_s, 1, edge_index[0])
        - torch::index_select(positions_s, 1, edge_index[1])
      );
      unit_shifts = torch::round(deltas);
      shifts = torch::matmul(
        cell.transpose(1, 0), unit_shifts
      ).transpose(1, 0);
    }
  } else {
    shifts = torch::zeros({n_edges, 3}, torch_float_dtype);
    unit_shifts = torch::zeros({n_edges, 3}, torch_float_dtype);
  }

  shifts = shifts.to(device);
  unit_shifts = unit_shifts.to(device);

  // other things
  // TODO: some of these things are required by MACE. We should disable
  // some of them when not using MACE, maybe by distinguishing the MACE model.
  auto batch = torch::zeros({n_atoms}, torch::dtype(torch::kInt64));
  auto ptr = torch::empty({2}, torch::dtype(torch::kInt64));
  auto weight = torch::empty({1}, torch_float_dtype);
  ptr[0] = 0;
  ptr[1] = n_atoms;
  weight[0] = 1.0;

  // load data to device
  // TODO: some of these things are required by MACE. We should disable
  // some of them when not using MACE, maybe by distinguishing the MACE model.
  batch = batch.to(device);
  ptr = ptr.to(device);
  weight = weight.to(device);

  // pack the input, call the model
  // TODO: some of these things are required by MACE. We should disable
  // some of them when not using MACE, maybe by distinguishing the MACE model.
  c10::Dict<std::string, torch::Tensor> input;
  input.insert("batch", batch);
  input.insert("cell", cell);
  input.insert("edge_index", edge_index);
  input.insert("node_attrs", node_attrs);
  positions.set_requires_grad(true);
  input.insert("positions", positions);
  input.insert("ptr", ptr);
  input.insert("weight", weight);
  input.insert("shifts", shifts);
  input.insert("unit_shifts", unit_shifts);

  // create and insert system mask and n_system
  torch::Tensor system_masks = torch::zeros({n_atoms, 1}, torch::dtype(torch::kBool));
  for (size_t i = 0; i < atom_list_a.size(); i++)
    system_masks[i] = true;
  system_masks = system_masks.to(device);
  input.insert("system_masks", system_masks);

  torch::Tensor n_system = torch::ones({1, 1}, torch::dtype(torch::kInt64));
  n_system = n_system.to(device);
  n_system[0][0] = (int64_t)atom_list_a.size();
  input.insert("n_system", n_system);
  

  // TODO: figure out how to enable virials. Maybe we could port MACE's python
  // code to our python module.
  auto output = model.forward({input}).toTensor();

  // helper variables
  std::vector<PLMD::Vector> derivatives(n_atoms);
  auto grad_output = torch::ones({1}).expand({1, 1}).to(device);

  // Here we simply compute the output and its derivatives
  for (int i = 0; i < n_out; i++) {
    // set CV values
    string name_comp = "node-" + std::to_string(i);
    getPntrToComponent(name_comp)->set(output[0][i].cpu().item<double>());
    // set derivatives
    auto gradients = torch::autograd::grad(
      {output.slice(1, i, (i + 1))},
      {positions},
      {grad_output}, // grad_outputs
      true,          // retain_graph
      false          // create_graph
    )[0].cpu();
    #pragma omp parallel for num_threads(n_threads)
    for (int j = 0; j < n_atoms; j++) {
      derivatives[j][0] = gradients[j][0].item<double>() * to_ang;
      derivatives[j][1] = gradients[j][1].item<double>() * to_ang;
      derivatives[j][2] = gradients[j][2].item<double>() * to_ang;
    }
    #pragma omp parallel for num_threads(n_threads)
    for (int j = 0; j < n_atoms; j++) {
      int index = atom_list_active[j];
      setAtomsDerivatives(
        getPntrToComponent(name_comp), index, derivatives[j]
      );
  
    }
  }
}


int PytorchGNN::atomic_number_from_name(std::string name)
{
  std::transform(
    name.begin(),
    name.end(),
    name.begin(),
    [](unsigned char c){return std::tolower(c);}
  );
  auto iter = std::find(periodic_table.begin(), periodic_table.end(), name);
  if (iter == periodic_table.end())
    plumed_merror(
      "Can not find element name '" + name + "' from the periodic table!"
    );
  return std::distance(periodic_table.begin(), iter) + 1;
}

std::string PytorchGNN::model_summary(
    std::string model_name, torch::jit::Module module, int level_max, int level
) {
  std::stringstream ss;

  std::string model_type = module.type()->name()->name();
  ss << "  (" << model_name << "): " << model_type;

  if (module.named_children().size() != 0) {
    if (level <= level_max) {
      ss << " {\n";
      for (const torch::jit::NameModule& s : module.named_children())
          ss << torch::jit::jit_log_prefix(
              "  ",
              model_summary(s.name, s.value, level_max, level + 1)
          );
      ss << "  }\n";
    } else {
      ss << " { ... }";
    }
  } else {
    ss << "\n";
  }

  return ss.str();
}

bool PytorchGNN::groups_have_intersection(void) {
  std::vector<AtomNumber> intersections;
  std::vector<AtomNumber> atom_list_a_copy(atom_list_a);
  std::vector<AtomNumber> atom_list_b_copy(atom_list_b);

  std::sort(atom_list_a_copy.begin(), atom_list_a_copy.end());
  std::sort(atom_list_b_copy.begin(), atom_list_b_copy.end());

  std::set_intersection(
    atom_list_a_copy.begin(),
    atom_list_a_copy.end(),
    atom_list_b_copy.begin(),
    atom_list_b_copy.end(),
    back_inserter(intersections)
  );

  return intersections.size() > 0;
}

void PytorchGNN::find_active_atoms(int n_threads) {
  if (atom_list_b.size() > 0) {
    atom_list_active.clear();
    std::vector<int> neighbors(neighbor_list->size());

    #pragma omp parallel for num_threads(n_threads)
    for (size_t i = 0; i < neighbor_list->size(); i++)
      neighbors[i] = neighbor_list->getClosePair(i).second;

    // TODO: make this faster
    std::unordered_set<int> neighbors_set;
    for (int i : neighbors)
        neighbors_set.insert(i);
    neighbors.assign(neighbors_set.begin(), neighbors_set.end());

    for (size_t i = 0; i < atom_list_a.size(); i++)
      atom_list_active.push_back(i);
    for (size_t i = 0; i < neighbors.size(); i++)
      atom_list_active.push_back(neighbors[i]);
  } else if (atom_list_active.size() == 0) {
    atom_list_active.clear();

    for (size_t i = 0; i < atom_list_a.size(); i++)
      atom_list_active.push_back(i);
  }
}

} // pytorch_gnn

} // colvar

} // PLMD

#endif // PLUMED_HAS_LIBTORCH
