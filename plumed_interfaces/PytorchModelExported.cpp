/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (c) 2026 of Luigi Bonati and Enrico Trizio.

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
// convert LibTorch version to string
//#define STRINGIFY(x) #x
//#define TOSTR(x) STRINGIFY(x)
//#define LIBTORCH_VERSION TO_STR(TORCH_VERSION_MAJOR) "." TO_STR(TORCH_VERSION_MINOR) "." TO_STR(TORCH_VERSION_PATCH)

#include "core/PlumedMain.h"
#include "function/Function.h"
#include "function/ActionRegister.h"

#include <torch/torch.h>
#include <torch/script.h>

#if __has_include("torch/csrc/inductor/aoti_package/model_package_loader.h")
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#else
#error Can not find header file: <torch/csrc/inductor/aoti_package/model_package_loader.h> ! Is your LibTorch too old?
#endif

#include <fstream>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>
#include <sstream>


using namespace std;

namespace PLMD {
namespace function {
namespace pytorch {

//+PLUMEDOC PYTORCH_FUNCTION PYTORCH_MODEL_EXPORTED
/*
Load an AOTInductor-compiled PyTorch feed-forward neural network exported with `mlcolvar.utils.export.export()`.

This action is the AOTInductor counterpart of `PYTORCH_MODEL`. In contrast to the TorchScript-based interface, the derivatives are not computed through LibTorch autograd at runtime. Instead, the exported model must already return
both the model outputs and their derivatives with respect to the input arguments.

The exported model must return four tensors:

- the model output, with shape `[1, n_cvs]`;
- the derivatives of the model output with respect to the input arguments, with shape `[n_cvs, 1, n_inputs]`;
- a third tensor reserved for Kolmogorov-bias models;
- a fourth tensor reserved for Kolmogorov-bias gradients.

For ordinary feed-forward neural-network CVs, only the first two tensors are used by this action. The model must be exported with `calculate_gradients=True` and without `k_bias_options`.

The output components are exposed as `node-0`, `node-1`, etc. For example, if the action is labeled `model`, the first output can be referenced as `model.node-0`.

By default, the model file is assumed to be `model.pt2`, unless otherwise specified using the `FILE` keyword. The floating-point precision and running device are fixed at export time.

Note that this action requires the LibTorch C++ library with AOTInductor package support.

\par Examples

Load an AOTInductor model called `model.pt2` that takes two dihedral angles as input and returns two output components.

\plumedfile
phi: TORSION ATOMS=5,7,9,15
psi: TORSION ATOMS=7,9,15,17

model: PYTORCH_MODEL_EXPORTED FILE=model.pt2 ARG=phi,psi

PRINT FILE=COLVAR ARG=phi,psi,model.node-0,model.node-1 STRIDE=100
\endplumedfile

*/
//+ENDPLUMEDOC



class PytorchModelExported :
  public Function
{
  unsigned _n_in = 0;
  unsigned _n_out = 0;

  std::string _fname = "model.pt2";

  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> _model;
  
  torch::ScalarType torch_float_dtype = torch::kFloat32;
  torch::Device device = torch::kCPU;

public:
  explicit PytorchModelExported(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);
};

// This is the macro that registers the action in plumed. 
// The name is arbitrary, but it has to be unique among all the actions in plumed.
// Thus, if this is loaded at runtime with a the pytorch module already present
// it needs to be registered with a different name, e.g., PYTORCH_MODEL_RUNTIME
PLUMED_REGISTER_ACTION(PytorchModelExported,"PYTORCH_MODEL_EXPORTED")

void PytorchModelExported::registerKeywords(Keywords& keys) {
  Function::registerKeywords(keys);
  keys.use("ARG");
  keys.add("optional","FILE","Filename of the PyTorch compiled model");
  keys.addOutputComponent("node", "default", "Model outputs");
}

PytorchModelExported::PytorchModelExported(const ActionOptions&ao):
  Action(ao),
  Function(ao)
{
  // print libtorch version
  std::stringstream ss;
  ss << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH;
  std::string version;
  ss >> version; // extract into the string.
  log.printf(("  LibTorch version: "+version+"\n").data());

  //number of inputs of the model
  _n_in=getNumberOfArguments();

  //parse model name
  parse("FILE", _fname);
  checkRead();

  try {
    _model = std::make_unique<torch::inductor::AOTIModelPackageLoader>(_fname);
  }
  catch (const c10::Error& e) {
    std::ifstream infile(_fname);
    bool exist = infile.good();
    infile.close();

    if (exist) {
      plumed_merror(
        "Cannot load FILE: '" + _fname +
        "'. Please check that it is an AOTInductor-compiled PyTorch package. "
        "Reason: " + std::string(e.what())
      );
    }
    else {
      plumed_merror("The FILE: '" + _fname + "' does not exist.");
    }
  }

  // Read metadata.
  auto metadata = _model->get_metadata();

  // Check model type.
  if (metadata.find("model_type") != metadata.end()) {
    std::string model_type(metadata.at("model_type").c_str());

    if (model_type != "ffnn") {
      plumed_merror(
        "PYTORCH_MODEL_EXPORTED expects a feed-forward neural-network model "
        "with metadata model_type='ffnn', but found model_type='" +
        model_type + "'."
      );
    }
  }

  // Check gradients.
  if (metadata.find("calculate_gradients") != metadata.end()) {
    std::string calculate_gradients(metadata.at("calculate_gradients").c_str());

    if (calculate_gradients != "True" && calculate_gradients != "1") {
      plumed_merror(
        "PYTORCH_MODEL_EXPORTED requires a model exported with "
        "calculate_gradients=True."
      );
    }
  }

  // Dtype.
  if (metadata.find("float_dtype") != metadata.end()) {
    std::string float_dtype_exported(metadata.at("float_dtype").c_str());

    if (float_dtype_exported == "32") {
      torch_float_dtype = torch::kFloat32;
    }
    else if (float_dtype_exported == "64") {
      torch_float_dtype = torch::kFloat64;
    }
    else {
      plumed_merror(
        "Unknown float dtype '" + float_dtype_exported +
        "' found in exported model '" + _fname + "'."
      );
    }
  }

  // Device.
  bool use_cuda = false;

  if (metadata.find("AOTI_DEVICE_KEY") != metadata.end()) {
    std::string device_exported(metadata.at("AOTI_DEVICE_KEY").c_str());

    if (device_exported == "cuda") {
      if (!torch::cuda::is_available()) {
        plumed_merror(
          "Exported model '" + _fname +
          "' requires CUDA, but CUDA is not available in this LibTorch build."
        );
      }

      device = torch::kCUDA;
      use_cuda = true;
    }
    else {
      device = torch::kCPU;
      use_cuda = false;
    }
  }

  // Number of CVs from metadata if available.
  if (metadata.find("n_cvs") != metadata.end()) {
    _n_out = std::atoi(metadata.at("n_cvs").c_str());
  }

  // Run a dummy forward pass to check output shapes.
  log.printf("  Checking output dimension:\n");

  std::vector<float> input_test(_n_in, 0.0f);

  torch::Tensor single_input = torch::from_blob(
    input_test.data(),
    {(long)1, (long)_n_in},
    torch::TensorOptions().dtype(torch::kFloat32)
  ).clone();

  single_input = single_input.to(device).to(torch_float_dtype);

  std::vector<torch::Tensor> inputs;
  inputs.push_back(single_input);

  std::vector<torch::Tensor> outputs;

  try {
    outputs = _model->run(inputs);
  }
  catch (const c10::Error& e) {
    plumed_merror(
      "Error while running the exported model on a test input. "
      "Please check the number of ARG entries and the exported model signature. "
      "Reason: " + std::string(e.what())
    );
  }

  torch::Tensor cv = outputs[0];
  torch::Tensor grad = outputs[1];

  unsigned n_out_from_output = static_cast<unsigned>(cv.size(1));

  if (_n_out == 0) {
    _n_out = n_out_from_output;
  }
  else if (_n_out != n_out_from_output) {
    plumed_merror(
      "The number of CVs stored in metadata does not match the model output shape."
    );
  }

  // Create output components.
  for (unsigned j = 0; j < _n_out; j++) {
    std::string name_comp = "node-" + std::to_string(j);
    addComponentWithDerivatives(name_comp);
    componentIsNotPeriodic(name_comp);
  }

  // Print log.
  log.printf("  Number of inputs: %d \n", _n_in);
  log.printf("  Number of outputs: %d \n", _n_out);
  log.printf("  Model file: %s \n", _fname.c_str());

  log.printf("  Will run on device: ");
  if (use_cuda) {
    log.printf("CUDA\n");
  }
  else {
    log.printf("CPU\n");
  }

  log.printf("  Floating-point precision: ");
  if (torch_float_dtype == torch::kFloat64) {
    log.printf("float64\n");
  }
  else {
    log.printf("float32\n");
  }

  if (metadata.find("n_parameters") != metadata.end()) {
    std::string n_parameters(metadata.at("n_parameters").c_str());
    log << "  Model parameters: " + n_parameters + "\n";
  }

  if (metadata.find("model_summary") != metadata.end()) {
    std::string model_summary(metadata.at("model_summary").c_str());
    log << "  Model architecture:\n";
    log << model_summary;
  }

  log.printf("  Bibliography: ");
  log << plumed.cite("Bonati, Rizzi and Parrinello, J. Phys. Chem. Lett. 11, 2998-3004 (2020)");
  log << plumed.cite("Bonati, Trizio, Rizzi and Parrinello, J. Chem. Phys. 159, 014801 (2023)");
  log.printf("\n");
}


void PytorchModelExported::calculate() {

  std::vector<float> current_S(_n_in);

  for (unsigned i = 0; i < _n_in; i++) {
    current_S[i] = static_cast<float>(getArgument(i));
  }

  torch::Tensor input_S = torch::from_blob(
    current_S.data(),
    {(long)1, (long)_n_in},
    torch::TensorOptions().dtype(torch::kFloat32)
  ).clone();

  input_S = input_S.to(device).to(torch_float_dtype);

  std::vector<torch::Tensor> inputs;
  inputs.push_back(input_S);

  std::vector<torch::Tensor> outputs;

  try {
    outputs = _model->run(inputs);
  }
  catch (const c10::Error& e) {
    plumed_merror(
      "Error while running the exported AOT model. Reason: " +
      std::string(e.what())
    );
  }

  if (outputs.size() != 4) {
    plumed_merror(
      "The exported AOT model must return four tensors: "
      "CV values, CV gradients, KBias value, and KBias gradients."
    );
  }

  torch::Tensor cv = outputs[0].contiguous().to(torch::kCPU);
  torch::Tensor grad = outputs[1].contiguous().to(torch::kCPU);

  for (unsigned j = 0; j < _n_out; j++) {
    std::string name_comp = "node-" + std::to_string(j);
    Value* value = getPntrToComponent(name_comp);

    double cv_j = cv.index({0, (long)j}).item<double>();
    value->set(cv_j);

    for (unsigned i = 0; i < _n_in; i++) {
      // Python exporter convention for FFNN:
      // gradients.shape = [n_cvs, 1, n_inputs]
      double dcv_darg = grad.index({(long)j, 0, (long)i}).item<double>();
      setDerivative(value, i, dcv_darg);
    }
  }
}


} //PLMD
} //function
} //pytorch

#endif //PLUMED_HAS_LIBTORCH
