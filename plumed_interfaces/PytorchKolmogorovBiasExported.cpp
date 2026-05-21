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

//+PLUMEDOC PYTORCH_FUNCTION PYTORCH_KOLMOGOROV_BIAS_EXPORTED
/*
Load an AOTInductor-compiled PyTorch feed-forward neural-network model that returns latent CVs and a Kolmogorov bias.

This action expects a model exported with mlcolvar.utils.export.export() using calculate_gradients=True and k_bias_options enabled.

The exported model must return four tensors:

- outputs[0]: model outputs, with shape [1, 2], corresponding to [z, q];
- outputs[1]: derivatives of [z, q] with respect to ARG,
              with shape [2, 1, n_inputs];
- outputs[2]: Kolmogorov bias value, kbias;
- outputs[3]: derivative of kbias with respect to ARG.

The output components are:

- z
- q
- kbias

Example:

\plumedfile
d1: DISTANCE ATOMS=1,2
d2: DISTANCE ATOMS=2,3

model: PYTORCH_KOLMOGOROV_BIAS_EXPORTED FILE=model.pt2 ARG=d1,d2

PRINT FILE=COLVAR ARG=model.z,model.q,model.kbias STRIDE=100
\endplumedfile

*/
//+ENDPLUMEDOC


class PytorchKolmogorovBiasExported :
  public Function
{
  unsigned _n_in = 0;

  std::string _fname = "model.pt2";

  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> _model;

  torch::ScalarType torch_float_dtype = torch::kFloat32;
  torch::Device device = torch::kCPU;

public:
  explicit PytorchKolmogorovBiasExported(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);
};


PLUMED_REGISTER_ACTION(
  PytorchKolmogorovBiasExported,
  "PYTORCH_KOLMOGOROV_BIAS_EXPORTED"
)


void PytorchKolmogorovBiasExported::registerKeywords(Keywords& keys) {
  Function::registerKeywords(keys);

  keys.use("ARG");

  keys.add(
    "optional",
    "FILE",
    "Filename of the AOTInductor-compiled PyTorch model"
  );

  keys.addOutputComponent(
    "z",
    "default",
    "Latent coordinate z"
  );

  keys.addOutputComponent(
    "q",
    "default",
    "Committor value q"
  );

  keys.addOutputComponent(
    "kbias",
    "default",
    "Kolmogorov bias potential"
  );
}


PytorchKolmogorovBiasExported::PytorchKolmogorovBiasExported(
  const ActionOptions& ao
):
  Action(ao),
  Function(ao)
{
  // Print LibTorch version.
  std::stringstream ss;
  ss << TORCH_VERSION_MAJOR << "."
     << TORCH_VERSION_MINOR << "."
     << TORCH_VERSION_PATCH;

  std::string version;
  ss >> version;

  log.printf(("  LibTorch version: " + version + "\n").data());

  // Number of PLUMED arguments.
  _n_in = getNumberOfArguments();

  // Parse model file.
  parse("FILE", _fname);
  checkRead();

  // Load AOTInductor package.
  try {
    _model = std::make_unique<torch::inductor::AOTIModelPackageLoader>(_fname);
  }
  catch (const c10::Error& e) {
    plumed_merror("Cannot load exported model file: '" + _fname + "'. Reason: " + std::string(e.what()));
  }

  // Read metadata.
  auto metadata = _model->get_metadata();

  // Check metadata.
  if (metadata.at("model_type") != "ffnn")
    plumed_merror("This interface expects model_type='ffnn'.");

  if (metadata.at("calculate_k_bias") != "True")
    plumed_merror("This interface requires calculate_k_bias=True.");

  if (metadata.at("calculate_gradients") != "True")
    plumed_merror("This interface requires calculate_gradients=True.");

  if (std::atoi(metadata.at("n_cvs").c_str()) != 2)
    plumed_merror("This interface expects n_cvs=2, corresponding to [z, q].");

  // Dtype.
  std::string float_dtype_exported(metadata.at("float_dtype").c_str());

  if (float_dtype_exported == "32")
    torch_float_dtype = torch::kFloat32;
  else if (float_dtype_exported == "64")
    torch_float_dtype = torch::kFloat64;
  else
    plumed_merror("Unknown float dtype in exported model.");

  // Device.
  bool use_cuda = false;
  std::string device_exported(metadata.at("AOTI_DEVICE_KEY").c_str());

  if (device_exported == "cuda") {
    if (!torch::cuda::is_available())
      plumed_merror("Exported model requires CUDA, but CUDA is not available.");

    device = torch::kCUDA;
    use_cuda = true;
  }
  else {
    device = torch::kCPU;
    use_cuda = false;
  }

  // Dummy forward pass.
  log.printf("  Checking output dimensions:\n");

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
      "Please check ARG, input dimension, dtype, and exported model signature. "
      "Reason: " + std::string(e.what())
    );
  }

  if (outputs.size() != 4) {
    plumed_merror(
      "The exported AOT model must return four tensors: "
      "CV values, CV gradients, kbias value, and kbias gradients."
    );
  }

  torch::Tensor cv = outputs[0];
  torch::Tensor grad = outputs[1];
  torch::Tensor kbias = outputs[2];
  torch::Tensor grad_b = outputs[3];

  // Create components.
  addComponentWithDerivatives("z");
  componentIsNotPeriodic("z");

  addComponentWithDerivatives("q");
  componentIsNotPeriodic("q");

  addComponentWithDerivatives("kbias");
  componentIsNotPeriodic("kbias");

  // Print log.
  log.printf("  Number of inputs: %d \n", _n_in);
  log.printf("  Number of model CVs: 2, interpreted as [z, q]\n");
  log.printf("  Additional output: kbias\n");
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

  if (metadata.find("lambd") != metadata.end()) {
    std::string lambd(metadata.at("lambd").c_str());
    log << "  Kolmogorov bias lambda: " + lambd + "\n";
  }

  if (metadata.find("epsilon") != metadata.end()) {
    std::string epsilon(metadata.at("epsilon").c_str());
    log << "  Kolmogorov bias epsilon: " + epsilon + "\n";
  }

  if (metadata.find("beta") != metadata.end()) {
    std::string beta(metadata.at("beta").c_str());
    log << "  Kolmogorov bias beta: " + beta + "\n";
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

  log.printf("  Output alignment:\n");
  log.printf("    z      <- outputs[0][0][0]\n");
  log.printf("    q      <- outputs[0][0][1]\n");
  log.printf("    kbias  <- outputs[2]\n");

  log.printf("  Bibliography: ");
  log<<plumed.cite("Bonati, Rizzi, and Parrinello, J. Phys. Chem. Lett. 11, 2998-3004 (2020)");
  log<<plumed.cite("Bonati, Trizio, Rizzi, and Parrinello, J. Chem. Phys. 159, 014801 (2023)");
  log<<plumed.cite("Kang, Trizio, and Parrinello, Nat. Comp. Sci. 4, 451-460 (2024)");
  log<<plumed.cite("Trizio, Kang and Parrinello, Nat. Comp. Sci. 5, 582-591 (2025)");
  log<<plumed.cite("Trizio, Rossi and Parrinello, J. Chem. Phys. (2026)");
  log.printf("\n");
}


void PytorchKolmogorovBiasExported::calculate() {

  // Build input tensor in the dtype expected by the exported model.
  // Important: for float32 exported models, never create a double input.
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
      "Error while running the exported AOT Kolmogorov-bias model. Reason: " +
      std::string(e.what())
    );
  }

  if (outputs.size() != 4) {
    plumed_merror(
      "The exported AOT model must return four tensors: "
      "CV values, CV gradients, kbias value, and kbias gradients."
    );
  }

  torch::Tensor cv = outputs[0].contiguous().to(torch::kCPU);
  torch::Tensor grad = outputs[1].contiguous().to(torch::kCPU);
  torch::Tensor kbias = outputs[2].contiguous().to(torch::kCPU);
  torch::Tensor grad_b = outputs[3].contiguous().to(torch::kCPU).reshape({-1});

  // ======================================================
  // 1. z value and derivatives
  // ======================================================

  Value* value_z = getPntrToComponent("z");

  double z_value = cv.index({0, 0}).item<double>();
  value_z->set(z_value);

  for (unsigned i = 0; i < _n_in; i++) {
    double dz_darg = grad.index({0, 0, (long)i}).item<double>();
    setDerivative(value_z, i, dz_darg);
  }

  // ======================================================
  // 2. q value and derivatives
  // ======================================================

  Value* value_q = getPntrToComponent("q");

  double q_value = cv.index({0, 1}).item<double>();
  value_q->set(q_value);

  for (unsigned i = 0; i < _n_in; i++) {
    double dq_darg = grad.index({1, 0, (long)i}).item<double>();
    setDerivative(value_q, i, dq_darg);
  }

  // ======================================================
  // 3. kbias value and derivatives
  // ======================================================

  Value* value_b = getPntrToComponent("kbias");

  double kbias_value = kbias.reshape({-1}).index({0}).item<double>();
  value_b->set(kbias_value);

  for (unsigned i = 0; i < _n_in; i++) {
    double dkbias_darg = grad_b.index({(long)i}).item<double>();
    setDerivative(value_b, i, dkbias_darg);
  }
}


} // pytorch
} // function
} // PLMD

#endif // __PLUMED_HAS_LIBTORCH