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

// #ifdef __PLUMED_HAS_LIBTORCH
// convert LibTorch version to string
//#define STRINGIFY(x) #x
//#define TOSTR(x) STRINGIFY(x)
//#define LIBTORCH_VERSION TO_STR(TORCH_VERSION_MAJOR) "." TO_STR(TORCH_VERSION_MINOR) "." TO_STR(TORCH_VERSION_PATCH)

#include "core/PlumedMain.h"
#include "function/Function.h"
#include "function/ActionRegister.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <fstream>
#include <cmath>


// We have to do a backward compatability hack for <1.10
// https://discuss.pytorch.org/t/how-to-check-libtorch-version/77709/4
// Basically, the check in torch::jit::freeze
// (see https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp#L479)
// is wrong, and we have ro "reimplement" the function
// to get around that...
// it's broken in 1.8 and 1.9
// BUT the internal logic in the function is wrong in 1.10
// So we only use torch::jit::freeze in >=1.11
// credits for this implementation of the hack to the NequIP guys
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
  #define DO_TORCH_FREEZE_HACK
  // For the hack, need more headers:
  #include <torch/csrc/jit/passes/freeze_module.h>
  #include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#endif

using namespace std;

namespace PLMD {
namespace function {
namespace pytorch {

class PytorchModelBias :
  public Function
{
  unsigned _n_in;
  unsigned _n_out;
  double lambda;
  double epsilon;
  double sigmoid_p;
  double beta;
  bool use_q_for_bias;
  double lambda_over_beta;
  double log_epsilon;
  torch::jit::Module _model;
  torch::Device device = torch::kCPU;
  torch::Tensor t_sigmoid_p;
  torch::Tensor t_epsilon;
  torch::Tensor t_grad_output;
  torch::Tensor t_grad_output2;
public:
  explicit PytorchModelBias(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);

  std::vector<float> tensor_to_vector(const torch::Tensor& x);
};

PLUMED_REGISTER_ACTION(PytorchModelBias,"PYTORCH_MODEL_BIAS")

void PytorchModelBias::registerKeywords(Keywords& keys) {
  Function::registerKeywords(keys);
  keys.use("ARG");
  keys.add("compulsory","LAMBDA","Prefactor of the bias");
  keys.add("compulsory","BETA","Inverse temperature in the right energy units");
  keys.add("optional","FILE","Filename of the PyTorch compiled model");
  keys.add("optional","EPSILON","Numerical regularization term in the logarithm, default 1e-6");
  keys.add("optional","SIGMOID_P","Hyperameters p in the sigma activation for committor models, default 3");
  keys.addFlag("USE_Q_FOR_BIAS", false, "USe the activated output for the bias calculation, may kill small gradients, default false");
  keys.addOutputComponent("z", "default", "Model outputs");
  keys.addOutputComponent("q", "default", "Model outputs");
  keys.addOutputComponent("kbias", "default", "Model outputs");
}

std::vector<float> PytorchModelBias::tensor_to_vector(const torch::Tensor& x) {
  return std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
}

PytorchModelBias::PytorchModelBias(const ActionOptions&ao):
  Action(ao),
  Function(ao)
{ //print pytorch version

  //number of inputs of the model
  _n_in=getNumberOfArguments();

  //parse model name
  std::string fname="model.ptc";
  parse("FILE",fname);

  //parse params
  parse("LAMBDA", lambda);
  parse("BETA", beta);

  //parse params
  sigmoid_p = 3.0;
  parse("SIGMOID_P", sigmoid_p);

  epsilon = 1e-6;
  parse("EPSILON", epsilon);

  use_q_for_bias = false;
  parseFlag("USE_Q_FOR_BIAS", use_q_for_bias);

  // Cache constants used at every step to avoid repeated allocations.
  lambda_over_beta = lambda / beta;
  log_epsilon = std::log(epsilon);
  t_sigmoid_p = torch::tensor(sigmoid_p).to(device);
  t_epsilon = torch::tensor(epsilon).to(device);
  t_grad_output = torch::ones({1}).expand({1, 1}).to(device);
  t_grad_output2 = torch::ones({1}).to(device);

  // we create the metatdata dict 
  std::unordered_map<std::string, std::string> metadata = {
    {"_jit_bailout_depth", ""},
    {"_jit_fusion_strategy", ""}
  };

  //deserialize the model from file
  try {
    _model = torch::jit::load(fname, device, metadata);
  } 

  //if an error is thrown check if the file exists or not
  catch (const c10::Error& e) {
    std::ifstream infile(fname);
    bool exist = infile.good();
    infile.close();
    if (exist) {
      // print libtorch version
      std::stringstream ss;
      ss << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH;
      std::string version;
      ss >> version; // extract into the string.
      plumed_merror("Cannot load FILE: '"+fname+"'. Please check that it is a Pytorch compiled model (exported with 'torch.jit.trace' or 'torch.jit.script') and that the Pytorch version matches the LibTorch one ("+version+").");
    }
    else {
      plumed_merror("The FILE: '"+fname+"' does not exist.");
    }
  }
  checkRead();

  // Optimize model 
  _model.eval();
  #ifdef DO_TORCH_FREEZE_HACK
      // Do the hack
      // Copied from the implementation of torch::jit::freeze,
      // except without the broken check
      // See https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp
      bool optimize_numerics = true;  // the default
      // the {} is preserved_attrs
      auto out_mod = torch::jit::freeze_module(
        _model, {}
      );
      // See 1.11 bugfix in https://github.com/pytorch/pytorch/pull/71436
      auto graph = out_mod.get_method("forward").graph();
      OptimizeFrozenGraph(graph, optimize_numerics);
      _model = out_mod;
    #else
      // Do it normally
     _model = torch::jit::freeze(_model);
    #endif

  #if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
    // Set JIT bailout to avoid long recompilations for many steps
    size_t jit_bailout_depth;
    if (metadata["_jit_bailout_depth"].empty()) {
      // This is the default used in the Python code
      jit_bailout_depth = 1;
    } else {
      jit_bailout_depth = std::stoi(metadata["_jit_bailout_depth"]);
    }
    torch::jit::getBailoutDepth() = jit_bailout_depth;
  #else
    // In PyTorch >=1.11, this is now set_fusion_strategy
    torch::jit::FusionStrategy strategy;
    if (metadata["_jit_fusion_strategy"].empty()) {
      // This is the default used in the Python code
      strategy = {{torch::jit::FusionBehavior::DYNAMIC, 0}};
    } else {
      std::stringstream strat_stream(metadata["_jit_fusion_strategy"]);
      std::string fusion_type, fusion_depth;
      while(std::getline(strat_stream, fusion_type, ',')) {
        std::getline(strat_stream, fusion_depth, ';');
        strategy.push_back({fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC : torch::jit::FusionBehavior::DYNAMIC, std::stoi(fusion_depth)});
      }
    }
    torch::jit::setFusionStrategy(strategy);
  #endif

// TODO check torch::jit::optimize_for_inference() for more complex models
// This could speed up the code, it was not available on LTS 
  #if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 10)
  _model = torch::jit::optimize_for_inference(_model);
  #endif
// END -> Optimize model 


 //check the dimension of the output
  log.printf("Checking output dimension:\n");
  std::vector<float> input_test (_n_in);
  torch::Tensor single_input = torch::tensor(input_test).view({1,_n_in});
  single_input = single_input.to(device);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back( single_input );
  torch::Tensor output = _model.forward( inputs ).toTensor();
  vector<float> cvs = this->tensor_to_vector (output);
  _n_out=cvs.size();
  if(_n_out!=1) {
    plumed_merror("PYTORCH_MODEL_BIAS expects a model with a single scalar output, but got "+std::to_string(_n_out)+" outputs.");
  }

//create components of output  
string name_comp = "z";
addComponentWithDerivatives( name_comp );
componentIsNotPeriodic( name_comp );
  
name_comp = "q";
addComponentWithDerivatives( name_comp );
componentIsNotPeriodic( name_comp );
  
name_comp = "kbias";
addComponentWithDerivatives( name_comp );
componentIsNotPeriodic( name_comp );


  //print log
  //log.printf("Pytorch Model Loaded: %s \n",fname);
  log.printf("Number of input: %d \n",_n_in);
  log.printf("Number of outputs: %d \n",_n_out);
  log.printf("  Bibliography: ");
  log<<plumed.cite("Bonati, Rizzi, and Parrinello, J. Phys. Chem. Lett. 11, 2998-3004 (2020)");
  log<<plumed.cite("Bonati, Trizio, Rizzi, and Parrinello, J. Chem. Phys. 159, 014801 (2023)");
  log<<plumed.cite("Kang, Trizio, and Parrinello, Nat. Comp. Sci. 4, 451-460 (2024)");
  log<<plumed.cite("Trizio, Kang and Parrinello, Nat. Comp. Sci. 5, 582-591 (2025)");
  log.printf("\n");
}


void PytorchModelBias::calculate() {

// retrieve arguments
vector<float> current_S(_n_in);
for(unsigned i=0; i<_n_in; i++)
  current_S[i]=getArgument(i);
//convert to tensor
torch::Tensor input_S = torch::tensor(current_S).view({1,_n_in}).to(device);
input_S.set_requires_grad(true);
//convert to Ivalue
std::vector<torch::jit::IValue> inputs;
inputs.push_back( input_S );
//calculate output
torch::Tensor output_z = _model.forward( inputs ).toTensor();
torch::Tensor output_q = torch::sigmoid(t_sigmoid_p * output_z);
torch::Tensor one_minus_q = 1 - output_q;
torch::Tensor sigmoid_prime = t_sigmoid_p * output_q * one_minus_q;

// for(unsigned j=0; j<_n_out; j++) {  --> TODO maybe fix for more dimensions

// compute gradients of CV 
// get derivatives of z
torch::Tensor gradient_z = torch::autograd::grad({output_z},
                      {input_S},
    /*grad_outputs=*/ {t_grad_output},
    /*retain_graph=*/true,
    /*create_graph=*/true)[0]; // the [0] is to get a tensor and not a vector<at::tensor>

// get derivatives of q using the derivatives of sigmoid
torch::Tensor gradient_q = gradient_z * sigmoid_prime;

// compute bias from gradients
torch::Tensor log_grad_sq;
if (use_q_for_bias) // standard definition
  log_grad_sq = torch::log( torch::sum(gradient_q * gradient_q) + t_epsilon );
else // change of variable from q to z
  log_grad_sq = torch::log(torch::sum(gradient_z * gradient_z) * torch::pow(sigmoid_prime.squeeze(), 2) + t_epsilon);

log_grad_sq = -lambda_over_beta * ( log_grad_sq - log_epsilon );

// get derivatives of bias --> forces
torch::Tensor gradient2 = torch::autograd::grad({log_grad_sq},
                        {input_S},
      /*grad_outputs=*/ {t_grad_output2},
      /*retain_graph=*/false,
      /*create_graph=*/false)[0]; // the [0] is to get a tensor and not a vector<at::tensor>

// we set the derivatives for plumed
vector<float> der_q = this->tensor_to_vector ( gradient_q );  
string name_comp = "q";
    //set derivatives of component j
    for(unsigned i=0; i<_n_in; i++)
      setDerivative( getPntrToComponent(name_comp), i, der_q[i] ); 

vector<float> der_z = this->tensor_to_vector ( gradient_z );  
name_comp = "z";
    //set derivatives of component j
    for(unsigned i=0; i<_n_in; i++)
      setDerivative( getPntrToComponent(name_comp), i, der_z[i] ); 

vector<float> der2 = this->tensor_to_vector ( gradient2 );  
name_comp = "kbias";
    //set derivatives of component j
    for(unsigned i=0; i<_n_in; i++)
      setDerivative( getPntrToComponent(name_comp), i, der2[i] ); 

//set CV values
vector<float> cvs_q = this->tensor_to_vector (output_q);
name_comp = "q";
getPntrToComponent(name_comp)->set(cvs_q[0]);

vector<float> cvs_z = this->tensor_to_vector (output_z);
name_comp = "z";
getPntrToComponent(name_comp)->set(cvs_z[0]);

// set BIAS value
vector<float> bias = this->tensor_to_vector (log_grad_sq);
name_comp = "kbias";
getPntrToComponent(name_comp)->set(bias[0]);

}


} //pytorch
} //function
} //PLMD

// #endif //PLUMED_HAS_LIBTORCH
