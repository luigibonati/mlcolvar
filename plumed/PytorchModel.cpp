/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   PLUMED2 - Pytorch interface.

   Author: Luigi Bonati - https://github.com/luigibonati

   Please read and cite: 
     "Data-Driven Collective Variables for Enhanced Sampling"
     Bonati, Rizzi and Parrinello - JPCL (2020)

 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#ifdef __PLUMED_HAS_LIBTORCH

#include "core/PlumedMain.h"
#include "function/Function.h"
#include "function/ActionRegister.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <cmath>

using namespace std;

std::vector<float> tensor_to_vector(const torch::Tensor& x) {
    return std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
}

namespace PLMD {
namespace function {

//+PLUMEDOC FUNCTION PYTORCH MODEL
/*
Load a model trained with Pytorch. The derivatives are set using native backpropagation in Pytorch.

\par Examples
Define a model that takes as inputs two distances d1 and d2 

\plumedfile
model: PYTORCH_MODEL MODEL=model.pt ARG=d1,d2
\endplumedfile

The N nodes of the neural network are saved as "model.node-0", "model.node-1", ..., "model.node-(N-1)".

*/
//+ENDPLUMEDOC


class PytorchModel :
  public Function
{
  unsigned _n_in;
  unsigned _n_out;
  torch::jit::script::Module _model;
public:
  explicit PytorchModel(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);
};


PLUMED_REGISTER_ACTION(PytorchModel,"PYTORCH_MODEL")

void PytorchModel::registerKeywords(Keywords& keys) {
  Function::registerKeywords(keys);
  keys.use("ARG");
  keys.add("optional","MODEL","filename of the trained model"); 
  keys.addOutputComponent("node", "default", "NN outputs"); 
}

PytorchModel::PytorchModel(const ActionOptions&ao):
  Action(ao),
  Function(ao)
{
  //number of inputs of the model
  _n_in=getNumberOfArguments();

  //parse model name
  std::string fname="model.pt";
  parse("MODEL",fname); 
 
  //deserialize the model from file
  try {
    _model = torch::jit::load(fname);
  }
  catch (const c10::Error& e) {
    error("Cannot load Pytorch model. Check that the model is present and that the version of Pytorch is compatible with the Libtorch linked to PLUMED.");    
  }

  checkRead();

  //check the dimension of the output
  log.printf("Checking output dimension:\n");

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back( single_input );
  torch::Tensor output = _model.forward( inputs ).toTensor(); 
  vector<float> cvs = tensor_to_vector (output);
  _n_out=cvs.size();

  //create components
  for(unsigned j=0; j<_n_out; j++){
    string name_comp = "node-"+std::to_string(j);
    addComponentWithDerivatives( name_comp );
    componentIsNotPeriodic( name_comp );
  }
 
  //print log
  //log.printf("Pytorch Model Loaded: %s \n",fname);
  log.printf("Number of input: %d \n",_n_in); 
  log.printf("Number of outputs: %d \n",_n_out); 
  log.printf("  Bibliography: ");
  log<<plumed.cite("Luigi Bonati, Valerio Rizzi, and Michele Parrinello, J. Phys. Chem. Lett. 11, 2998-3004 (2020)");
  log.printf("\n");

}

void PytorchModel::calculate() {

  //retrieve arguments
  vector<float> current_S(_n_in);
  for(unsigned i=0; i<_n_in; i++)
    current_S[i]=getArgument(i);
  //convert to tensor
  torch::Tensor input_S = torch::tensor(current_S).view({1,_n_in});
  input_S.set_requires_grad(true);
  //convert to Ivalue
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back( input_S );
  //calculate output
  torch::Tensor output = _model.forward( inputs ).toTensor();  
  //set CV values
  vector<float> cvs = tensor_to_vector (output);
  for(unsigned j=0; j<_n_out; j++){
    string name_comp = "node-"+std::to_string(j);
    getPntrToComponent(name_comp)->set(cvs[j]);
  }
  //derivatives
  for(unsigned j=0; j<_n_out; j++){
   //backpropagation
    output[0][j].backward();
    //convert to vector
    vector<float> der = tensor_to_vector (input_S.grad() );
    string name_comp = "node-"+std::to_string(j);
    //set derivatives of component j
    for(unsigned i=0; i<_n_in; i++)
      setDerivative( getPntrToComponent(name_comp) ,i,der[i]);
    //reset gradients
    input_S.grad().zero_();
    //for(unsigned i=0; i<_n_in; i++)
    //	input_S.grad()[0][i] = 0.;
 
  }

}
}
}

#endif
