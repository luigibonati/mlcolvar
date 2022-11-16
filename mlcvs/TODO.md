# TODOs

## General
- [X] remove dtype and use set_default_type 
- [X] remove device and use .to?
- [ ] use _ for private functions
- [X] add submodules
- [X] change train in fit and create alias for predict

## io
- [X] take general fucntion from md-stateinterpreter

## Models 
### LinearCV
- [X] inherit from nn.Module and save params as buffers
- [X] change name of transform and fit?
- [X] change features_names to feature_names (also NNCV)
- [X] allow use pandas dataframe as input? (set features)
- [X] enable standardize inputs/outputs (problem: plumed input)
- [ ] check forward with differnet number of inputs than initialized

### NNCV
- [X] print plumed input file
- [X] export model
- [ ] allow to take as input nn.module and copy architecture

## Lda
### Both
- [x] compose LDA instead of inheritance
- [ ] expose methods? (eigvals, matrices... )

### LDA_CV
- [X] implement HLDA
 
### DeepLDA_CV
- [ ] add multiclass loss function (loss function)
- [ ] change names to private members
- [ ] add dataloader option to valid_data

### earlystopping
- [X] save model.state_dict and then load

### REFACTORING
- [X] remove custom loss
- [X] prepare_dataset function
- [X] move params (e.g. train) outside
- [X] move fit to nn base
- [X] add custom_train_epoch to fit 
- [x] add eval_dataset 
- [x] change log to dictionary
- [ ] add tests for custom_train
- [ ] changed .to into set_device

### MISCELLANEA
- [ ] create dataloader from file
- [ ] evaluate dataloader function
- [ ] add option to script rather than trace jit model