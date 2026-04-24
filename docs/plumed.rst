PLUMED module
=============

**Deploying CVs in PLUMED2**

In order to use the ML CVs to enhance the sampling we can export them into the PLUMED2 open-source plug-in for molecular simulations. 

To do so, we will compile our model using the just in time compiler. This can be done conveniently with the method `to_torchscript` as described in this `tutorial <notebooks/tutorials/intro_1_training.html#Deploy-the-model-in-PLUMED>`_.

This creates a file which can be execute outside Python, e.g. in a standalone C++ programs. 

In this way we can load the CVs in PLUMED by using PyTorch C++ APIs (LibTorch). We have developed an interface (`PYTORCH_MODEL <https://www.plumed.org/doc-master/user-doc/html/PYTORCH_MODEL/>`_) which is now part of the official PLUMED2 software as an additional module, starting from version 2.9. To configure PLUMED with Libtorch please have a look at the PLUMED `documentation <https://www.plumed.org/doc-master/user-doc/html/module_pytorch/>`_.

The most updated versions of PLUMED interfaces are available in the repository, in the subfolder ``plumed_interfaces/``.
These include also interfaces not yet available in the official PLUMED release. 
The interfaces are:
- ``PytorchModel.cpp``: descriptor-based interface for evaluating a generic TorchScript model from PLUMED and exposing one component for each output node.
- ``PytorchModelGNN.cpp``: graph-based interface for TorchScript GNN models, where the input graph is built directly in PLUMED from atomic positions and types.

In addition to these generic interfaces, there are two specific interfaces for the Kolmogorov Bias approach:
- ``PytorchKolmogorovBias.cpp``: descriptor-based interface for committor models, returning the raw ``z`` output, the activated committor ``q``, and the corresponding Kolmogorov bias.
- ``PytorchKolmogorovBiasGNN.cpp``: graph-based committor interface that combines the GNN evaluation with the computation of ``q`` and the Kolmogorov bias.
