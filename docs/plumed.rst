PLUMED module
=============

**Deploying CVs in PLUMED2**

In order to use the ML CVs to enhance the sampling we can export them into the PLUMED2 open-source plug-in for molecular simulations. 
To do so, we will compile our model using the just in time compiler (``torch.jit``). This creates a file which can be execute outside Python, e.g. in a standalone C++ programs. 

In this way we can load the CVs in PLUMED by using PyTorch C++ APIs (LibTorch). We have developed an interface (`PYTORCH_MODEL <https://www.plumed.org/doc-master/user-doc/html/PYTORCH_MODEL/>`_) which is now part of the official PLUMED2 software as an additional module, starting from version 2.9. To configure PLUMED with Libtorch please have a look at the PLUMED `documentation <https://www.plumed.org/doc-master/user-doc/html/module_pytorch/>`_.
