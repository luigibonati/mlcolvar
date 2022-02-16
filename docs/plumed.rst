PLUMED-PyTorch interface
========================

**Deploying CVs in PLUMED2**

Here we describe how to deploy the Pytorch-trained collective variables to enhance the sampling with the PLUMED2 open-source plug-in for molecular simulations. To do so, we will compile our model using ``torch.jit``, and then load it in PLUMED via the LibTorch C++ API library.

To do so, there are two options described below. The first requires you to add the Pytorch interface to the PLUMED source code and manually configure PLUMED, while the latter is based on a custom PLUMED version which already contains the interface and a modified configure which can look for libtorch libraries at configuration time.

.. toctree::
  :maxdepth: 1

  plumed_interface
  plumed_module