PLUMED-PyTorch interface
========================

**Deploying CVs in PLUMED2**

Here we describe how to deploy the Pytorch-trained collective variables to enhance the sampling with the PLUMED2 open-source plug-in for molecular simulations. To do so, we will compile our model using `torch.jit`, and then load it in PLUMED via the LibTorch C++ API library.

**Requirements**

* `PLUMED2 <https://www.plumed.org/download>`_ (v >= 2.8)

* `LibTorch <https://pytorch.org>`_ (same version of Pytorch used to train the model)

* PLUMED-Pytorch interface, available in this repository.

Optionally, the `PytorchModel.cpp` interface can be compiled with PLUMED by putting it inside the folder `plumed2/src/function/` before installing it. Otherwise, it will be necessary to link it at runtime using the `LOAD <https://www.plumed.org/doc-master/user-doc/html/_l_o_a_d.html>`_ keyword::

    LOAD FILE=PytorchModel.cpp


**Configure PLUMED with LibTorch**

The following instructions are valid for Pytorch (and LibTorch) version 1.8.2 LTS. Note that the flags for linking might be different for other versions of LibTorch.

Here $LIBTORCH contains the location of the precompiled binaries and $PLUMED the target folder for PLUMED installation:

.. code-block:: 

    ./configure --prefix=$PLUMED --enable-cxx=14 --enable-rpath \
                --disable-external-lapack --disable-external-blas \
                CXXFLAGS="-O3 -dynamic -D_GLIBCXX_USE_CXX14_ABI=1 -std=c++14" \
                CPPFLAGS="-I${LIBTORCH}/include/torch/csrc/api/include/ -I${LIBTORCH}/include/ -I${LIBTORCH}/include/torch" \
                LDFLAGS="-L${LIBTORCH}/lib -ltorch -lc10 -ltorch_cpu -Wl,-rpath,${LIBTORCH}/lib"

Notes:

- If using the CUDA-enabled binaries `-ltorch_cuda` needs to be added to LDFLAGS.
  
- If using the pre-cxx11 ABI binaries the corresponding flag should be disabled in CXXFLAGS: `-D_GLIBCXX_USE_CXX11_ABI=0`.

- Due to a conflict with the BLAS/LAPACK libraries contained in the LibTorch binaries, the search for other external libraries has to be disabled.

- The option to enable C++14 has been added in PLUMED since version 2.8.

- In order to be sure that the interface is correctly installed it is advised to add it to the PLUMED src folder rather than linked at runtime. This allows to detect possible linking errors at compile time.
 
**Load the model in the PLUMED input file**

In the PLUMED input file one should specify the model and the arguments. The interface detects the number of outputs and create a component for each of them, which can be accessed as cv.node-0, cv.node-1, ... ::

    cv: PYTORCH_MODEL FILE=model.pt ARG=d1,d2,...,dN
