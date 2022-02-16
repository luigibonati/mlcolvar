Add the interface to PLUMED
===========================

**Requirements**

* `PLUMED2 <https://www.plumed.org/download>`_ (v >= 2.8)

* `LibTorch <https://pytorch.org>`_ (same version of Pytorch used to train the model)

* `PLUMED-Pytorch interface <https://github.com/luigibonati/mlcvs/blob/main/plumed/PytorchModel.cpp>`_ , available in this repository.

**Add the LibTorch interface to PLUMED**

The ``PytorchModel.cpp`` code needs to be added to the ``plumed2/src/function/`` folder before installing PLUMED. 

**Configure PLUMED with LibTorch**

The following instructions are valid for Pytorch (and LibTorch) version 1.8.* LTS. Note that the flags for linking might be different for other versions of LibTorch.

Here ``$LIBTORCH`` contains the location of the precompiled binaries and ``$PLUMED`` the target folder for PLUMED installation:

.. code-block:: 

    ./configure --prefix=$PLUMED --enable-cxx=14 \
                --disable-external-lapack --disable-external-blas \
                CPPFLAGS="-I${LIBTORCH}/include/torch/csrc/api/include/ -I${LIBTORCH}/include/ -I${LIBTORCH}/include/torch" \
                LDFLAGS="-L${LIBTORCH}/lib -ltorch -lc10 -ltorch_cpu -Wl,-rpath,${LIBTORCH}/lib"

Notes:

- If using the CUDA-enabled binaries ``-ltorch_cuda -lc10_cuda`` need to be added to LDFLAGS (note: CUDA support is not enabled in the interface).
  
- If using the pre-cxx11 ABI LibTorch binaries the following flag should be used: ``CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"``.

- Due to a conflict with the BLAS/LAPACK libraries contained in the LibTorch binaries, the search for other external libraries has to be disabled.

- The option to enable C++14 has been added in PLUMED since version 2.8.
 
**Load the model in the PLUMED input file**

In the PLUMED input file one should specify the model and the arguments. Here `x1,x2,..,xN` are the inputs which needs to be previusly defined inside the PLUMED input. The interface automatically detects the number of outputs and create a component for each of them, which can be accessed as cv.node-0, cv.node-1, ... ::

    cv: PYTORCH_MODEL MODEL=model.pt ARG=x1,x2,...,xN

**Alternative: loading the interface at runtime**

One can also take advantage of the `LOAD <https://www.plumed.org/doc-master/user-doc/html/_l_o_a_d.html>`_ keyword to compile the ``PytorchModel.cpp`` interface at runtime ::

    LOAD FILE=PytorchModel.cpp

However, also in this case PLUMED has to be configured as detailed in the previous section. Furthermore, some of the PLUMED header files should be changed as to allow it to found them: ``#include "Function.h"`` --> ``#include "function/Function.h"`` and ``#include "ActionRegister.h"`` --> ``#include "function/ActionRegister.h"``
