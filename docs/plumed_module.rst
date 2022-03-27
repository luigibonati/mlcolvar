Download custom PLUMED version (deprecated)
============================================

This page illustrates how to download and install the modified version of PLUMED which (1) can look at configuration time for the presence of libtorch and (2) contains a newer interface for loading Pytorch models. 

Requirements
""""""""""""

* Modified `PLUMED2 <https://github.com/luigibonati/plumed2/tree/pytorch_module>`_ version. This repository is kept synced with the master branch of Plumed. It will be (hopefully) soon merged into the official plumed release. 

* `LibTorch <https://pytorch.org>`_ (version 1.8.* LTS)

Download modified PLUMED
""""""""""""""""""""""""""""""""""""

Clone the repository and select the ``pytorch_module`` branch: 

.. code-block::

    git clone https://github.com/luigibonati/plumed2.git -b pytorch_module plumed2-pytorch
    cd plumed2-pytorch

Download Libtorch
""""""""""""""""""""""""

It's strongly recommended to use both Pytorch and Libtorch version LTS 1.8.*. The reason is that the C++ Libtorch APIs are not stable, and there are major breaking changes across different versions. If the model is created with a different version of Pytorch the interface might not be able to load it.  

.. code-block::

    wget https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcpu.zip ;
    unzip libtorch-cxx11-abi-shared-with-deps-1.8.2+cpu.zip ;
    rm libtorch-cxx11-abi-shared-with-deps-1.8.2+cpu.zip

The location of the include and library files need to be exported in the environment. For convenience, we save them in a file ``sourceme.sh`` inside the libtorch folder:

.. code-block:: 

    LIBTORCH=${PWD}/libtorch
    echo "export CPATH=${LIBTORCH}/include/torch/csrc/api/include/:${LIBTORCH}/include/:${LIBTORCH}/include/torch:$CPATH" >> ${LIBTORCH}/sourceme.sh
    echo "export INCLUDE=${LIBTORCH}/include/torch/csrc/api/include/:${LIBTORCH}/include/:${LIBTORCH}/include/torch:$INCLUDE" >> ${LIBTORCH}/sourceme.sh
    echo "export LIBRARY_PATH=${LIBTORCH}/lib:$LIBRARY_PATH" >> ${LIBTORCH}/sourceme.sh
    echo "export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH" >> ${LIBTORCH}/sourceme.sh
    . ${LIBTORCH}/sourceme.sh

.. note::

   Remember to add the line ``. ${LIBTORCH}/sourceme.sh`` to your ``~/.bashrc`` or  ``~/.bash_profile`` file. 

Configure PLUMED
""""""""""""""""""""""""

To configure PLUMED we need to (1) specify to look for the libtorch library (``--enable-libtorch``) and (2) enable the pytorch module which contains the PLUMED-Pytorch interface (``--enable-modules=pytorch``). If ``$PLUMED`` is the target folder for PLUMED installation:

.. code-block:: 
    
    PLUMED=$PWD/install
    ./configure --prefix=$PLUMED --enable-cxx=14 --disable-external-lapack --disable-external-blas \
                --enable-libtorch LIBS="-ltorch -lc10 -ltorch_cpu" \
                --enable-modules=pytorch  

.. Tip:: 

    If the procedure is successful the configure log will report the following line: ::

        checking libtorch without extra libs... yes
        
    while if it is not able to find the needed libraries it will print a warning: ::

        checking libtorch without extra libs... no
        configure: WARNING: cannot enable __PLUMED_HAS_LIBTORCH

.. WARNING::

   - Due to a conflict with the BLAS/LAPACK libraries contained in the LibTorch binaries, the search for other external libraries has to be disabled.
   - It appears that there is a conflict in using the intel compiler with the precompiled LibTorch library, while it works fine with gcc and clang

.. note::

   - If using the CUDA-enabled binaries ``-ltorch_cuda -lc10_cuda`` need to be added to LIBS (note: CUDA support is not enabled in the interface).
   - If you want to use the pre-cxx11 ABI LibTorch binaries the following flag should be added to the configure: ``CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"``.
  
Install PLUMED
""""""""""""""""""""""""

Once configured we can install PLUMED:

.. code-block::

    make -j 16 # number of threads
    make install

Remember also to export the relevant paths as described in the log. 

Regtests (optional)
""""""""""""""""""""""""

To check that the installation of PLUMED was successful you can execute: ::

    make regtests

or to check only the Pytorch interface: :: 

    cd regtest/pytorch && make testclean 

Load the model in the PLUMED input file
""""""""""""""""""""""""""""""""""""""""""""""""

In the PLUMED input file one should specify the model and the arguments. Here `x1,x2,..,xN` are the inputs which needs to be previusly defined inside the PLUMED input. The interface automatically detects the number of outputs and create a component for each of them, which can be accessed as cv.node-0, cv.node-1, ... ::

    cv: PYTORCH_MODEL FILE=model.ptc ARG=x1,x2,...,xN

The default name for the model is ``model.ptc``. 

Differences with the previous interface
""""""""""""""""""""""""""""""""""""""""""""""""

- The name of the model has to be specified with the ``FILE`` keyword rather than the ``MODEL`` one.
- Better handling of derivatives (using ``torch.autograd.grad`` rather than ``backward``).
- Fixes bug of derivatives in the case the model has more than one output. 
- Better error printing: different error messages if it is not able to load the file because it does not exist or because it is not a valid pytorch compiled model. In the latter case it prints also the version of LibTorch.  