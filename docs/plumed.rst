PLUMED module
=============

Deploying CVs in PLUMED2
------------------------

Machine-learning collective variables can be deployed in PLUMED2 and used
during molecular simulations, for example to enhance the sampling of rare
events.

Two deployment strategies are currently supported for graph-based CV models:
TorchScript and PyTorch Ahead-of-Time (AOT) compilation.


TorchScript
^^^^^^^^^^^

A trained model can be compiled using TorchScript with the
``to_torchscript`` method, as described in this
`tutorial <notebooks/tutorials/intro_1_training.html#Deploy-the-model-in-PLUMED>`_.

This produces a serialized model that can be executed outside Python, for
example from a standalone C++ program.

The model can then be evaluated in PLUMED using the PyTorch C++ API
(LibTorch). The descriptor-based
`PYTORCH_MODEL <https://www.plumed.org/doc-master/user-doc/html/PYTORCH_MODEL/>`_
interface is part of the official PLUMED2 PyTorch module starting from
PLUMED 2.9.

Instructions for configuring PLUMED with LibTorch are available in the
`PLUMED PyTorch module documentation <https://www.plumed.org/doc-master/user-doc/html/module_pytorch/>`_.


Ahead-of-Time compilation
^^^^^^^^^^^^^^^^^^^^^^^^^

Graph-based CV models can also be exported using PyTorch Ahead-of-Time (AOT)
compilation.

AOT compilation produces a compiled model package that can be loaded directly
from C++, avoiding the TorchScript runtime representation.

Models can be exported using the utilities in ``mlcolvar.utils.aot``. For
example:

.. code-block:: python

   from mlcolvar.utils.aot import export

   export(
       model=model,
       example_inputs=example_graph,
       file_name="model.pt2",
       calculate_gradients=True,
   )

The resulting ``model.pt2`` package can be evaluated in PLUMED using the
AOT-specific GNN interfaces provided in this repository.

For models used with the Kolmogorov bias, the corresponding bias parameters
can be included during export:

.. code-block:: python

   export(
       model=model,
       example_inputs=example_graph,
       file_name="model.pt2",
       calculate_gradients=True,
       k_bias_options={
           "beta": 0.5,
           "lambd": 1.0,
       },
   )


PLUMED interfaces
-----------------

The most updated PLUMED interfaces are available in the repository under the
``plumed_interfaces/`` directory.

These include interfaces that may not yet be available in the official PLUMED
release.


TorchScript interfaces
^^^^^^^^^^^^^^^^^^^^^^

The generic TorchScript interfaces are:

- ``PytorchModel.cpp``: descriptor-based interface for evaluating a generic
  TorchScript model from PLUMED and exposing one component for each output
  node.

- ``PytorchModelGNN.cpp``: graph-based interface for TorchScript GNN models,
  where the input graph is constructed directly in PLUMED from atomic
  positions and atom types.

For the Kolmogorov bias:

- ``PytorchKolmogorovBias.cpp``: descriptor-based interface for committor
  models, returning the raw ``z`` output, the activated committor ``q``, and
  the corresponding Kolmogorov bias.

- ``PytorchKolmogorovBiasGNN.cpp``: graph-based committor interface combining
  GNN evaluation with the computation of ``q`` and the Kolmogorov bias.


AOT interfaces
^^^^^^^^^^^^^^

The corresponding graph-based AOT interfaces are:

- ``PytorchModelGNNAOT.cpp``: interface for GNN-based CV models exported with
  AOT compilation. The graph is constructed directly in PLUMED and the
  compiled ``model.pt2`` package is evaluated through the PyTorch AOT runtime.

- ``PytorchKolmogorovBiasGNNAOT.cpp``: AOT interface for graph-based
  committor models. It evaluates the compiled model and exposes the raw
  coordinate ``z``, the activated committor ``q``, and the corresponding
  Kolmogorov bias.

The corresponding PLUMED actions are:

.. code-block:: text

   PYTORCH_GNN
   PYTORCH_KOLMOGOROV_BIAS_GNN
   PYTORCH_GNN_AOT
   PYTORCH_KOLMOGOROV_BIAS_GNN_AOT