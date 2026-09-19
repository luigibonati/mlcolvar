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

For graph-based CVs, AOT compilation can be particularly useful because GNN
evaluation is performed repeatedly during molecular dynamics simulations and
can introduce a non-negligible runtime overhead. By compiling the model ahead
of time, AOT can reduce runtime overhead and improve the overall simulation
throughput.


GNN benchmark
~~~~~~~~~~~~~

To compare TorchScript and AOT deployment, we benchmarked two SchNet-based
DeepTDA collective variables with different graph sizes and model
complexities.

For Ala2, the graph contains the 10 heavy atoms of the molecule, with edges
constructed using a 10 Å cutoff. No additional long-range graph is used.
The SchNet model contains two interaction layers, 16 radial basis functions,
16 filters, and 16 hidden channels.

For chignolin, all protein heavy atoms form the short-range graph using a
4 Å cutoff. In addition, the C-alpha atoms form a long-range subsystem graph
using an 18 Å cutoff. The SchNet model contains three interaction layers,
16 radial basis functions, 32 filters, and 32 hidden channels.

.. plot::
   :include-source: false
   :align: center

   import matplotlib.pyplot as plt
   import numpy as np

   labels = [
       "Ala2 CPU",
       "Ala2 CUDA",
       "Chignolin CPU",
       "Chignolin CUDA",
   ]

   aot = [132.795, 180.209, 60.604, 53.603]
   torchscript = [89.993, 69.557, 22.089, 43.752]

   x = np.arange(len(labels))
   width = 0.36

   fig, ax = plt.subplots(figsize=(8, 5))

   bars_aot = ax.bar(
       x - width / 2,
       aot,
       width,
       label="AOT",
   )

   bars_ts = ax.bar(
       x + width / 2,
       torchscript,
       width,
       label="TorchScript",
   )

   ax.set_ylabel("Simulation throughput (ns/day)")
   ax.set_xticks(x)
   ax.set_xticklabels(labels)
   ax.legend()

   for bars in (bars_aot, bars_ts):
       for bar in bars:
           height = bar.get_height()
           ax.text(
               bar.get_x() + bar.get_width() / 2,
               height,
               f"{height:.1f}",
               ha="center",
               va="bottom",
               fontsize=8,
           )

   ax.set_ylim(0, max(aot + torchscript) * 1.15)

   fig.tight_layout()
   plt.show()

AOT achieves higher simulation throughput than TorchScript in all tested
configurations.

AOT models are device- and precision-specific, whereas TorchScript models can
select the execution device at runtime

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