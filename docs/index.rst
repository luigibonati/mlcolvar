.. mlcolvar documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mlcolvar: Machine Learning Collective Variables
===============================================

.. image:: https://img.shields.io/badge/Github-mlcolvar-brightgreen
   :target: https://github.com/luigibonati/mlcolvar   

.. image:: https://img.shields.io/badge/preprint-arXiv:2305.19980-red
   :target: https://arxiv.org/abs/2305.19980
   
``mlcolvar`` is a Python library aimed to help design data-driven collective-variables (CVs) for enhanced sampling simulations. The key features are:

1. A unified framework to help test and use (some) of the CVs proposed in the literature. 
2. A modular interface to simplify the development of new approaches and the contamination between them.
3. A streamlined distribution of CVs in the context of advanced sampling. 

The library is built upon the `PyTorch <https://pytorch.org/>`_ ML library as well as the `Lightning <https://lightning.ai/>`_ high-level framework. 

Some of the **CVs** which are implemented, organized by learning setting:

* Unsupervised: PCA, (Variational) AutoEncoders [`1 <http://dx.doi.org/%2010.1002/jcc.25520>`_, `2 <http://dx.doi.org/2010.1021/acs.jctc.1c00415>`_ ]
* Supervised: LDA [`3 <(http://dx.doi.org/10.1021/acs.jpclett.8b00733>`_], DeepLDA [`4 <(http://dx.doi.org/2010.1021/acs.jpclett.0c00535>`_], DeepTDA [`5 <(http://dx.doi.org/%2010.1021/acs.jpclett.1c02317>`_]
* Time-informed: TICA [`6 <(http://dx.doi.org/%2010.1063/1.4811489>`_], DeepTICA/SRVs [`7 <(http://dx.doi.org/10.1073/pnas.2113533118>`_, `8 <(http://dx.doi.org/%2010.1063/1.5092521>`_ ], VDE [`9 <(http://dx.doi.org/10.1103/PhysRevE.97.062412>`_]

And many others can be implemented based on the building blocks or with simple modifications. Check out the documentation and the examples section!

The **workflow** for training and deploying CVs is illustrated in the figure:

.. image:: notebooks/tutorials/images/graphical_overview_mlcvs.png
  :width: 800
  :alt: Example workflow 

The resulting CVs can be then deployed for enhancing sampling with the `PLUMED <https://www.plumed.org/>`_ package via the `Pytorch <https://www.plumed.org/doc-master/user-doc/html/_p_y_t_o_r_c_h__m_o_d_e_l.html>`_ interface, , available since version 2.9.  

Table of contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api
   tutorials
   examples
   plumed
   contributing

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
