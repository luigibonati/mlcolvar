mlcolvar: Machine learning collective variables
===============================================
[//]: # (Badges)
[![Documentation Status](https://readthedocs.org/projects/mlcolvar/badge/?version=latest)](https://mlcolvar.readthedocs.io/en/latest/?badge=latest)
[![GitHub Actions Build Status](https://github.com/luigibonati/mlcolvar/workflows/CI/badge.svg)](https://github.com/luigibonati/mlcolvar/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/luigibonati/mlcolvar/branch/main/graph/badge.svg?token=H01H68KNNG)](https://codecov.io/gh/luigibonati/mlcolvar)
[![license](https://img.shields.io/github/license/luigibonati/mlcolvar)](https://github.com/luigibonati/mlcolvar/blob/main/LICENSE)

`mlcolvar`, short for Machine Learning COLlective VARiables, is a Python library aimed to help design data-driven collective-variables (CVs) for enhanced sampling simulations.

---

The **goals** of `mlcolvar` are to develop:

1. A unified framework to help test and use (some) of the CVs proposed in the literature. 
2. A modular interface to simplify the development of new approaches and the contamination between them.
3. A streamlined distribution of CVs in the context of advanced sampling. 

The library is built upon the [PyTorch](https://pytorch.org/) ML library as well as the [Lightning](https://lightning.ai/) high-level framework. 

---

The **workflow** for training CVs is illustrated in the figure below. The resulting CVs can be then deployed for enhancing sampling with the [PLUMED](https://www.plumed.org/) package via the [pytorch](https://www.plumed.org/doc-master/user-doc/html/_p_y_t_o_r_c_h__m_o_d_e_l.html>`_) interface. 

<center><img src="docs/notebooks/tutorials/images/graphical_overview_mlcvs.png" width="800" /></center> 

---

**Notes**: in early versions (`v<=0.2.*`) the library was called `mlcvs`. This is still accessible for compatibility with PLUMED masterclasses in the [releases](https://github.com/luigibonati/mlcolvar/releases) or by cloning the `pre-lightning` branch.

---

Copyright (c) 2023 Luigi Bonati, Enrico Trizio and Andrea Rizzi. 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
