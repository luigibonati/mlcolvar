# mlcolvar paper experiments

Experiments files from 'A unified framework for machine learning collective variables for enhanced sampling simulations: mlcolvar'.

Luigi Bonati, Enrico Trizio, Andrea Rizzi and Michele Parrinello

[arXiv preprint](https://arxiv.org/abs/2305.19980)

Here you can find all the files needed to reproduce the experiments reported in the paper on a three states toy model in two dimension.
To run the simulations you need PLUMED with the pytorch and ves modules active.

#### Contents:
  - Input files for the training of the models (input_data folder)
  - Input files for reproducing the simulations in the paper (results folder)
  - Jupyter notebooks for the training of the models and analysis (main_folder)
  - Trained models used in the paper (results_folder)

#### Colab links for Jupyter notebooks
- [Notebook unsupervised](https://colab.research.google.com/github/luigibonati/mlcolvar/blob/main/docs/notebooks/paper_experiments/paper_1_unsupervised.ipynb)
- [Notebook supervised](https://colab.research.google.com/github/luigibonati/mlcolvar/blob/main/docs/notebooks/paper_experiments/paper_2_supervised.ipynb)
- [Notebook timelagged](https://colab.research.google.com/github/luigibonati/mlcolvar/blob/main/docs/notebooks/paper_experiments/paper_3_timelagged.ipynb)
- [Notebook multitask](https://colab.research.google.com/github/luigibonati/mlcolvar/blob/main/docs/notebooks/paper_experiments/paper_4_multitask.ipynb)

#### mlcolvar library 
- [Documentation](https://mlcolvar.readthedocs.io)
- [GitHub](https://github.com/luigibonati/mlcolvar)