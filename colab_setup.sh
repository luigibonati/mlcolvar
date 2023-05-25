#!/bin/bash

# This is a environment setup file for the use of mlcolvar (mainly tutorial notebooks but not only) in Colab

if test -f "colab_is_setup.txt"; then
    echo This Colab environment is already set up!
else
    git clone --quiet https://github.com/luigibonati/mlcolvar.git mlcolvar
    echo - Cloned mlcolvar from git

    cp -r mlcolvar/docs/notebooks/tutorials/data data
    cp -r mlcolvar/docs/notebooks/paper_experiments/input_data input_data
    cp -r mlcolvar/docs/notebooks/paper_experiments/results results
    cp -r mlcolvar/docs/notebooks/paper_experiments/utils utils
    echo - Copied notebooks data

    cd mlcolvar 
    pip install -r requirements.txt -q .
    echo - Installed mlcolvar requirements
    echo - Installed mlcolvar

    cd ../
    rm -r mlcolvar
    echo - Removed mlcolvar folder

    echo This Colab environment is setup, enjoy! > colab_is_setup.txt
    echo Done!
fi