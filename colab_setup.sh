#!/bin/bash

# This is a environment setup file for the use of mlcolvar (mainly tutorial notebooks but not only) in Colab

echo - Cloning mlcolvar from git..
git clone --quiet https://github.com/luigibonati/mlcolvar.git mlcolvar

echo - Copying notebooks data..
cp -r mlcolvar/docs/notebooks/tutorials/data data
cp -r mlcolvar/docs/notebooks/paper_experiments/input_data input_data
cp -r mlcolvar/docs/notebooks/paper_experiments/results results
cp -r mlcolvar/docs/notebooks/paper_experiments/utils utils

echo - Installing mlcolvar requirements..
cd mlcolvar 
pip install -r requirements.txt -q
 
echo - Installing mlcolvar..
pip install -q .
cd ../

echo - Removing mlcolvar folder..
rm -r mlcolvar

echo Done!