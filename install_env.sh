#!/bin/bash
# 
# Installer for Devito-modelling environment
# 
# Run: ./install_env.sh 
#
# M. Ravasi, 23/01/2024

echo 'Creating Devito-modelling environment'

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate devitomod
echo 'Created and activated environment:' $(which python)

# check packages work as expected
echo 'Checking devito version and running a command...'
python -c 'import devito; print(devito.__version__);'

echo 'Done!'