# Devito-modelling

A collection of codes to perform all sort of seismic modeling with Devito.

## Project structure
This repository is organized as follows:

* :open_file_folder: **devitomod**: python library containing routines to perform seismic modelling with devito;
* :open_file_folder: **data**: folder containing data used in the examples
* :open_file_folder: **notebooks**: set of jupyter notebooks showcasing how to use devitomod to model seismic data.

## Notebooks
The following notebooks are provided:

- :orange_book: ``Acoustic2d.ipynb``: notebook performing 2D Acoustic modelling;
- :orange_book: ``Elastic2d.ipynb``: notebook performing 2D Elastic modelling;


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

Remember to always activate the environment by typing:
```
conda activate devitomod
```

Also, to enable OpenMP when running a Devito code, make sure to set the following environment variable before running a code
or opening jupyter-lab
```
export DEVITO_LANGUAGE=openmp
```
