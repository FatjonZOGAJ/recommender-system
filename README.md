# Collaborative Filtering

This repository is part of the solutions for the course Computational Intelligence Lab at ETH Zurich, Spring Semester
2021.

## Overview

Contains the following algorithms:

- Singular Value Decomposition
- Non-Negative Matrix Factorization
- Autoencoder
- Autorec
- Neural Collaborative Filtering
- Kernel Net
- Bayesian Factorization Machine

## Reproduce Results

Use Python version 3.7.4

### Create environment

    conda create --name collaborative-filtering python=3.7.4 

[comment]: <> (    python -m venv "collaborative-filtering")

### Activate environment

    conda activate collaborative-filtering

[comment]: <> (    source collaborative-filtering/bin/activate)

### Install dependencies

    pip install --user -r requirements.txt 

### Execute scripts

    python main.py

This will reproduce the best submitted Kaggle result. However, note that this may take up to 2 days to finish.

    python cv_experiments.py

This will reproduce the results used for generating the plots of the report.

    python movie_clustering.py
    python movie_distance.py

This will create the matrices used for movie features in the Bayesian Factorization Machine model. Note that Jaccard indices will only be calculated on the fly, if not downloaded beforehand. The Jaccard indices can be downloaded from [Polybox](https://polybox.ethz.ch/index.php/s/Ff7YPYEIHOHOGpD) and put them into `data/features`.

## Report

To build the report, simply call:

    cd report

    bash build-paper.sh 

# TODOS

Sign plagiarism form 

Submit final submission on kaggle

Start iterative approach experiment

Write ensemble Document code

Write on how to run the experiments/generate the data write on how to reproduce best score
create params in autorec
