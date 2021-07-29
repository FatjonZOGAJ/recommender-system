# Collaborative Filtering

This repository is part of  the solutions for the course Computational Intelligence Lab at ETH Zurich, 
Spring Semester 2021.

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
### Train the model
    python main.py
## Report
To build the report, simply call:

    cd report

    bash build-paper.sh 