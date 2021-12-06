# DOP-GNN

Implementation of 2020 Fall CS280 final project 
"DOP-GNN: Communication-Efficient Frameworks for Distributed Optimization with Graph Neural Network"

## Overview

- baseline.py		% Implementations of DGD, EXTRA, DIGing baseline
- dataset.py		% Synthetic dataset generation and a example of dataset loader
- func_generator.py	% Generate Random QP problems
- gat_conv.py		% Graph Attention Convolution Layer
- loss.py			% loss function
- naive_supervised.py	% run experiments
- net.py			% Implementations of DOP-GNN Models
- optbaseline.py	% get optimal solutions for optimization problems by CVXPY

## Requirements

	numpy
	networkx
	cvxpy
    pytorch
    pytorch_geometric

## Run Experiments

    python naive_supervised.py
