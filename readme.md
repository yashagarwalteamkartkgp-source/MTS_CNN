# MTS-CNN Fault Detection (Research Paper Implementation)

## Overview
This project implements a Multi-Channel CNN with a diagnostic layer for fault detection in multivariate time-series data.

## Features
- Synthetic semiconductor-like dataset
- Multi-channel CNN
- Diagnostic layer (sensor importance)
- Full training + evaluation pipeline
- Visualization (loss, t-SNE, predictions)

## Setup

pip install -r requirements.txt

## Run

python run_experiment.py 

## Output

The model will be saved in the `outputs/models` directory.
The plots will be saved in the `outputs/plots` directory.
The t-SNE visualization will be saved in the `outputs/plots/tsne_visualization.png`.
The confusion matrix will be saved in the `outputs/plots/confusion_matrix.png`.
