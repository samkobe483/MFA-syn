# MFASyn: Multimodal Drug Synergy Prediction via Adaptive Feature Reconstruction and Bilinear Interaction Pooling

A deep learning framework for anti-cancer drug combination synergy prediction using adaptive feature selection, dual-pathway cell encoding, and second-order interaction modeling.

## Overview

MFASyn (Multimodal Feature Adaptive Synergy) integrates:
- Adaptive Drug Features: Uses SE-Blocks (Squeeze-and-Excitation) to filter Morgan fingerprints and Transformer encoders for drug-target/pathway interactions.
- Dual-Pathway Cell Line Encoder:
Feature Branch: Processes gene expression data.
Structure Branch: Utilizes GIN (Graph Isomorphism Network) to extract topological features from Protein-Protein Interaction (PPI) networks.
-Bilinear Interaction Pooling: Explicitly models second-order multiplicative interactions between drugs and cell lines to capture non-additive dependencies.
-Regression Analysis: Optimized for predicting continuous synergy scores (e.g., Loewe, Bliss) with high precision.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/MFASyn.git
cd MFASyn
```

### 2. Create conda environment

```bash
Recommended dependencies: torch, torch_geometric, rdkit, lifelines
conda env create -f environment.yaml
```

### 3. Activate environment

```bash
conda activate MFASyn
```

## Dataset

The project uses drug synergy data from:
- **Oneil dataset**: Drug combinations from O'Neil et al.
- **ALMANAC dataset**: NCI-ALMANAC drug synergy data

Place your data files in the `data/` directory:
- `smiles.csv` - Drug SMILES structures
- `drug_protein_feature.pkl` - Drug-target protein features
- `drug_pathway_feature.pkl` - Drug-pathway features
- `cell_features.csv` - Cell line gene expression data
- `cell_feat.npy` - Cell line feature matrix
- `oneil_synergyloewe.txt` / `almanac_synergyloewe.txt` - Synergy scores

## Usage

### Training

```bash
# # Train the model and evaluate regression performance
python main.py
```

### Evaluation

```bash
# # Generate regression scatter plots and performance metrics (MSE, PCC, SCC, CI)
# Plots are saved in the results/figures/ directory


### Configuration

Modify `get_dataset.py` to switch datasets:
```python
SYNERGY_FILENAME = 'oneil_synergyloewe.txt'  # or 'almanac_synergyloewe.txt'
```

## Project Structure

```
MFAsyn/
├── data/                    # Data files
├── results/                 # Training results
├── main.py                  # Main training script
├── model.py                 # DSPSCL model
├── get_dataset.py           # Data loading and processing
└── environment.yaml         # Conda environment
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- RDKit
- PyTorch Geometric
- See `environment.yaml` for full dependencies

## Citation

If you use this code, please cite our work.

