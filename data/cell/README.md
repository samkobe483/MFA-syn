以下是根据 Cell Line Embedding README 的格式重写后的 **完整 AFBMSyn README**，你可以直接一键复制并粘贴到 GitHub 的 `README.md` 文件中。

```markdown
# AFBMSyn: Multimodal Drug Synergy Prediction based on Adaptive Feature Learning and Bilinear Interaction Modeling

A deep learning framework for anti-cancer drug combination synergy prediction using adaptive feature selection, dual-pathway cell encoding, and second-order interaction modeling.

---

## 📌 Overview

AFBMSyn (Adaptive Feature and Bilinear Modeling Synergy) integrates:

| Component | Description |
|-----------|-------------|
| **Adaptive Drug Features** | SE‑Blocks (Squeeze‑and‑Excitation) to filter Morgan fingerprints + Transformer encoders for drug‑target/pathway interactions |
| **Dual‑Pathway Cell Line Encoder** | *Feature branch*: gene expression data; *Structure branch*: GIN (Graph Isomorphism Network) on PPI networks |
| **Bilinear Interaction Pooling** | Models second‑order multiplicative interactions between drugs and cell lines |
| **Regression Analysis** | Optimized for continuous synergy scores (Loewe, Bliss, etc.) |

---

## 🛠️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/AFBMSyn.git
cd AFBMSyn
```

### 2️⃣ Create conda environment

```bash
conda env create -f environment.yaml
```

### 3️⃣ Activate environment

```bash
conda activate AFBMSyn
```

> **Alternative (pip)**  
> ```bash
> pip install torch torch-geometric rdkit lifelines
> ```

---

## 📊 Dataset

The project uses drug synergy data from:

| Dataset | Source |
|---------|--------|
| **Oneil** | O’Neil *et al.* drug combinations |
| **ALMANAC** | NCI‑ALMANAC drug synergy data |

Place your data files in the `data/` directory:

| File | Description |
|------|-------------|
| `smiles.csv` | Drug SMILES structures |
| `drug_protein_feature.pkl` | Drug‑target protein features |
| `drug_pathway_feature.pkl` | Drug‑pathway features |
| `cell_features.csv` | Cell line gene expression data |
| `cell_feat.npy` | Cell line feature matrix |
| `oneil_synergyloewe.txt` | Synergy scores (O’Neil dataset) |
| `almanac_synergyloewe.txt` | Synergy scores (ALMANAC dataset) |

---

## 🚀 Usage

### Training

```bash
python main.py
```

Trains the model and evaluates regression performance.

### Evaluation

```bash
# Generates regression scatter plots and metrics (MSE, PCC, SCC, CI)
# Output saved in results/figures/
```

### Configuration

Switch datasets by editing `get_dataset.py`:

```python
SYNERGY_FILENAME = 'oneil_synergyloewe.txt'   # or 'almanac_synergyloewe.txt'
```

---

## 📁 Project Structure

```
AFBMSyn/
├── data/                    # Data files
├── results/                 # Training results (figures, metrics)
├── main.py                  # Main training script
├── model.py                 # AFBMSyn model definition
├── get_dataset.py           # Data loading and preprocessing
└── environment.yaml         # Conda environment
```

---

## 📦 Requirements

- Python 3.8+
- PyTorch 1.10+
- RDKit
- PyTorch Geometric
- lifelines
- numpy, pandas, scikit-learn

See `environment.yaml` for full details.

---

## 📖 Citation

If you use this code, please cite our work:

> *AFBMSyn: Multimodal Drug Synergy Prediction based on Adaptive Feature Learning and Bilinear Interaction Modeling* (manuscript in preparation)

For the cell line embedding component, also cite:

> Xiaowen Wang, et al. *PRODeepSyn: predicting anticancer synergistic drug combinations by embedding cell lines with protein–protein interaction network.* Briefings in Bioinformatics, 2022.


