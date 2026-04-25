# Cell Line Embedding with PPI Network

This sub-project generates cell line embeddings using protein-protein interaction (PPI) networks and omics data (gene expression and mutation). Used for drug synergy prediction.

---

## File Description

| File | Description |
|------|-------------|
| `const.py` | Data path configuration |
| `dataset.py` | Dataset classes (C2VDataset, C2VSymDataset) |
| `model.py` | Model architecture (GINEncoder, Cell2Vec, RandomW) |
| `train.py` | Training script for cell embeddings |
| `gen_feat.py` | Generate normalized cell features from embeddings |
| `utils.py` | Utility functions (model saving, loss visualization) |
| `train_gin_example.py` | Example showing GIN encoder usage |

---

## Data Files (in `data/Cell/data/data/`)

| File | Description |
|------|-------------|
| `ppi.coo.npy` | PPI network edges (COO format) |
| `node_features.npy` | Node features for PPI graph |
| `target_ge.npy` | Gene expression targets |
| `nodes_ge.npy` | Valid gene nodes for GE |
| `target_mut.npy` | Mutation targets |
| `nodes_mut.npy` | Valid gene nodes for MUT |
| `cell_feat.npy` | Output: normalized cell features |

---

## Run Instructions

### Step 1: Prepare data

Ensure the following files exist in `data/Cell/data/data/`:

- `ppi.coo.npy` - PPI network edges
- `node_features.npy` - Node features
- `target_ge.npy`, `nodes_ge.npy` - Gene expression data
- `target_mut.npy`, `nodes_mut.npy` - Mutation data

### Step 2: Train cell embeddings

```bash
python train.py
Trains GE and MUT embeddings (default: 128 hidden dim, 384 embedding dim)

Step 3: Generate cell features
bash
python gen_feat.py mdl_ge_128x384_sample mdl_mut_128x384_sample
Generates normalized cell features from saved embeddings

Output
data/Cell/data/data/cell_feat.npy - Cell line feature matrix

Model Architecture
GINEncoder: Graph Isomorphism Network for PPI graph encoding

Cell2Vec: Cell line embedding model combining PPI features with learnable cell embeddings

Supports both regression and classification tasks

Citation
Xiaowen Wang, et al. "PRODeepSyn: predicting anticancer synergistic drug combinations by embedding cell lines with protein–protein interaction network." Briefings in Bioinformatics, 2022.
