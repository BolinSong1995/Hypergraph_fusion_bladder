# CT-Fusion

Reference implementation of the CT-Fusion branch used in the IMPACT-Fusion
framework for progression-free survival (PFS) modeling in muscle-invasive
bladder cancer.

CT-Fusion integrates CT tumor subvolume representations and WSI patch-level
representations using a feature-similarity-based hypergraph. The implementation
is designed to operate on precomputed features, allowing external evaluation
without distribution of institutional raw imaging or pathology data.

## Architecture

For each patient:

- CT input: four 768-dimensional tumor subvolume embeddings.
- WSI input: a variable number of 768-dimensional patch embeddings.
- Both modalities are projected to a shared 256-dimensional latent space.
- Non-negative cosine similarity is calculated between each CT node and all
  available WSI patch nodes.
- The top 10 WSI patches are selected for each CT node.
- The union of selected patches forms a compact WSI node set (maximum 40).
- Six hyperedges are constructed:
  - one intra-CT hyperedge,
  - one intra-WSI hyperedge,
  - four CT-centered cross-modal hyperedges.
- One hypergraph convolution layer performs message passing.
- Learnable attention pooling creates a patient-level representation.
- A Cox risk head produces a continuous CT-Fusion prognostic score.

## Repository structure

```text
CT-Fusion/
├── ct_fusion/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── engine.py
│   ├── inference.py
│   ├── model.py
│   └── utils.py
├── examples/
│   └── README.md
├── scripts/
│   └── inspect_checkpoint.py
├── train.py
├── predict.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone Hypergraph_fusion_bladder
cd CT-Fusion
pip install -r requirements.txt
```

## Expected data format

The training clinical CSV must contain:

```text
ID, WSI, PFS, PFS_status
```

Each patient must have:

**CT features**

```text
<ct_feature_dir>/<ID>.pt
```

containing a PyTorch tensor of shape:

```text
[4, 768]
```

**WSI features**

```text
<wsi_feature_dir>/<WSI>.csv
```

containing:

```text
patch_file,feature_0,...,feature_767
```

See `examples/README.md`.

## Training

```bash
python train.py       --clinical-csv /path/to/clinical.csv       --ct-feature-dir /path/to/ct_features       --wsi-feature-dir /path/to/wsi_features       --output-dir /path/to/output
```

Default training settings reproduce the manuscript-described configuration:

- random seed: 1337
- development/internal validation split: 5:1
- top-K WSI patches per CT node: 10
- shared latent dimension: 256
- hypergraph layers: 1
- AdamW learning rate: 1e-4
- weight decay: 1e-5
- Cox patient accumulation size: 16
- maximum epochs: 50

The best checkpoint is selected by C-index on the internal validation cohort.

## Single-patient inference

```bash
python predict.py       --checkpoint /path/to/ct_fusion_hypergraph_best.pt       --ct /path/to/patient.pt       --wsi /path/to/patient.csv
```

## Checkpoint inspection

```bash
python scripts/inspect_checkpoint.py /path/to/checkpoint.pt
```


## Data availability

Institutional patient-level data are not included in this repository because
their release may be restricted by institutional review board and data-use
requirements.

The repository starts from precomputed CT and WSI representations and
reproduces the CT-Fusion hypergraph model, training, checkpoint selection,
and inference stages.
