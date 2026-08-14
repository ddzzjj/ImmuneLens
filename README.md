# ImmuneLens

ImmuneLens is an interpretable multimodal representation-learning framework for paired single-cell gene expression (GEX) and T-cell receptor (TCR) sequence data.

The model learns GEX-private, TCR-private, and cross-modal shared representations to integrate transcriptional states with TCR-associated information.

This repository contains the code used for the analyses presented in the ImmuneLens manuscript.

---

## Main analyses

- **Figure 2:** Cross-cohort immune reference atlas construction and query mapping
- **Figure 3:** Antigen specificity prediction using paired GEX and TCR information
- **Figure 4:** Multimodal analysis of T-cell states in a LUAD neoadjuvant immunotherapy cohort
- **Figure 5:** Immunotherapy response prediction and clone-level analysis in LUSC

---

## Repository structure

```text
ImmuneLens/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── immunelens/
│   ├── __init__.py
│   └── model.py
│
├── scripts/
│   ├── train_atlas.py
│   ├── query_bcc.py
│   ├── antigen_prediction.py
│   ├── luad_analysis.py
│   ├── luad_downstream.py
│   ├── lusc_response.py
│   └── lusc_downstream.py
│
├── configs/
│   ├── atlas.yaml
│   ├── query.yaml
│   ├── antigen.yaml
│   └── response.yaml
│
└── docs/
    ├── data_preparation.md
    └── reproducing_results.md
```

The `immunelens/` directory contains the core model components. The `scripts/`
directory contains the training and downstream analysis scripts used for the
main analyses in the manuscript.

Example configuration files are provided under `configs/`, and additional
instructions for data preparation and reproduction are available under `docs/`.

---

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 1.13
- CUDA >= 11.6 recommended for GPU training

Clone the repository and install the required packages:

```bash
git clone https://github.com/ddzzjj/ImmuneLens.git
cd ImmuneLens
pip install -r requirements.txt
```

---

## Data

All datasets analyzed in the manuscript are publicly available.

| Dataset | Accession / Source | Application |
|---|---|---|
| CRC | [GSE236581](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236581) | Reference atlas |
| NSCLC | [GSE243013](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243013) | Reference atlas and immunotherapy analyses |
| COVID-19 | [GSE158055](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158055) | Reference atlas |
| BCC | [GSE123813](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123813) | External query mapping |
| 10x multimodal T-cell dataset | [10x Genomics](https://www.10xgenomics.com/datasets) | Antigen specificity prediction |

Detailed data preparation instructions are provided in:

```text
docs/data_preparation.md
```

---

## Pretrained TCR encoders

ImmuneLens uses pretrained sequence models for TCR representation.

- **TCR-BERT:** [wukevin/tcr-bert](https://huggingface.co/wukevin/tcr-bert)
- **ESM-2:** [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D)

The pretrained models can be downloaded from Hugging Face and placed in a local
directory such as:

```text
pretrained/
├── tcr-bert/
└── esm2/
```

Model paths can be specified in the corresponding scripts or configuration files.

---

## Reproducing the manuscript analyses

The scripts below correspond to the main analyses presented in Figures 2–5.

### Figure 2 — Cross-cohort immune reference atlas

Train the multi-cohort immune reference atlas:

```bash
python scripts/train_atlas.py
```

Map the external BCC dataset onto the reference atlas:

```bash
python scripts/query_bcc.py
```

The reference atlas is constructed using the CRC, NSCLC, and COVID-19 datasets,
with the BCC dataset used for external query mapping.

---

### Figure 3 — Antigen specificity prediction

Run antigen specificity prediction using paired GEX and TCR information:

```bash
python scripts/antigen_prediction.py
```

The analysis supports TCR-BERT and ESM-2 as alternative TCR sequence encoders.

---

### Figure 4 — LUAD multimodal analysis

Run the LUAD multimodal analysis:

```bash
python scripts/luad_analysis.py
```

Perform downstream analysis and plotting:

```bash
python scripts/luad_downstream.py
```

These scripts are used to investigate transcriptional and TCR-associated
heterogeneity in the LUAD neoadjuvant immunotherapy cohort.

---

### Figure 5 — LUSC immunotherapy response analysis

Run immunotherapy response prediction:

```bash
python scripts/lusc_response.py
```

Perform downstream clone-level analysis and plotting:

```bash
python scripts/lusc_downstream.py
```

These analyses evaluate sample-level treatment response prediction and
characterize response-associated CD8 T-cell clones.

---

## Configuration

Example configuration files are provided under:

```text
configs/
```

including:

```text
configs/atlas.yaml
configs/query.yaml
configs/antigen.yaml
configs/response.yaml
```

Dataset paths, pretrained model locations, output directories, and
analysis-specific parameters can be adjusted through these files or through
command-line arguments where supported.

For available command-line options, use:

```bash
python scripts/<script_name>.py --help
```

---

## Data preparation

The datasets used in the manuscript originate from different studies and require
study-specific preprocessing and metadata organization.

Detailed instructions are provided in:

```text
docs/data_preparation.md
```

This document describes the expected input files and preprocessing steps for the
CRC, NSCLC, COVID-19, BCC, and 10x Genomics datasets.

---

## Reproducing paper results

A more detailed figure-by-figure guide is provided in:

```text
docs/reproducing_results.md
```

This document describes how the scripts correspond to the analyses presented in
Figures 2–5 and provides additional information about required inputs and
generated outputs.

---

## Notes

This repository is being prepared alongside the ImmuneLens manuscript.

The current version contains the core model implementation and analysis scripts
used for the main results presented in the manuscript. Documentation and
reproducibility instructions may be further updated during manuscript
development and peer review.

---

## License

This project is released under the MIT License.

See `LICENSE` for details.

---

## Contact

For questions about the code, please open an issue in this repository.

For questions about the manuscript, please contact the corresponding author
listed in the manuscript.