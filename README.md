# Protein-Ligand Affinity Modeling for PmHAS Variant Screening

This repository contains a local GPU workflow for training and applying a protein-ligand affinity prediction model to support screening of *Pasteurella multocida* hyaluronan synthase (PmHAS) mutants.

## Objective

The main objective is to build a computational pipeline that can learn protein-ligand affinity patterns on a large benchmark dataset and then apply the trained model to prioritize PmHAS mutants for downstream experimental analysis.

## Workflow

### Script 1: Train the affinity model
`script1_train_jglaser.py`

- Trains a sequence-based protein-ligand affinity predictor on the JGlaser binding affinity dataset
- Uses ProtT5-XL for protein embeddings
- Uses ChemBERTa for ligand embeddings
- Uses a gated cross-attention fusion head
- Saves:
  - `checkpoints/best_affinity_model.pt`
  - `train_metrics.json`

### Script 3: Screen PmHAS mutants
`script3_screen_pmhas.py`

- Loads the PmHAS mutant dataset
- Identifies the real wild-type sequence
- Labels mutants relative to wild type
- Embeds mutant sequences
- Predicts affinity for selected substrates
- Saves ranked mutant tables and summary outputs

### Script 4: Evaluate on PmHAS docking scores
`script4_eval_pmhas.py`

- Compares model predictions against docking-derived pseudo-ground-truth
- Reports RMSE, Pearson correlation, Spearman correlation, and R² where available
- Intended as an initial downstream transfer evaluation

### Script 5: Plot learning curves
`script5_plot_learning_curves.py`

- Reads `train_metrics.json`
- Generates a thesis-ready learning curve figure
- Saves:
  - `learning_curves/thesis_learning_curve.png`
  - `learning_curves/best_epoch_summary.csv`

## Environment

- Python 3.10
- Windows 10/11
- NVIDIA RTX 3070 class GPU
- Local execution only; no HPC required

## Installation

```bash
pip install -r requirements.txt
```

## Example usage

```bash
python script1_train_jglaser.py
python script3_screen_pmhas.py
python script4_eval_pmhas.py
python script5_plot_learning_curves.py
```

## Repository contents to keep

Recommended for version control:
- source scripts
- `train_metrics.json`
- figure outputs used in thesis
- lightweight summary CSV/JSON files

Not recommended for version control:
- large HDF5 embedding files
- temporary caches
- large intermediate output folders unless needed as supplementary data

## Thesis note

The model is validated first on a source affinity dataset and then transferred to the PmHAS mutant dataset. Current downstream transfer to docking-based PmHAS scores is part of ongoing model refinement.
