#!/usr/bin/env python3
"""
Script 4: Evaluate the trained affinity model on pmHAS docking scores.

Goal:
- Treat the docking scores in "pmHAS 8000 Mutants Data.csv" as pseudo-ground-truth.
- For each substrate (GlcNAc, UDP Glc, GalNAc), compute:
  - RMSE(model_pred, docking_score)
  - Pearson r
  - R²
  - Spearman rho
- Save metrics and per-substrate predictions.

Important:
- Uses checkpoint hyperparameters from best_affinity_model.pt
- Reuses helper functions from the updated script3_screen_pmhas.py
- Rebuilds stale pmHAS embeddings automatically when dimensions mismatch
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch

from script3_screen_pmhas import (
    WORK,
    PMHAS_CSV,
    EMB_DIR,
    CKPT_PATH,
    SUBSTRATE_SMILES,
    DEVICE,
    embed_smiles_list,
    AffinityPredictor,
    load_model_hparams_from_checkpoint,
    ensure_pmhas_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

OUTPUT_DIR = WORK / "pmhas_eval"
OUTPUT_DIR.mkdir(exist_ok=True)


def rankdata_average(a):
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    arr = a[sorter]

    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = np.cumsum(obs) - 1
    count = np.bincount(dense)
    cumulative = np.cumsum(count)
    starts = cumulative - count
    avg_ranks = (starts + cumulative - 1) / 2.0 + 1.0
    return avg_ranks[dense][inv]


def spearmanr_np(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))

    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        pearson = float("nan")
        r2 = float("nan")
        spearman = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        spearman = spearmanr_np(y_true, y_pred)

    return {
        "rmse": rmse,
        "pearson": pearson,
        "r2": r2,
        "spearman": spearman,
        "n": int(len(y_true)),
        "y_true_mean": float(y_true.mean()),
        "y_true_std": float(y_true.std()),
        "y_pred_mean": float(y_pred.mean()),
        "y_pred_std": float(y_pred.std()),
    }


def get_true_scores(df, sub_name):
    if sub_name == "GlcNAc":
        return df["GlcNAc"].values.astype(np.float32)
    if sub_name == "UDP_Glc":
        return df["UDP Glc"].values.astype(np.float32)
    if sub_name == "GalNAc":
        return df["GalNAc"].values.astype(np.float32)
    raise ValueError(f"Unknown substrate column mapping for {sub_name}")


def main():
    log.info("=" * 70)
    log.info("SCRIPT 4: Evaluate trained model vs pmHAS docking scores")
    log.info("=" * 70)
    log.info(f"Device: {DEVICE}")

    if not CKPT_PATH.exists():
        log.error(f"Checkpoint not found: {CKPT_PATH}")
        sys.exit(1)

    log.info(f"Loading checkpoint: {CKPT_PATH}")
    ckpt, hp = load_model_hparams_from_checkpoint(CKPT_PATH)

    PROT_EMB_DIM = hp["PROT_EMB_DIM"]
    LIG_EMB_DIM = hp["LIG_EMB_DIM"]
    PROJ_DIM = hp["PROJ_DIM"]
    N_CROSS_HEADS = hp["N_CROSS_HEADS"]
    HIDDEN_DIM = hp["HIDDEN_DIM"]
    DROPOUT = hp["DROPOUT"]

    log.info(
        f"Checkpoint hyperparams -> PROT_EMB_DIM={PROT_EMB_DIM}, "
        f"LIG_EMB_DIM={LIG_EMB_DIM}, PROJ_DIM={PROJ_DIM}, "
        f"N_CROSS_HEADS={N_CROSS_HEADS}, HIDDEN_DIM={HIDDEN_DIM}, DROPOUT={DROPOUT}"
    )

    log.info(f"Loading pmHAS data: {PMHAS_CSV}")
    df = pd.read_csv(PMHAS_CSV)
    df = df.dropna(subset=["Receptor", "GlcNAc", "UDP Glc", "GalNAc"])
    df = df[df["Receptor"].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    df = df.reset_index(drop=True)
    log.info(f"After cleaning: {len(df)} rows")

    sequences = df["Receptor"].tolist()
    n_mutants = len(sequences)

    prot_h5 = EMB_DIR / "pmhas_prot_emb.h5"
    ensure_pmhas_embeddings(sequences, prot_h5, PROT_EMB_DIM)

    substrate_names = list(SUBSTRATE_SMILES.keys())
    substrate_smiles = [SUBSTRATE_SMILES[s] for s in substrate_names]
    log.info(f"Embedding substrates: {substrate_names}")
    substrate_embs = embed_smiles_list(substrate_smiles)
    log.info(f"Substrate embeddings shape: {substrate_embs.shape}")

    if substrate_embs.shape[1] != LIG_EMB_DIM:
        log.error(
            f"Ligand embedding dim mismatch: got {substrate_embs.shape[1]}, checkpoint expects {LIG_EMB_DIM}"
        )
        sys.exit(1)

    model = AffinityPredictor(
        prot_dim=PROT_EMB_DIM,
        lig_dim=LIG_EMB_DIM,
        proj_dim=PROJ_DIM,
        n_heads=N_CROSS_HEADS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Model loaded successfully")

    metrics = {}
    prot_h5_file = h5py.File(prot_h5, "r")
    prot_emb_ds = prot_h5_file["embeddings"]
    batch_size = 512

    for sub_idx, sub_name in enumerate(substrate_names):
        log.info(f"Evaluating substrate: {sub_name}")
        sub_emb = torch.tensor(substrate_embs[sub_idx], dtype=torch.float32).to(DEVICE)
        y_true = get_true_scores(df, sub_name)

        preds = []
        for start in range(0, n_mutants, batch_size):
            end = min(start + batch_size, n_mutants)
            prot_batch = torch.tensor(prot_emb_ds[start:end], dtype=torch.float32).to(DEVICE)
            lig_batch = sub_emb.unsqueeze(0).expand(end - start, -1)

            with torch.no_grad():
                pred = model(prot_batch, lig_batch)
                preds.append(pred.cpu().numpy())

            del prot_batch, lig_batch, pred
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        y_pred = np.concatenate(preds)
        sub_metrics = compute_metrics(y_true, y_pred)
        metrics[sub_name] = sub_metrics

        log.info(
            f" {sub_name}: RMSE={sub_metrics['rmse']:.4f}, "
            f"r={sub_metrics['pearson']:.4f}, "
            f"R²={sub_metrics['r2']:.4f}, "
            f"rho={sub_metrics['spearman']:.4f}, "
            f"n={sub_metrics['n']}"
        )

        pred_df = pd.DataFrame({
            "sequence": sequences,
            "docking_score": y_true,
            "predicted_affinity": y_pred,
            "residual": y_true - y_pred,
        })
        pred_path = OUTPUT_DIR / f"pmhas_predictions_{sub_name}.csv"
        pred_df.to_csv(pred_path, index=False)
        log.info(f" Saved predictions -> {pred_path}")

    prot_h5_file.close()

    metrics["checkpoint_hyperparams"] = hp
    metrics["n_mutants"] = int(n_mutants)

    metrics_path = OUTPUT_DIR / "pmhas_model_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved -> {metrics_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for sub_name in substrate_names:
            pred_path = OUTPUT_DIR / f"pmhas_predictions_{sub_name}.csv"
            pred_df = pd.read_csv(pred_path)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(
                pred_df["docking_score"],
                pred_df["predicted_affinity"],
                s=10,
                alpha=0.5,
                edgecolors="none"
            )
            ax.set_xlabel("Docking score")
            ax.set_ylabel("Predicted affinity")
            ax.set_title(
                f"{sub_name}\n"
                f"r={metrics[sub_name]['pearson']:.3f}, "
                f"R²={metrics[sub_name]['r2']:.3f}"
            )

            x = pred_df["docking_score"].values
            y = pred_df["predicted_affinity"].values
            if len(x) > 1 and np.std(x) > 1e-12:
                m, b = np.polyfit(x, y, 1)
                xx = np.linspace(x.min(), x.max(), 100)
                yy = m * xx + b
                ax.plot(xx, yy, linewidth=2)

            fig.tight_layout()
            fig_path = OUTPUT_DIR / f"pmhas_scatter_{sub_name}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(f" Saved scatter plot -> {fig_path}")

    except ImportError:
        log.warning("matplotlib not available; skipping scatter plots.")

    log.info("=" * 70)
    log.info("Script 4 complete.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()