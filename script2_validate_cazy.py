#!/usr/bin/env python3
"""
Script 2: Validate the trained binding affinity model on CAZy GT2/glycosyltransferase data.

This script:
  1. Loads the CAZy-Dataset-3.csv (650 GT2-family enzyme–substrate pairs)
  2. Embeds protein sequences with ProtT5-XL (masked mean pooling → HDF5)
  3. Embeds ligand SMILES with ChemBERTa (masked mean pooling → HDF5)
  4. Loads the best checkpoint from Script 1
  5. Predicts binding affinity for all CAZy pairs
  6. Compares predictions vs. experimental binding affinities
  7. Reports RMSE, Pearson R, Spearman ρ, scatter plot
  8. Outputs: cazy_validation_results.csv, cazy_scatter.png

Constraints:
  - RTX 3070 (8 GB VRAM), Windows 10/11, Python 3.10
  - T5Tokenizer for ProtT5-XL, RobertaTokenizer for ChemBERTa
  - Masked mean pooling (not CLS) everywhere
  - HDF5 streaming for embeddings
  - num_workers=0 in all DataLoaders
"""

import os
import sys
import gc
import json
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── paths (auto-detect: script directory is the project root) ────────────
WORK = Path(__file__).resolve().parent
CAZY_CSV = WORK / "CAZy Dataset.csv"
CKPT_PATH = WORK / "checkpoints" / "best_affinity_model.pt"
EMB_DIR = WORK / "embeddings"
EMB_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = WORK / "cazy_validation_results.csv"
OUTPUT_PLOT = WORK / "cazy_scatter.png"

# ── model config (must match Script 1) ──
PROT_MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
LIG_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
PROT_EMB_DIM = 1024
LIG_EMB_DIM = 768
PROJ_DIM = 256
N_CROSS_HEADS = 4
HIDDEN_DIM = 512
DROPOUT = 0.1
MAX_PROT_LEN = 1024
MAX_LIG_LEN = 128
EMB_BATCH = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════
# SHARED COMPONENTS (same as Script 1)
# ═══════════════════════════════════════════════════════════════════════════

def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling: average over non-pad tokens only."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_proteins_to_hdf5(sequences: list[str], h5_path: Path, batch_size: int = EMB_BATCH):
    """Embed protein sequences with ProtT5-XL using masked mean pooling → HDF5."""
    from transformers import T5Tokenizer, T5EncoderModel

    log.info("Loading ProtT5-XL tokenizer + encoder …")
    tokenizer = T5Tokenizer.from_pretrained(PROT_MODEL_NAME, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROT_MODEL_NAME)
    model.eval()
    model.half().to(DEVICE)

    n = len(sequences)
    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset("embeddings", shape=(n, PROT_EMB_DIM), dtype="float32")
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_seqs = sequences[start:end]
            spaced = [" ".join(list(s[:MAX_PROT_LEN])) for s in batch_seqs]
            enc = tokenizer(
                spaced, padding=True, truncation=True,
                max_length=MAX_PROT_LEN + 2, return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(DEVICE)
            attn_mask = enc["attention_mask"].to(DEVICE)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
                out = model(input_ids=input_ids, attention_mask=attn_mask)
            pooled = masked_mean_pool(out.last_hidden_state.float(), attn_mask.float())
            ds[start:end] = pooled.cpu().numpy()
            if (start // batch_size) % 20 == 0:
                log.info(f"  Protein embeddings: {end}/{n}")
            del input_ids, attn_mask, out, pooled
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    log.info(f"Protein embeddings saved → {h5_path}")


def embed_ligands_to_hdf5(smiles_list: list[str], h5_path: Path, batch_size: int = EMB_BATCH):
    """Embed SMILES with ChemBERTa (RobertaTokenizer) using masked mean pooling → HDF5."""
    from transformers import RobertaTokenizer, RobertaModel

    log.info("Loading ChemBERTa tokenizer + model …")
    tokenizer = RobertaTokenizer.from_pretrained(LIG_MODEL_NAME)
    model = RobertaModel.from_pretrained(LIG_MODEL_NAME)
    model.eval().half().to(DEVICE)

    n = len(smiles_list)
    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset("embeddings", shape=(n, LIG_EMB_DIM), dtype="float32")
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_smi = smiles_list[start:end]
            enc = tokenizer(
                batch_smi, padding=True, truncation=True,
                max_length=MAX_LIG_LEN, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(DEVICE)
            attn_mask = enc["attention_mask"].to(DEVICE)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
                out = model(input_ids=input_ids, attention_mask=attn_mask)
            pooled = masked_mean_pool(out.last_hidden_state.float(), attn_mask.float())
            ds[start:end] = pooled.cpu().numpy()
            if (start // batch_size) % 20 == 0:
                log.info(f"  Ligand embeddings: {end}/{n}")
            del input_ids, attn_mask, out, pooled
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    log.info(f"Ligand embeddings saved → {h5_path}")


class HDF5AffinityDataset(Dataset):
    """Streams pre-computed protein + ligand embeddings from HDF5."""

    def __init__(self, prot_h5: Path, lig_h5: Path, labels: np.ndarray, indices: np.ndarray):
        self.prot_h5_path = str(prot_h5)
        self.lig_h5_path = str(lig_h5)
        self.labels = labels.astype(np.float32)
        self.indices = indices.astype(np.int64)
        self._prot_h5 = None
        self._lig_h5 = None

    def _open(self):
        if self._prot_h5 is None:
            self._prot_h5 = h5py.File(self.prot_h5_path, "r")
            self._lig_h5 = h5py.File(self.lig_h5_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open()
        real_idx = int(self.indices[idx])
        prot_emb = self._prot_h5["embeddings"][real_idx]
        lig_emb = self._lig_h5["embeddings"][real_idx]
        label = self.labels[idx]
        return (
            torch.tensor(prot_emb, dtype=torch.float32),
            torch.tensor(lig_emb, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

    def __del__(self):
        if self._prot_h5 is not None:
            self._prot_h5.close()
        if self._lig_h5 is not None:
            self._lig_h5.close()


class GatedCrossAttention(nn.Module):
    """Gated cross-attention fusion (identical to Script 1)."""

    def __init__(self, prot_dim, lig_dim, proj_dim, n_heads, dropout=0.1):
        super().__init__()
        self.proj_prot = nn.Linear(prot_dim, proj_dim)
        self.proj_lig = nn.Linear(lig_dim, proj_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.gate_net = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim), nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prot_emb, lig_emb):
        p = self.proj_prot(prot_emb)
        l = self.proj_lig(lig_emb)
        p_seq = p.unsqueeze(1)
        l_seq = l.unsqueeze(1)
        attn_out, _ = self.cross_attn(p_seq, l_seq, l_seq)
        attn_out = attn_out.squeeze(1)
        gate = self.gate_net(torch.cat([p, l], dim=-1))
        fused = gate * attn_out
        fused = self.layer_norm(fused + p)
        fused = self.dropout(fused)
        return fused


class AffinityPredictor(nn.Module):
    """Full regression model (identical to Script 1)."""

    def __init__(self):
        super().__init__()
        self.fusion = GatedCrossAttention(
            PROT_EMB_DIM, LIG_EMB_DIM, PROJ_DIM, N_CROSS_HEADS, DROPOUT
        )
        self.regressor = nn.Sequential(
            nn.Linear(PROJ_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, prot_emb, lig_emb):
        fused = self.fusion(prot_emb, lig_emb)
        return self.regressor(fused).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("SCRIPT 2: Validate on CAZy GT2 glycosyltransferase dataset")
    log.info("=" * 70)

    # ── 1. Load CAZy data ────────────────────────────────────────────────
    log.info(f"Loading CAZy data: {CAZY_CSV}")
    df = pd.read_csv(CAZY_CSV)
    log.info(f"Raw CAZy data: {len(df)} rows, columns: {list(df.columns)}")

    # Keep only rows with valid sequence, SMILES, and binding affinity
    required_cols = ["Sequence", "SMILES", "Binding_affinity"]
    df = df.dropna(subset=required_cols)
    df = df[df["Sequence"].str.len() > 0]
    df = df[df["SMILES"].str.len() > 0]
    df = df.reset_index(drop=True)
    log.info(f"After cleaning: {len(df)} rows")

    # ── 2. Convert CAZy binding affinity to same scale as JGlaser ────────
    # CAZy data has docking scores (kcal/mol, negative), e.g. -7.06
    # JGlaser uses neg_log10_affinity_M (pKd scale, ~0-15, typically 4-10)
    # Convert: ΔG = RT·ln(Kd)  →  pKd = -log10(Kd) = -ΔG / (2.303·RT)
    # At 298K: RT = 0.5922 kcal/mol  →  pKd = -ΔG / 1.3633
    R = 1.987e-3  # kcal/(mol·K)
    T = 298.15     # Kelvin
    RT = R * T     # 0.5922 kcal/mol
    factor = 2.303 * RT  # 1.3633

    df["pKd_experimental"] = -df["Binding_affinity"] / factor
    log.info(f"Converted ΔG (kcal/mol) → pKd:")
    log.info(f"  ΔG range: [{df['Binding_affinity'].min():.2f}, {df['Binding_affinity'].max():.2f}]")
    log.info(f"  pKd range: [{df['pKd_experimental'].min():.2f}, {df['pKd_experimental'].max():.2f}]")

    sequences = df["Sequence"].tolist()
    smiles = df["SMILES"].tolist()
    labels = df["pKd_experimental"].values.astype(np.float32)

    # ── 3. Embed proteins → HDF5 ────────────────────────────────────────
    prot_h5 = EMB_DIR / "cazy_prot_emb.h5"
    if prot_h5.exists():
        log.info(f"CAZy protein embeddings exist: {prot_h5}")
    else:
        embed_proteins_to_hdf5(sequences, prot_h5)

    # ── 4. Embed ligands → HDF5 ─────────────────────────────────────────
    lig_h5 = EMB_DIR / "cazy_lig_emb.h5"
    if lig_h5.exists():
        log.info(f"CAZy ligand embeddings exist: {lig_h5}")
    else:
        embed_ligands_to_hdf5(smiles, lig_h5)

    # ── 5. Load trained model ────────────────────────────────────────────
    log.info(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

    # Reconstruct hyperparams from checkpoint
    hp = ckpt.get("hyperparams", {})
    log.info(f"Checkpoint hyperparams: {hp}")
    log.info(f"Checkpoint val_rmse: {ckpt.get('val_rmse', 'N/A')}")
    log.info(f"Checkpoint val_pearson: {ckpt.get('val_pearson', 'N/A')}")

    model = AffinityPredictor().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Model loaded successfully")

    # ── 6. Run inference on CAZy data ────────────────────────────────────
    n = len(labels)
    indices = np.arange(n)
    cazy_ds = HDF5AffinityDataset(prot_h5, lig_h5, labels, indices)
    cazy_loader = DataLoader(cazy_ds, batch_size=64, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for prot, lig, lab in cazy_loader:
            prot, lig = prot.to(DEVICE), lig.to(DEVICE)
            preds = model(prot, lig)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(lab.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # ── 7. Compute metrics ───────────────────────────────────────────────
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    mae = np.mean(np.abs(all_preds - all_labels))
    pearson_r, pearson_p = stats.pearsonr(all_preds, all_labels)
    spearman_rho, spearman_p = stats.spearmanr(all_preds, all_labels)

    log.info("=" * 50)
    log.info("CAZy GT2 VALIDATION RESULTS")
    log.info("=" * 50)
    log.info(f"  N samples:     {n}")
    log.info(f"  RMSE:          {rmse:.4f}")
    log.info(f"  MAE:           {mae:.4f}")
    log.info(f"  Pearson R:     {pearson_r:.4f} (p={pearson_p:.2e})")
    log.info(f"  Spearman ρ:    {spearman_rho:.4f} (p={spearman_p:.2e})")
    log.info("=" * 50)

    # ── 8. Per-ligand breakdown ──────────────────────────────────────────
    df["pred_pKd"] = all_preds
    df["residual"] = all_preds - all_labels

    log.info("\nPer-ligand validation metrics:")
    for lig_name, grp in df.groupby("Ligand_name"):
        if len(grp) < 3:
            continue
        g_preds = grp["pred_pKd"].values
        g_labels = grp["pKd_experimental"].values
        g_rmse = np.sqrt(np.mean((g_preds - g_labels) ** 2))
        g_r, _ = stats.pearsonr(g_preds, g_labels) if len(grp) > 2 else (0.0, 1.0)
        log.info(f"  {lig_name:20s} | n={len(grp):3d} | RMSE={g_rmse:.3f} | R={g_r:.3f}")

    # ── 9. Save results CSV ──────────────────────────────────────────────
    result_df = df[["Protein_name", "Chain_ID", "Ligand_name", "Binding_affinity",
                     "pKd_experimental", "pred_pKd", "residual"]].copy()
    result_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"\nResults saved → {OUTPUT_CSV}")

    # ── 10. Scatter plot ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # (a) Predicted vs Experimental
        ax = axes[0]
        ax.scatter(all_labels, all_preds, alpha=0.4, s=15, c="steelblue", edgecolors="none")
        lims = [min(all_labels.min(), all_preds.min()) - 0.5,
                max(all_labels.max(), all_preds.max()) + 0.5]
        ax.plot(lims, lims, "--", color="gray", linewidth=1)
        ax.set_xlabel("Experimental pKd (from docking ΔG)", fontsize=11)
        ax.set_ylabel("Predicted pKd", fontsize=11)
        ax.set_title(f"CAZy GT2 Validation\nRMSE={rmse:.3f}  R={pearson_r:.3f}  ρ={spearman_rho:.3f}", fontsize=12)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")

        # (b) Residual distribution
        ax = axes[1]
        ax.hist(all_preds - all_labels, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (Predicted − Experimental)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Residual Distribution", fontsize=12)

        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Scatter plot saved → {OUTPUT_PLOT}")
    except ImportError:
        log.warning("matplotlib not available; skipping scatter plot.")

    # ── 11. Save validation metrics ──────────────────────────────────────
    val_metrics = {
        "dataset": "CAZy-GT2",
        "n_samples": int(n),
        "rmse": float(rmse),
        "mae": float(mae),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }
    with open(WORK / "cazy_validation_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    log.info(f"Validation metrics saved → {WORK / 'cazy_validation_metrics.json'}")

    log.info("=" * 70)
    log.info("Script 2 complete.")


if __name__ == "__main__":
    main()
