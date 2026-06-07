#!/usr/bin/env python3
"""
Script 1: Train a sequence-based protein–ligand binding affinity predictor
on the JGlaser binding_affinity dataset.

Architecture:
  - ProtT5-XL-U50 (protein encoder) with masked mean pooling → 1024-d
  - ChemBERTa (ligand encoder) with masked mean pooling → 768-d
  - Gated cross-attention fusion head
  - Regression MLP → scalar pKd

Constraints:
  - RTX 3070 (8 GB VRAM), Windows 10/11, Python 3.10
  - HDF5 streaming for embeddings — never np.concatenate on full dataset
  - T5Tokenizer for ProtT5-XL, RobertaTokenizer for ChemBERTa
  - num_workers=0 in all DataLoaders
  - Masked mean pooling (not CLS) for all transformer encoders

Pipeline:
  Phase 1 — Pre-embed proteins and ligands into HDF5 shards (CPU/GPU batch)
  Phase 2 — Train gated cross-attention regression head from HDF5
"""

import os
import sys
import gc
import math
import time
import json
import logging
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── paths (auto-detect: script directory is the project root) ────────────
WORK = Path(__file__).resolve().parent
PARQUET = WORK / "bapulm_data.parquet"
EMB_DIR = WORK / "embeddings"
EMB_DIR.mkdir(exist_ok=True)
CKPT_DIR = WORK / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
METRICS_PATH = WORK / "train_metrics.json"

# ── hyperparameters ──────────────────────────────────────────────────────
PROT_MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
LIG_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
PROT_EMB_DIM = 1024        # ProtT5-XL hidden size
LIG_EMB_DIM = 768          # ChemBERTa hidden size
PROJ_DIM = 256             # shared projection dimension
N_CROSS_HEADS = 4
HIDDEN_DIM = 512
DROPOUT = 0.1

MAX_PROT_LEN = 512        # truncate proteins longer than this
MAX_LIG_LEN = 128          # truncate SMILES longer than this (tokens)

EMB_BATCH = 1             # batch size for embedding phase
TRAIN_BATCH = 256
VAL_BATCH = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 40
PATIENCE = 8
EMB_SHARD_SIZE = 50_000    # rows per HDF5 shard

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

# ── Subsample JGlaser for feasibility on single GPU ─────────────────────
# 1.8M rows is too large to embed entirely; subsample to 200k for training
SUBSAMPLE_N = 200_000
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: EMBED INTO HDF5
# ═══════════════════════════════════════════════════════════════════════════

def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling: average over non-pad tokens only."""
    mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
    summed = (hidden_states * mask).sum(dim=1)   # (B, D)
    counts = mask.sum(dim=1).clamp(min=1e-9)     # (B, 1)
    return summed / counts                        # (B, D)


def embed_proteins_to_hdf5(sequences: list[str], h5_path: Path, batch_size: int = EMB_BATCH):
    """Embed protein sequences with ProtT5-XL using masked mean pooling → HDF5 (OOM-safe for 8 GB)."""
    from transformers import T5Tokenizer, T5EncoderModel

    log.info("Loading ProtT5-XL tokenizer + encoder …")
    tokenizer = T5Tokenizer.from_pretrained(PROT_MODEL_NAME, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROT_MODEL_NAME)

    # OOM‑safety features
    model.gradient_checkpointing_enable()           # reduce activations
    model.eval()
    model.to(DEVICE)
    model.half()                                    # fp16 weights on GPU
    torch.set_grad_enabled(False)

    n = len(sequences)
    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset("embeddings", shape=(n, PROT_EMB_DIM), dtype="float32")

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_seqs = sequences[start:end]

            # ProtT5 expects spaces between amino acids
            spaced = [" ".join(list(s[:MAX_PROT_LEN])) for s in batch_seqs]

            enc = tokenizer(
                spaced,
                padding=True,
                truncation=True,
                max_length=MAX_PROT_LEN + 2,
                return_tensors="pt",
                add_special_tokens=True,
            )

            input_ids = enc["input_ids"].to(DEVICE)
            attn_mask = enc["attention_mask"].to(DEVICE)

            # STRICT autocast on GPU
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                hidden = out.last_hidden_state  # fp16

            # mean pool in fp32 on GPU, then move to CPU
            mask_f = attn_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
            pooled = pooled.float().cpu().numpy()

            ds[start:end] = pooled

            if (start // batch_size) % 50 == 0:
                log.info(f" Protein embeddings: {end}/{n}")

            # free GPU memory
            del enc, input_ids, attn_mask, out, hidden, mask_f, pooled
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    log.info(f"Protein embeddings saved → {h5_path}  ({n} vectors)")


def embed_ligands_to_hdf5(smiles_list: list[str], h5_path: Path, batch_size: int = EMB_BATCH):
    """Embed SMILES with ChemBERTa (RobertaTokenizer) using masked mean pooling → HDF5."""
    from transformers import RobertaTokenizer, RobertaModel

    log.info(f"Loading ChemBERTa tokenizer + model …")
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
                batch_smi,
                padding=True,
                truncation=True,
                max_length=MAX_LIG_LEN,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(DEVICE)
            attn_mask = enc["attention_mask"].to(DEVICE)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
                out = model(input_ids=input_ids, attention_mask=attn_mask)
            pooled = masked_mean_pool(out.last_hidden_state.float(), attn_mask.float())
            ds[start:end] = pooled.cpu().numpy()

            if (start // batch_size) % 50 == 0:
                log.info(f"  Ligand embeddings: {end}/{n}")
            del input_ids, attn_mask, out, pooled
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    log.info(f"Ligand embeddings saved → {h5_path}  ({n} vectors)")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: TRAINING
# ═══════════════════════════════════════════════════════════════════════════

class HDF5AffinityDataset(Dataset):
    """Streams pre-computed protein + ligand embeddings and affinity labels from HDF5."""

    def __init__(self, prot_h5: Path, lig_h5: Path, labels: np.ndarray, indices: np.ndarray):
        self.prot_h5_path = str(prot_h5)
        self.lig_h5_path = str(lig_h5)
        self.labels = labels.astype(np.float32)
        self.indices = indices.astype(np.int64)
        # We open file handles lazily per-worker (Windows safe)
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
    """
    Gated cross-attention fusion for protein and ligand embeddings.

    Q = protein projection, K/V = ligand projection
    Gate = sigmoid(FFN([prot_proj ; lig_proj]))
    Output = gate ⊙ CrossAttn(Q, K, V)
    """

    def __init__(self, prot_dim: int, lig_dim: int, proj_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.proj_prot = nn.Linear(prot_dim, proj_dim)
        self.proj_lig = nn.Linear(lig_dim, proj_dim)
        # Multi-head cross-attention (query=prot, key/value=lig)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prot_emb: torch.Tensor, lig_emb: torch.Tensor) -> torch.Tensor:
        """
        prot_emb: (B, prot_dim)  — pooled protein embedding
        lig_emb:  (B, lig_dim)   — pooled ligand embedding
        returns:  (B, proj_dim)  — fused representation
        """
        p = self.proj_prot(prot_emb)  # (B, proj_dim)
        l = self.proj_lig(lig_emb)    # (B, proj_dim)

        # Reshape to (B, 1, proj_dim) for attention (single-token sequence)
        p_seq = p.unsqueeze(1)
        l_seq = l.unsqueeze(1)

        # Cross-attention: protein queries, ligand keys/values
        attn_out, _ = self.cross_attn(p_seq, l_seq, l_seq)  # (B, 1, proj_dim)
        attn_out = attn_out.squeeze(1)                        # (B, proj_dim)

        # Gating
        gate = self.gate_net(torch.cat([p, l], dim=-1))       # (B, proj_dim)
        fused = gate * attn_out                                # (B, proj_dim)

        fused = self.layer_norm(fused + p)  # residual from protein
        fused = self.dropout(fused)
        return fused


class AffinityPredictor(nn.Module):
    """Full regression model: GatedCrossAttention → MLP → scalar."""

    def __init__(self):
        super().__init__()
        self.fusion = GatedCrossAttention(
            prot_dim=PROT_EMB_DIM,
            lig_dim=LIG_EMB_DIM,
            proj_dim=PROJ_DIM,
            n_heads=N_CROSS_HEADS,
            dropout=DROPOUT,
        )
        self.regressor = nn.Sequential(
            nn.Linear(PROJ_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, prot_emb, lig_emb):
        fused = self.fusion(prot_emb, lig_emb)
        return self.regressor(fused).squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for prot, lig, labels in loader:
        prot, lig, labels = prot.to(device), lig.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(prot, lig)
        loss = criterion(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    n = 0
    for prot, lig, labels in loader:
        prot, lig, labels = prot.to(device), lig.to(device), labels.to(device)
        preds = model(prot, lig)
        loss = criterion(preds, labels)
        total_loss += loss.item() * len(labels)
        n += len(labels)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    pearson = np.corrcoef(all_preds, all_labels)[0, 1]
    return total_loss / n, rmse, pearson


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("SCRIPT 1: Train binding affinity model on JGlaser dataset")
    log.info("=" * 70)

    # ── 1. Load and subsample data ───────────────────────────────────────
    log.info(f"Loading parquet: {PARQUET}")
    df = pd.read_parquet(PARQUET, columns=["seq", "smiles_can", "neg_log10_affinity_M"])
    log.info(f"Full dataset: {len(df)} rows")

    # Drop rows with NaN or empty sequences
    df = df.dropna(subset=["seq", "smiles_can", "neg_log10_affinity_M"])
    df = df[df["seq"].str.len() > 0]
    df = df[df["smiles_can"].str.len() > 0]
    log.info(f"After cleaning: {len(df)} rows")

    # Subsample
    if len(df) > SUBSAMPLE_N:
        df = df.sample(n=SUBSAMPLE_N, random_state=SEED).reset_index(drop=True)
        log.info(f"Subsampled to {len(df)} rows")

    sequences = df["seq"].tolist()
    smiles = df["smiles_can"].tolist()
    labels = df["neg_log10_affinity_M"].values.astype(np.float32)

    # ── 2. Embed proteins → HDF5 ────────────────────────────────────────
    prot_h5 = EMB_DIR / "jglaser_prot_emb.h5"
    if prot_h5.exists():
        log.info(f"Protein embeddings already exist: {prot_h5}")
    else:
        log.info("Embedding proteins with ProtT5-XL …")
        embed_proteins_to_hdf5(sequences, prot_h5)

    # ── 3. Embed ligands → HDF5 ─────────────────────────────────────────
    lig_h5 = EMB_DIR / "jglaser_lig_emb.h5"
    if lig_h5.exists():
        log.info(f"Ligand embeddings already exist: {lig_h5}")
    else:
        log.info("Embedding ligands with ChemBERTa …")
        embed_ligands_to_hdf5(smiles, lig_h5)

    # ── 4. Train/val split ───────────────────────────────────────────────
    n = len(labels)
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n)
    split = int(0.9 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    log.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    train_ds = HDF5AffinityDataset(prot_h5, lig_h5, train_labels, train_idx)
    val_ds = HDF5AffinityDataset(prot_h5, lig_h5, val_labels, val_idx)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH, shuffle=False, num_workers=0)

    # ── 5. Build model ───────────────────────────────────────────────────
    model = AffinityPredictor().to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    # ── 6. Training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_rmse, val_pearson = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        log.info(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_rmse={val_rmse:.4f} | "
            f"val_r={val_pearson:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.1f}s"
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_rmse": float(val_rmse),
            "val_pearson": float(val_pearson),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = CKPT_DIR / "best_affinity_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_pearson": val_pearson,
                "hyperparams": {
                    "PROT_EMB_DIM": PROT_EMB_DIM,
                    "LIG_EMB_DIM": LIG_EMB_DIM,
                    "PROJ_DIM": PROJ_DIM,
                    "N_CROSS_HEADS": N_CROSS_HEADS,
                    "HIDDEN_DIM": HIDDEN_DIM,
                    "DROPOUT": DROPOUT,
                },
            }, ckpt_path)
            log.info(f"  ✓ Best model saved → {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    # ── 7. Save metrics ──────────────────────────────────────────────────
    with open(METRICS_PATH, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Training metrics saved → {METRICS_PATH}")

    # ── 8. Final summary ─────────────────────────────────────────────────
    best_epoch = min(history, key=lambda x: x["val_loss"])
    log.info("=" * 70)
    log.info(f"Best epoch: {best_epoch['epoch']}")
    log.info(f"  val_loss  = {best_epoch['val_loss']:.4f}")
    log.info(f"  val_rmse  = {best_epoch['val_rmse']:.4f}")
    log.info(f"  val_r     = {best_epoch['val_pearson']:.4f}")
    log.info("=" * 70)
    log.info("Script 1 complete.")


if __name__ == "__main__":
    main()
