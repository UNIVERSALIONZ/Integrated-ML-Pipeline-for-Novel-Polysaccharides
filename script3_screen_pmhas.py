#!/usr/bin/env python3
"""
Script 3: Screen and rank all PmHAS SSM mutants at positions 48, 50, 108
against novel substrates (GlcNAc, UDP_Glc, GalNAc).

Fixes vs older versions:
- Loads model dimensions from checkpoint hyperparams instead of hard-coding.
- Verifies pmHAS protein embedding HDF5 shape matches checkpoint expectation.
- Rebuilds stale embeddings automatically.
- Keeps real WT identification and mutant labeling logic.
- Saves ranked tables and heatmaps into pmhas_results/.
"""

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

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────
WORK = Path(__file__).resolve().parent
PMHAS_CSV = WORK / "pmHAS 8000 Mutants Data.csv"
CKPT_PATH = WORK / "checkpoints" / "best_affinity_model.pt"
EMB_DIR = WORK / "embeddings"
EMB_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = WORK / "pmhas_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── encoder names ────────────────────────────────────────────────────────
PROT_MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
LIG_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

# ── defaults; actual dims are loaded from checkpoint ────────────────────
DEFAULT_PROT_EMB_DIM = 1024
DEFAULT_LIG_EMB_DIM = 768
DEFAULT_PROJ_DIM = 256
DEFAULT_N_CROSS_HEADS = 4
DEFAULT_HIDDEN_DIM = 512
DEFAULT_DROPOUT = 0.1

MAX_PROT_LEN = 512
MAX_LIG_LEN = 128
PROT_EMB_BATCH = 1
PRED_BATCH = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

# ── Biological constants ─────────────────────────────────────────────────
MUT_STR_INDICES = [47, 49, 107]
MUT_BIO_POSITIONS = [48, 50, 108]
WT_RESIDUES = {"48": "T", "50": "F", "108": "F"}

SUBSTRATE_SMILES = {
    "GlcNAc": "CC(=O)N[C@@H]1[C@H](O)[C@@H](O)[C@H](O)O[C@@H]1CO",
    "UDP_Glc": "O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)[C@@H](O)[C@H]2O)c(=O)[nH]1",
    "GalNAc": "CC(=O)N[C@@H]1[C@H](O)[C@H](O)[C@H](O)O[C@@H]1CO",
}


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_proteins_to_hdf5(sequences, h5_path, prot_emb_dim, batch_size=PROT_EMB_BATCH):
    from transformers import T5Tokenizer, T5EncoderModel

    log.info("Loading ProtT5-XL tokenizer + encoder ...")
    tokenizer = T5Tokenizer.from_pretrained(PROT_MODEL_NAME, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROT_MODEL_NAME)
    model.gradient_checkpointing_enable()
    model.eval()
    model.to(DEVICE)
    model.half()

    n = len(sequences)
    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset("embeddings", shape=(n, prot_emb_dim), dtype="float32")

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_seqs = sequences[start:end]
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

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                hidden = out.last_hidden_state

            mask_f = attn_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
            pooled = pooled.float().cpu().numpy()

            if pooled.shape[1] != prot_emb_dim:
                raise ValueError(
                    f"Embedded protein dim mismatch: got {pooled.shape[1]}, expected {prot_emb_dim}"
                )

            ds[start:end] = pooled

            if (start // batch_size) % 50 == 0:
                log.info(f" Protein embeddings: {end}/{n}")

            del enc, input_ids, attn_mask, out, hidden, mask_f, pooled
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    log.info(f"Protein embeddings saved -> {h5_path} ({n} vectors)")


def embed_smiles_list(smiles_list):
    from transformers import RobertaTokenizer, RobertaModel

    tokenizer = RobertaTokenizer.from_pretrained(LIG_MODEL_NAME)
    model = RobertaModel.from_pretrained(LIG_MODEL_NAME)
    model.eval().half().to(DEVICE)

    enc = tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=MAX_LIG_LEN,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attn_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
        out = model(input_ids=input_ids, attention_mask=attn_mask)
        pooled = masked_mean_pool(out.last_hidden_state.float(), attn_mask.float())
        result = pooled.cpu().numpy()

    del model, tokenizer, input_ids, attn_mask, out, pooled
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


class GatedCrossAttention(nn.Module):
    def __init__(self, prot_dim, lig_dim, proj_dim, n_heads, dropout=0.1):
        super().__init__()
        self.proj_prot = nn.Linear(prot_dim, proj_dim)
        self.proj_lig = nn.Linear(lig_dim, proj_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate_net = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.Sigmoid(),
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
    def __init__(self, prot_dim, lig_dim, proj_dim, n_heads, hidden_dim, dropout):
        super().__init__()
        self.fusion = GatedCrossAttention(prot_dim, lig_dim, proj_dim, n_heads, dropout)
        self.regressor = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, prot_emb, lig_emb):
        fused = self.fusion(prot_emb, lig_emb)
        return self.regressor(fused).squeeze(-1)


def label_mutation(seq: str) -> str:
    mutations = []
    for str_idx, bio_pos in zip(MUT_STR_INDICES, MUT_BIO_POSITIONS):
        wt_res = WT_RESIDUES[str(bio_pos)]
        mut_res = seq[str_idx]
        if mut_res != wt_res:
            mutations.append(f"{wt_res}{bio_pos}{mut_res}")
    return "/".join(mutations) if mutations else "WT"


def count_mutations(label: str) -> int:
    if label == "WT":
        return 0
    return label.count("/") + 1


def load_model_hparams_from_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    hp = ckpt.get("hyperparams", {})

    prot_emb_dim = int(hp.get("PROT_EMB_DIM", DEFAULT_PROT_EMB_DIM))
    lig_emb_dim = int(hp.get("LIG_EMB_DIM", DEFAULT_LIG_EMB_DIM))
    proj_dim = int(hp.get("PROJ_DIM", DEFAULT_PROJ_DIM))
    n_cross_heads = int(hp.get("N_CROSS_HEADS", DEFAULT_N_CROSS_HEADS))
    hidden_dim = int(hp.get("HIDDEN_DIM", DEFAULT_HIDDEN_DIM))
    dropout = float(hp.get("DROPOUT", DEFAULT_DROPOUT))

    return ckpt, {
        "PROT_EMB_DIM": prot_emb_dim,
        "LIG_EMB_DIM": lig_emb_dim,
        "PROJ_DIM": proj_dim,
        "N_CROSS_HEADS": n_cross_heads,
        "HIDDEN_DIM": hidden_dim,
        "DROPOUT": dropout,
    }


def ensure_pmhas_embeddings(sequences, prot_h5, expected_dim):
    n_mutants = len(sequences)
    rebuild = False

    if prot_h5.exists():
        with h5py.File(prot_h5, "r") as f:
            shape = f["embeddings"].shape
            existing_n, existing_dim = shape[0], shape[1]

        if existing_n != n_mutants or existing_dim != expected_dim:
            log.info(
                f"Stale pmHAS embeddings found: shape={shape}, expected=({n_mutants}, {expected_dim}). Rebuilding ..."
            )
            rebuild = True
        else:
            log.info(f"PmHAS protein embeddings exist: {prot_h5} {shape}")
    else:
        rebuild = True

    if rebuild:
        if prot_h5.exists():
            prot_h5.unlink()
        log.info(f"Embedding {n_mutants} PmHAS mutant sequences with ProtT5-XL ...")
        embed_proteins_to_hdf5(sequences, prot_h5, prot_emb_dim=expected_dim, batch_size=PROT_EMB_BATCH)


def main():
    log.info("=" * 70)
    log.info("SCRIPT 3: Screen PmHAS SSM mutants against novel substrates")
    log.info("=" * 70)
    log.info("Real WT residues: T48, F50, F108")
    log.info("(The 'A' in PyMOL naming refers to Chain A, not Alanine)")

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
    log.info(f"Raw shape: {df.shape}, Columns: {list(df.columns)}")

    df = df.dropna(subset=["Receptor", "GlcNAc", "UDP Glc", "GalNAc"])
    df = df[df["Receptor"].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    df = df.reset_index(drop=True)
    log.info(f"After cleaning: {len(df)} rows")

    sequences = df["Receptor"].tolist()
    n_mutants = len(sequences)

    wt_mask = df["Receptor"].apply(lambda s: s[47] == "T" and s[49] == "F" and s[107] == "F")
    wt_rows = df[wt_mask]
    if len(wt_rows) == 0:
        log.error("FATAL: Could not find real WT (T48/F50/F108) in the dataset!")
        sys.exit(1)

    wt_row_idx = wt_rows.index[0]
    wt_seq = sequences[wt_row_idx]
    log.info(f"Real WT found at row index {wt_row_idx}")
    log.info(f"  WT sequence length: {len(wt_seq)}")
    log.info(f"  WT pos48={wt_seq[47]}, pos50={wt_seq[49]}, pos108={wt_seq[107]}")
    log.info(
        f"  WT docking scores: GlcNAc={df.loc[wt_row_idx, 'GlcNAc']:.1f}, "
        f"UDP_Glc={df.loc[wt_row_idx, 'UDP Glc']:.1f}, "
        f"GalNAc={df.loc[wt_row_idx, 'GalNAc']:.1f}"
    )

    log.info("Labeling mutations relative to real WT (T48/F50/F108) ...")
    mutation_labels, pos48_residues, pos50_residues, pos108_residues = [], [], [], []

    for seq in sequences:
        mutation_labels.append(label_mutation(seq))
        pos48_residues.append(seq[47])
        pos50_residues.append(seq[49])
        pos108_residues.append(seq[107])

    df["mutation"] = mutation_labels
    df["pos48"] = pos48_residues
    df["pos50"] = pos50_residues
    df["pos108"] = pos108_residues
    df["n_mutations"] = df["mutation"].apply(count_mutations)

    counts = df["n_mutations"].value_counts().sort_index()
    for n_mut, cnt in counts.items():
        label = {0: "WT", 1: "Single", 2: "Double", 3: "Triple"}.get(n_mut, f"{n_mut}-mut")
        log.info(f"  {label}: {cnt}")

    log.info(f"  Row 0 label: {mutation_labels[0]} (should be T48A/F50A/F108A)")
    log.info(f"  Row {wt_row_idx} label: {mutation_labels[wt_row_idx]} (should be WT)")

    exp_scores = {
        "GlcNAc": df["GlcNAc"].values.astype(np.float32),
        "UDP_Glc": df["UDP Glc"].values.astype(np.float32),
        "GalNAc": df["GalNAc"].values.astype(np.float32),
    }

    prot_h5 = EMB_DIR / "pmhas_prot_emb.h5"
    ensure_pmhas_embeddings(sequences, prot_h5, PROT_EMB_DIM)

    substrate_names = list(SUBSTRATE_SMILES.keys())
    substrate_smiles = [SUBSTRATE_SMILES[s] for s in substrate_names]
    log.info(f"Embedding {len(substrate_names)} substrate SMILES with ChemBERTa ...")
    for name, smi in zip(substrate_names, substrate_smiles):
        log.info(f"  {name}: {smi[:60]}...")
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

    log.info(f"Predicting binding affinities for {n_mutants} mutants x {len(substrate_names)} substrates ...")

    prot_h5_file = h5py.File(prot_h5, "r")
    prot_emb_ds = prot_h5_file["embeddings"]

    predictions = {}
    for sub_idx, sub_name in enumerate(substrate_names):
        log.info(f" Screening against {sub_name} ...")
        sub_emb = torch.tensor(substrate_embs[sub_idx], dtype=torch.float32).to(DEVICE)

        preds = []
        for start in range(0, n_mutants, PRED_BATCH):
            end = min(start + PRED_BATCH, n_mutants)
            prot_batch = torch.tensor(prot_emb_ds[start:end], dtype=torch.float32).to(DEVICE)
            lig_batch = sub_emb.unsqueeze(0).expand(end - start, -1)

            with torch.no_grad():
                pred = model(prot_batch, lig_batch)
                preds.append(pred.cpu().numpy())

            del prot_batch, lig_batch, pred
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        predictions[sub_name] = np.concatenate(preds)
        log.info(
            f"  range: [{predictions[sub_name].min():.3f}, {predictions[sub_name].max():.3f}], "
            f"mean: {predictions[sub_name].mean():.3f}"
        )

    prot_h5_file.close()

    log.info("Computing delta_pKd relative to REAL WT (T48/F50/F108) ...")
    wt_preds = {sub: float(predictions[sub][wt_row_idx]) for sub in substrate_names}
    for sub in substrate_names:
        log.info(f"  WT pred for {sub}: {wt_preds[sub]:.4f}")

    results = pd.DataFrame({
        "mutation": mutation_labels,
        "n_mutations": df["n_mutations"],
        "pos48": pos48_residues,
        "pos50": pos50_residues,
        "pos108": pos108_residues,
    })

    for sub in substrate_names:
        results[f"pred_pKd_{sub}"] = predictions[sub]
        results[f"delta_pKd_{sub}"] = predictions[sub] - wt_preds[sub]
        results[f"exp_score_{sub}"] = exp_scores[sub]

    delta_cols = [f"delta_pKd_{sub}" for sub in substrate_names]
    results["avg_delta_pKd"] = results[delta_cols].mean(axis=1)
    results["n_substrates_improved"] = sum(
        (results[f"delta_pKd_{sub}"] > 0).astype(int) for sub in substrate_names
    )

    log.info("=" * 70)
    log.info("RANKING RESULTS (relative to real WT: T48/F50/F108)")
    log.info("=" * 70)

    results_sorted = results.sort_values("avg_delta_pKd", ascending=False)
    top_mutants = results_sorted[results_sorted["mutation"] != "WT"]

    log.info("\n--- TOP 20 MUTANTS (by avg delta_pKd vs real WT) ---")
    for _, row in top_mutants.head(20).iterrows():
        log.info(
            f" {row['mutation']:30s} | "
            f"avg_delta={row['avg_delta_pKd']:+.4f} | "
            f"GlcNAc={row['delta_pKd_GlcNAc']:+.4f} | "
            f"UDP_Glc={row['delta_pKd_UDP_Glc']:+.4f} | "
            f"GalNAc={row['delta_pKd_GalNAc']:+.4f} | "
            f"n_improved={row['n_substrates_improved']}"
        )

    for sub in substrate_names:
        delta_col = f"delta_pKd_{sub}"
        sub_sorted = results[results["mutation"] != "WT"].sort_values(delta_col, ascending=False)
        log.info(f"\n--- TOP 10 MUTANTS for {sub} ---")
        for _, row in sub_sorted.head(10).iterrows():
            log.info(
                f" {row['mutation']:30s} | "
                f"delta={row[delta_col]:+.4f} | "
                f"pred_pKd={row[f'pred_pKd_{sub}']:.4f} | "
                f"exp_score={row[f'exp_score_{sub}']:.2f}"
            )

    multi_improved = results[
        (results["n_substrates_improved"] == 3) & (results["mutation"] != "WT")
    ].sort_values("avg_delta_pKd", ascending=False)

    log.info(f"\n--- MUTANTS IMPROVED IN ALL 3 SUBSTRATES: {len(multi_improved)} ---")
    if len(multi_improved) > 0:
        for _, row in multi_improved.head(20).iterrows():
            log.info(
                f" {row['mutation']:30s} | "
                f"avg_delta={row['avg_delta_pKd']:+.4f} | "
                f"GlcNAc={row['delta_pKd_GlcNAc']:+.4f} | "
                f"UDP_Glc={row['delta_pKd_UDP_Glc']:+.4f} | "
                f"GalNAc={row['delta_pKd_GalNAc']:+.4f}"
            )
    else:
        log.info(" (none found)")

    single_mutants = results[results["n_mutations"] == 1].sort_values("avg_delta_pKd", ascending=False)
    log.info(f"\n--- TOP 10 SINGLE MUTANTS (easiest to synthesize) ---")
    for _, row in single_mutants.head(10).iterrows():
        log.info(
            f" {row['mutation']:15s} | "
            f"avg_delta={row['avg_delta_pKd']:+.4f} | "
            f"GlcNAc={row['delta_pKd_GlcNAc']:+.4f} | "
            f"UDP_Glc={row['delta_pKd_UDP_Glc']:+.4f} | "
            f"GalNAc={row['delta_pKd_GalNAc']:+.4f}"
        )

    full_csv = OUTPUT_DIR / "pmhas_full_screening_results.csv"
    top_csv = OUTPUT_DIR / "pmhas_top_mutants.csv"
    single_csv = OUTPUT_DIR / "pmhas_single_mutants_ranked.csv"

    results_sorted.to_csv(full_csv, index=False)
    top_mutants.head(50).to_csv(top_csv, index=False)
    single_mutants.to_csv(single_csv, index=False)

    log.info(f"\nFull results saved -> {full_csv}")
    log.info(f"Top 50 mutants saved -> {top_csv}")
    log.info(f"All single mutants ranked -> {single_csv}")

    for sub in substrate_names:
        sub_csv = OUTPUT_DIR / f"pmhas_top_{sub}.csv"
        sub_sorted = results[results["mutation"] != "WT"].sort_values(
            f"delta_pKd_{sub}", ascending=False
        ).head(50)
        sub_sorted.to_csv(sub_csv, index=False)
        log.info(f"Top 50 for {sub} saved -> {sub_csv}")

    log.info("\n" + "=" * 70)
    log.info("SCREENING SUMMARY")
    log.info("=" * 70)
    log.info(f"Real WT: T48/F50/F108 (row {wt_row_idx})")
    log.info(f"Total mutants screened: {n_mutants} ({n_mutants - 1} non-WT)")
    log.info(f"Substrates: {substrate_names}")

    for sub in substrate_names:
        n_better = ((results[f"delta_pKd_{sub}"] > 0) & (results["mutation"] != "WT")).sum()
        n_worse = ((results[f"delta_pKd_{sub}"] < 0) & (results["mutation"] != "WT")).sum()
        best_row = results[results["mutation"] != "WT"].loc[
            results[results["mutation"] != "WT"][f"delta_pKd_{sub}"].idxmax()
        ]
        log.info(
            f" {sub}: {n_better} improved, {n_worse} worsened, "
            f"best={best_row['mutation']} (delta={best_row[f'delta_pKd_{sub}']:+.4f})"
        )

    log.info(f"Multi-substrate improvers (all 3): {len(multi_improved)}")
    log.info(f"Single mutants: {len(single_mutants)}")
    log.info(f"WT predictions: { {k: f'{v:.4f}' for k, v in wt_preds.items()} }")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        aa_order = "ACDEFGHIKLMNPQRSTVWY"

        for sub in substrate_names:
            delta_col = f"delta_pKd_{sub}"

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(
                f"PmHAS SSM Delta pKd vs Real WT (T48/F50/F108) - {sub}",
                fontsize=14, y=1.02
            )

            for ax_idx, (pos_col, bio_pos) in enumerate([("pos48", 48), ("pos50", 50), ("pos108", 108)]):
                wt_res = WT_RESIDUES[str(bio_pos)]
                pivot = results.groupby(pos_col)[delta_col].mean()
                vals = [pivot.get(aa, 0.0) for aa in aa_order]

                ax = axes[ax_idx]
                colors = []
                for aa, v in zip(aa_order, vals):
                    if aa == wt_res:
                        colors.append("#4393c3")
                    elif v < 0:
                        colors.append("#d73027")
                    else:
                        colors.append("#1a9850")

                ax.barh(range(len(aa_order)), vals, color=colors, edgecolor="white", height=0.8)
                ax.set_yticks(range(len(aa_order)))
                ax.set_yticklabels(list(aa_order), fontsize=9)
                ax.set_xlabel("Mean delta pKd vs WT", fontsize=10)
                ax.set_title(f"Position {bio_pos} (WT: {wt_res})", fontsize=12)
                ax.axvline(0, color="black", linewidth=0.5)
                ax.invert_yaxis()

                wt_idx = aa_order.index(wt_res)
                ax.annotate(
                    "WT", xy=(vals[wt_idx], wt_idx),
                    fontsize=8, fontweight="bold", color="#4393c3",
                    ha="left" if vals[wt_idx] >= 0 else "right"
                )

            plt.tight_layout()
            plot_path = OUTPUT_DIR / f"pmhas_heatmap_{sub}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info(f"Heatmap saved -> {plot_path}")

        fig, ax = plt.subplots(figsize=(12, 8))
        data_matrix = np.zeros((20, 9))
        col_labels = []
        col_idx = 0

        for sub in substrate_names:
            for pos_col, bio_pos in [("pos48", 48), ("pos50", 50), ("pos108", 108)]:
                wt_res = WT_RESIDUES[str(bio_pos)]
                pivot = results.groupby(pos_col)[f"delta_pKd_{sub}"].mean()
                for aa_idx, aa in enumerate(aa_order):
                    data_matrix[aa_idx, col_idx] = pivot.get(aa, 0.0)
                col_labels.append(f"{sub}\nP{bio_pos} (WT:{wt_res})")
                col_idx += 1

        vmax = max(abs(data_matrix.min()), abs(data_matrix.max()), 0.01)
        im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(9))
        ax.set_xticklabels(col_labels, fontsize=8, rotation=0, ha="center")
        ax.set_yticks(range(20))
        ax.set_yticklabels(list(aa_order), fontsize=9)
        ax.set_ylabel("Amino Acid", fontsize=11)
        ax.set_title(
            "PmHAS SSM: Mean Delta pKd per Residue/Position/Substrate\n"
            "(relative to real WT: T48/F50/F108)",
            fontsize=12
        )
        plt.colorbar(im, ax=ax, label="Delta pKd vs WT", shrink=0.8)

        for col_i, (_, bio_pos) in enumerate([(s, p) for s in substrate_names for p in [48, 50, 108]]):
            wt_res = WT_RESIDUES[str(bio_pos)]
            wt_row = aa_order.index(wt_res)
            ax.plot(col_i, wt_row, marker="*", color="black", markersize=10)

        plt.tight_layout()
        combined_plot = OUTPUT_DIR / "pmhas_combined_heatmap.png"
        plt.savefig(combined_plot, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Combined heatmap saved -> {combined_plot}")

    except ImportError:
        log.warning("matplotlib not available; skipping heatmaps.")

    metrics = {
        "wt_row_idx": int(wt_row_idx),
        "n_mutants": int(n_mutants),
        "checkpoint_hyperparams": hp,
        "wt_predictions": {k: float(v) for k, v in wt_preds.items()},
        "n_multi_improved": int(len(multi_improved)),
        "n_single_mutants": int(len(single_mutants)),
    }
    with open(OUTPUT_DIR / "pmhas_screening_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("=" * 70)
    log.info("Script 3 complete.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()