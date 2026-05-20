# pip install torch
"""
Train entity embedding network and build enriched feature matrices for run_003.

Replaces one-hot / target encoding with learned embeddings, class-conditional geo
rates, and interaction features from error analysis.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from preprocess import AGE_SENTINEL, DATA_DIR, RANDOM_STATE, TEST_SIZE, binary_columns
from run_manager import PREPROCESSING_DIR, PROCESSED_DIR, ROOT, RunManager

print = partial(print, flush=True)

EMBED_DIR = ROOT / "models" / "embedding"
EVAL_DIR = ROOT / "runs" / "evaluation"

# (column_name, embed_dim) — num_embeddings = n_train_classes + 1 (index 0 = unknown)
EMBED_COLS: dict[str, int] = {
    "geo_level_1_id": 16,
    "geo_level_2_id": 50,
    "geo_level_3_id": 50,
    "foundation_type": 3,
    "roof_type": 2,
    "land_surface_condition": 2,
    "ground_floor_type": 3,
    "other_floor_type": 3,
    "position": 3,
    "plan_configuration": 5,
    "legal_ownership_status": 3,
}

EMBED_PREFIX = {
    "geo_level_1_id": "geo1",
    "geo_level_2_id": "geo2",
    "geo_level_3_id": "geo3",
    "foundation_type": "foundation",
    "roof_type": "roof",
    "land_surface_condition": "land",
    "ground_floor_type": "ground",
    "other_floor_type": "other_floor",
    "position": "position",
    "plan_configuration": "plan",
    "legal_ownership_status": "ownership",
}

GEO_RATE_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
NUMERIC_COLS = [
    "count_floors_pre_eq",
    "age",
    "area_percentage",
    "height_percentage",
    "count_families",
]
GRADES = [1, 2, 3]

BATCH_SIZE = 4096
EPOCHS = 50
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Skip embedding retrain — reuse epoch-37 weights; only rebuild geo/interaction features
SKIP_TRAINING = True
RETRAIN_RUN_ID = "run_004"


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on: {device}")
    return device


# ── Part 1: Data preparation ─────────────────────────────────────────────


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load merged train and raw test; keep building_id on train for split alignment."""
    try:
        values = pd.read_csv(DATA_DIR / "train_values.csv")
        labels = pd.read_csv(DATA_DIR / "train_labels.csv")
        test = pd.read_csv(DATA_DIR / "test_values.csv")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing driven_data CSVs. Check data/driven_data/train_values.csv"
        ) from exc
    train = values.merge(labels, on="building_id", how="inner", validate="one_to_one")
    train["survived"] = (train["damage_grade"] <= 2).astype(int)
    return train, test


def apply_age_sentinel(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """995 = unknown; impute with train median (non-995)."""
    train = train.copy()
    test = test.copy()
    train["age_unknown"] = (train["age"] == AGE_SENTINEL).astype(int)
    test["age_unknown"] = (test["age"] == AGE_SENTINEL).astype(int)
    median_age = train.loc[train["age"] != AGE_SENTINEL, "age"].median()
    train["age"] = train["age"].replace(AGE_SENTINEL, np.nan).fillna(median_age)
    test["age"] = test["age"].replace(AGE_SENTINEL, np.nan).fillna(median_age)
    return train, test


def lightgbm_train_split(train: pd.DataFrame) -> pd.DataFrame:
    """Same 80/20 stratified split as preprocess.py (208,480 rows)."""
    tr, _ = train_test_split(
        train,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=train["survived"],
    )
    return tr.reset_index(drop=True)


def fit_label_encoders(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, LabelEncoder], dict[str, int]]:
    """Fit encoders on train; index 0 = unknown, train classes 1..K."""
    enc_train: dict[str, np.ndarray] = {}
    enc_test: dict[str, np.ndarray] = {}
    encoders: dict[str, LabelEncoder] = {}
    n_emb: dict[str, int] = {}

    for col in EMBED_COLS:
        le = LabelEncoder()
        le.fit(train[col].astype(str))
        encoders[col] = le
        tr = le.transform(train[col].astype(str).values) + 1
        te = np.zeros(len(test), dtype=np.int64)
        test_str = test[col].astype(str).values
        known = np.isin(test_str, le.classes_)
        if known.any():
            te[known] = le.transform(test_str[known]) + 1
        enc_train[col] = tr
        enc_test[col] = te
        n_emb[col] = len(le.classes_) + 1

    return enc_train, enc_test, encoders, n_emb


def transform_label_encoders(
    train: pd.DataFrame,
    test: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    """Apply saved encoders; index 0 = unknown."""
    enc_train: dict[str, np.ndarray] = {}
    enc_test: dict[str, np.ndarray] = {}
    n_emb: dict[str, int] = {}

    for col in EMBED_COLS:
        le = encoders[col]
        tr = le.transform(train[col].astype(str).values) + 1
        te = np.zeros(len(test), dtype=np.int64)
        test_str = test[col].astype(str).values
        known = np.isin(test_str, le.classes_)
        if known.any():
            te[known] = le.transform(test_str[known]) + 1
        enc_train[col] = tr
        enc_test[col] = te
        n_emb[col] = len(le.classes_) + 1

    return enc_train, enc_test, n_emb


def scale_numerics(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Apply pre-fitted scaler from preprocess (train-only fit)."""
    scaler_path = PREPROCESSING_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing {scaler_path}. Run src/preprocess.py first.")
    scaler: StandardScaler = joblib.load(scaler_path)
    train_num = scaler.transform(train[NUMERIC_COLS])
    test_num = scaler.transform(test[NUMERIC_COLS])
    return train_num, test_num, scaler


# ── Part 2: Embedding network ────────────────────────────────────────────


class NepalEmbeddingNet(nn.Module):
    """Entity embeddings + MLP for 3-class damage prediction."""

    def __init__(
        self,
        n_emb_per_col: dict[str, int],
        embed_dims: dict[str, int],
        n_numeric: int,
        n_binary: int,
    ) -> None:
        super().__init__()
        self.embed_cols = list(EMBED_COLS.keys())
        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(n_emb_per_col[col], embed_dims[col])
                for col in self.embed_cols
            }
        )
        input_dim = sum(embed_dims.values()) + n_numeric + n_binary
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        cat_inputs: dict[str, torch.Tensor],
        num_tensor: torch.Tensor,
        bin_tensor: torch.Tensor,
    ) -> torch.Tensor:
        embeds = [self.embeddings[col](cat_inputs[col]) for col in self.embed_cols]
        x = torch.cat(embeds + [num_tensor, bin_tensor], dim=1)
        return self.mlp(x)


def train_embedding_network(
    enc_train: dict[str, np.ndarray],
    train_num: np.ndarray,
    train_bin: np.ndarray,
    y: np.ndarray,
    n_emb: dict[str, int],
    device: torch.device,
) -> tuple[NepalEmbeddingNet, dict]:
    """Train with 80/20 monitor split; early stopping on val micro F1."""
    y_idx = y - 1
    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y_idx
    )

    def make_tensors(idxs: np.ndarray) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        cats = {
            col: torch.tensor(enc_train[col][idxs], dtype=torch.long, device=device)
            for col in EMBED_COLS
        }
        nums = torch.tensor(train_num[idxs], dtype=torch.float32, device=device)
        bins = torch.tensor(train_bin[idxs], dtype=torch.float32, device=device)
        labels = torch.tensor(y_idx[idxs], dtype=torch.long, device=device)
        return cats, nums, bins, labels

    counts = np.bincount(y_idx, minlength=3)
    weights = len(y_idx) / (3.0 * counts.astype(float))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    model = NepalEmbeddingNet(
        n_emb, EMBED_COLS, train_num.shape[1], train_bin.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(tr_idx) // BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = np.random.permutation(tr_idx)
        train_loss = 0.0
        n_batches = 0
        for start in range(0, len(perm), BATCH_SIZE):
            batch = perm[start : start + BATCH_SIZE]
            cats, nums, bins, labels = make_tensors(batch)
            optimizer.zero_grad()
            logits = model(cats, nums, bins)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            cats, nums, bins, labels = make_tensors(va_idx)
            val_logits = model(cats, nums, bins)
            val_loss = criterion(val_logits, labels).item()
            preds = np.array(val_logits.argmax(dim=1).detach().cpu().tolist()) + 1
            val_f1 = f1_score(y[va_idx], preds, average="micro", labels=GRADES)

        print(
            f"  epoch {epoch:2d}  train_loss={train_loss / n_batches:.4f}  "
            f"val_loss={val_loss:.4f}  val_micro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            EMBED_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), EMBED_DIR / "embedding_network.pt")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "best_val_micro_f1": best_f1}
    print(f"Best epoch: {best_epoch}, best val micro F1: {best_f1:.4f}")
    return model, info


# ── Part 3: Extract embeddings ───────────────────────────────────────────


def extract_embedding_features(
    model: NepalEmbeddingNet,
    enc_train: dict[str, np.ndarray],
    enc_test: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lookup learned embedding vectors per building."""
    weights = {
        col: np.array(model.embeddings[col].weight.detach().cpu().tolist())
        for col in EMBED_COLS
    }
    joblib.dump(weights, EMBED_DIR / "embedding_weights.pkl")

    def lookup(enc: dict[str, np.ndarray]) -> pd.DataFrame:
        parts = []
        for col, dim in EMBED_COLS.items():
            prefix = EMBED_PREFIX[col]
            w = weights[col]
            idxs = enc[col]
            vecs = w[idxs]
            cols = [f"{prefix}_emb_{i}" for i in range(dim)]
            parts.append(pd.DataFrame(vecs, columns=cols))
        return pd.concat(parts, axis=1)

    return lookup(enc_train), lookup(enc_test)


# ── Part 4: Class-conditional geo rates (Laplace-smoothed + CV) ─────────
# Raw area rates overfit small geos (run_003 CV/public gap). Smooth toward global
# priors and assign train rows via 5-fold CV so LightGBM never sees own-area rates.

GEO_RATE_ALPHA = 10
GEO_RATE_CV_FOLDS = 5


def global_grade_rates(ref: pd.DataFrame) -> dict[int, float]:
    """Training-set marginal P(grade) — used as Laplace prior and unseen-geo fallback."""
    n = len(ref)
    return {g: (ref["damage_grade"] == g).sum() / n for g in GRADES}


def smoothed_grade_rates_by_geo(
    ref: pd.DataFrame,
    geo_col: str,
    global_rates: dict[int, float],
) -> dict[int, pd.Series]:
    """Per grade: geo_id -> Laplace-smoothed P(grade | geo) from ref only."""
    totals = ref.groupby(geo_col).size()
    rates: dict[int, pd.Series] = {}
    for g in GRADES:
        cnt = ref.loc[ref["damage_grade"] == g].groupby(geo_col).size()
        rates[g] = (cnt + GEO_RATE_ALPHA * global_rates[g]) / (totals + GEO_RATE_ALPHA)
    return rates


def assign_geo_rates(
    df: pd.DataFrame,
    rate_maps: dict[str, dict[int, pd.Series]],
    global_rates: dict[int, float],
) -> pd.DataFrame:
    """Map smoothed rate tables onto rows; unseen geo IDs get global rates."""
    out = pd.DataFrame(index=df.index)
    for geo_col in GEO_RATE_COLS:
        prefix = EMBED_PREFIX[geo_col]
        maps = rate_maps[geo_col]
        for g in GRADES:
            out[f"{prefix}_p_grade{g}"] = df[geo_col].map(maps[g]).fillna(global_rates[g]).values
    return out


def print_smoothing_diagnostic(train: pd.DataFrame, global_rates: dict[int, float]) -> None:
    """Show raw vs smoothed grade-3 rate for the 5 smallest geo_level_3 areas."""
    geo_col = "geo_level_3_id"
    print("\nSmoothing effect on smallest geo areas:")
    print("    geo_id | n_buildings | raw_p3 | smoothed_p3")
    print("    -------+-------------+--------+------------")
    totals = train.groupby(geo_col).size()
    cnt3 = train.loc[train["damage_grade"] == 3].groupby(geo_col).size()
    g3 = global_rates[3]
    for geo_id, n in totals.sort_values().head(5).items():
        c = cnt3.get(geo_id, 0)
        raw = c / n
        smooth = (c + GEO_RATE_ALPHA * g3) / (n + GEO_RATE_ALPHA)
        print(f"    {geo_id:6} | {n:11} | {raw:6.4f} | {smooth:6.4f}")


def compute_cv_geo_rates(train: pd.DataFrame, global_rates: dict[int, float]) -> pd.DataFrame:
    """5-fold CV smoothed rates: each row uses only other folds' buildings."""
    geo_cols = [
        f"{EMBED_PREFIX[g]}_p_grade{grade}"
        for g in GEO_RATE_COLS
        for grade in GRADES
    ]
    out = pd.DataFrame(index=train.index, columns=geo_cols, dtype=float)
    for col in geo_cols:
        g = int(col.rsplit("grade", maxsplit=1)[-1])
        out[col] = global_rates[g]

    skf = StratifiedKFold(
        n_splits=GEO_RATE_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    for _fold, (tr_idx, val_idx) in enumerate(skf.split(train, train["damage_grade"])):
        tr_fold = train.iloc[tr_idx]
        val_df = train.iloc[val_idx]
        for geo_col in GEO_RATE_COLS:
            maps = smoothed_grade_rates_by_geo(tr_fold, geo_col, global_rates)
            prefix = EMBED_PREFIX[geo_col]
            for g in GRADES:
                col = f"{prefix}_p_grade{g}"
                out.loc[val_idx, col] = (
                    val_df[geo_col].map(maps[g]).fillna(global_rates[g]).values
                )

    print("Cross-validated geo rates computed — no leakage.")
    return out


def compute_geo_class_rates(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train: CV Laplace-smoothed rates. Test: full-train smoothed rates."""
    print(f"GEO_RATE_ALPHA = {GEO_RATE_ALPHA}")
    global_rates = global_grade_rates(train)
    print(
        f"Global rates: grade1={global_rates[1]:.4f}  "
        f"grade2={global_rates[2]:.4f}  grade3={global_rates[3]:.4f}"
    )
    print_smoothing_diagnostic(train, global_rates)

    geo_train = compute_cv_geo_rates(train, global_rates)
    rate_maps = {
        geo_col: smoothed_grade_rates_by_geo(train, geo_col, global_rates)
        for geo_col in GEO_RATE_COLS
    }
    geo_test = assign_geo_rates(test, rate_maps, global_rates)
    return geo_train, geo_test


# ── Part 5: Interaction features ─────────────────────────────────────────


def add_interaction_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    geo_rates_train: pd.DataFrame,
    geo_rates_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interactions use smoothed CV geo3_p_grade3 (not raw memorized rates)."""

    def build(df: pd.DataFrame, geo_rates: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        g3 = geo_rates["geo3_p_grade3"].values  # Laplace-smoothed; CV on train

        for letter in ["w", "i", "u", "r"]:
            flag = (df["foundation_type"].astype(str) == letter).astype(int).values
            out[f"geo3_collapse_x_foundation_{letter}"] = g3 * flag

        out["age_x_height"] = df["age"].values * df["height_percentage"].values
        out["floors_x_height"] = df["count_floors_pre_eq"].values * df["height_percentage"].values
        out["area_x_floors"] = df["area_percentage"].values * df["count_floors_pre_eq"].values

        super_cols = [c for c in df.columns if c.startswith("has_superstructure_")]
        out["n_superstructure_types"] = df[super_cols].sum(axis=1).values
        has_rc = (
            (df["has_superstructure_rc_engineered"] == 1)
            | (df["has_superstructure_rc_non_engineered"] == 1)
        ).astype(int).values
        has_weak = (
            (df["has_superstructure_adobe_mud"] == 1)
            | (df["has_superstructure_mud_mortar_stone"] == 1)
            | (df["has_superstructure_bamboo"] == 1)
        ).astype(int).values
        has_strong = (
            (df["has_superstructure_rc_engineered"] == 1)
            | (df["has_superstructure_cement_mortar_brick"] == 1)
            | (df["has_superstructure_cement_mortar_stone"] == 1)
        ).astype(int).values
        out["has_rc"] = has_rc
        out["has_weak_material"] = has_weak
        out["has_strong_material"] = has_strong
        out["material_strength_score"] = has_strong - has_weak
        out["geo3_collapse_x_weak_material"] = g3 * has_weak
        out["geo3_collapse_x_rc"] = g3 * has_rc
        return out

    return build(train, geo_rates_train), build(test, geo_rates_test)


# ── Part 6: Assemble final matrix ────────────────────────────────────────


def superstructure_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("has_superstructure_"))


def secondary_use_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("has_secondary_use"))


def assemble_features(
    emb_train: pd.DataFrame,
    emb_test: pd.DataFrame,
    geo_train: pd.DataFrame,
    geo_test: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_num: np.ndarray,
    test_num: np.ndarray,
    interact_train: pd.DataFrame,
    interact_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate all feature blocks in spec order."""
    super_cols = superstructure_columns(train)
    sec_cols = secondary_use_columns(train)

    def numeric_df(num_arr: np.ndarray, age_unknown: np.ndarray) -> pd.DataFrame:
        cols = NUMERIC_COLS + ["age_unknown"]
        data = np.column_stack([num_arr, age_unknown])
        return pd.DataFrame(data, columns=cols)

    num_tr = numeric_df(train_num, train["age_unknown"].values)
    num_te = numeric_df(test_num, test["age_unknown"].values)

    X_tr = pd.concat(
        [
            emb_train.reset_index(drop=True),
            geo_train.reset_index(drop=True),
            num_tr,
            train[super_cols].reset_index(drop=True),
            train[sec_cols].reset_index(drop=True),
            interact_train.reset_index(drop=True),
        ],
        axis=1,
    )
    X_te = pd.concat(
        [
            emb_test.reset_index(drop=True),
            geo_test.reset_index(drop=True),
            num_te,
            test[super_cols].reset_index(drop=True),
            test[sec_cols].reset_index(drop=True),
            interact_test.reset_index(drop=True),
        ],
        axis=1,
    )
    return X_tr, X_te


# ── Part 7: Diagnostics ──────────────────────────────────────────────────


def plot_embedding_diagnostics(
    model: NepalEmbeddingNet,
    train: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
) -> None:
    """PCA of geo3 embeddings colored by mean damage grade."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    w3 = np.array(model.embeddings["geo_level_3_id"].weight.detach().cpu().tolist())[1:]
    geo_ids = encoders["geo_level_3_id"].classes_
    mean_grade = train.groupby(train["geo_level_3_id"].astype(str))["damage_grade"].mean()
    colors = [mean_grade.get(str(g), 2.0) for g in geo_ids]

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    xy = pca.fit_transform(w3)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=colors, cmap="RdYlGn_r", s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="mean damage grade")
    ax.set_title("geo_level_3_id embeddings (PCA)")
    fig.savefig(EVAL_DIR / "embedding_diagnostics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    wf = np.array(model.embeddings["foundation_type"].weight.detach().cpu().tolist())[1:]
    print("\nFoundation type embedding coordinates (3-d):")
    for i, label in enumerate(encoders["foundation_type"].classes_):
        print(f"  {label}: {wf[i]}")


# ── Part 8: RunManager ───────────────────────────────────────────────────


def register_run_004(n_features: int) -> None:
    """Register feature-extraction run for Laplace-smoothed CV geo rates."""
    rm = RunManager()
    run_id = RETRAIN_RUN_ID
    desc = "Embeddings + Laplace-smoothed CV geo rates — fix overfit gap"
    if rm.run_path(run_id).exists():
        print(f"{run_id} already exists — skipping RunManager create.")
        return
    rm.create_run(
        description=desc,
        model_type="EmbeddingNet_feature_extractor",
        feature_set="embedded_192_features",
        params={
            "source": "embed.py",
            "lgbm_params": "trial_66",
            "geo_rate_alpha": GEO_RATE_ALPHA,
            "geo_rate_cv_folds": GEO_RATE_CV_FOLDS,
        },
        run_id=run_id,
        objective="multiclass",
        n_features=n_features,
        notes="Smoothed + CV geo rates; embeddings frozen from run_003",
    )
    meta = rm.load_metadata(run_id)
    meta["n_features_baseline"] = 61
    meta["n_features_embedded"] = n_features
    meta["embedding_dims"] = EMBED_COLS
    meta["geo_rate_alpha"] = GEO_RATE_ALPHA
    RunManager._write_json(rm.run_path(run_id) / "metadata.json", meta)


def register_run_003(n_features: int, train_info: dict) -> None:
    rm = RunManager()
    run_id = "run_003"
    if rm.run_path(run_id).exists():
        meta = rm.load_metadata(run_id)
        meta["embedding_epochs_trained"] = train_info.get("best_epoch")
        meta["best_val_f1"] = train_info.get("best_val_micro_f1")
        meta["n_features_embedded"] = n_features
        RunManager._write_json(rm.run_path(run_id) / "metadata.json", meta)
        print(f"Updated {run_id} metadata with embedding training stats.")
        return
    rm.create_run(
        description="Entity embeddings + class-conditional geo + interactions",
        model_type="EmbeddingNet_feature_extractor",
        feature_set="embedded_192_features",
        params={"source": "embed.py", "lgbm_params": "trial_66"},
        run_id="run_003",
        objective="multiclass",
        n_features=n_features,
        notes="Features for LightGBM run_003",
    )
    meta = rm.load_metadata("run_003")
    meta["embedding_epochs_trained"] = train_info.get("best_epoch")
    meta["best_val_f1"] = train_info.get("best_val_micro_f1")
    meta["n_features_baseline"] = 61
    meta["n_features_embedded"] = n_features
    meta["embedding_dims"] = EMBED_COLS
    RunManager._write_json(rm.run_path("run_003") / "metadata.json", meta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train embeddings and build feature matrices")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Load saved embedding_network.pt instead of retraining",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skip_train = SKIP_TRAINING or args.skip_train
    t0 = time.time()
    set_seed()
    device = get_device()
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n── Part 1: Data preparation ──")
    train_full, test_raw = load_raw_data()
    train_full, test_raw = apply_age_sentinel(train_full, test_raw)
    train_df = lightgbm_train_split(train_full)
    print(f"LightGBM train split: {len(train_df):,} rows")

    encoders_path = EMBED_DIR / "label_encoders.pkl"
    if skip_train and encoders_path.exists():
        encoders = joblib.load(encoders_path)
        enc_train, enc_test, n_emb = transform_label_encoders(train_df, test_raw, encoders)
        print(f"Loaded label encoders from {encoders_path}")
    else:
        enc_train, enc_test, encoders, n_emb = fit_label_encoders(train_df, test_raw)
        joblib.dump(encoders, encoders_path)

    bin_cols = binary_columns(train_df)
    train_bin = train_df[bin_cols].values.astype(np.float32)
    test_bin = test_raw[bin_cols].values.astype(np.float32)
    train_num, test_num, _scaler = scale_numerics(train_df, test_raw)

    print("\n── Part 2: Embedding network ──")
    model = NepalEmbeddingNet(
        n_emb, EMBED_COLS, train_num.shape[1], train_bin.shape[1]
    ).to(device)
    ckpt = EMBED_DIR / "embedding_network.pt"
    if skip_train:
        if not ckpt.exists():
            raise FileNotFoundError(f"SKIP_TRAINING requires {ckpt}")
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        print(f"SKIP_TRAINING: loaded {ckpt} (epoch 37 weights, no retrain)")
    else:
        y = train_df["damage_grade"].values
        model, _train_info = train_embedding_network(
            enc_train, train_num, train_bin, y, n_emb, device
        )

    print("\n── Part 3: Extract embeddings ──")
    emb_train, emb_test = extract_embedding_features(model, enc_train, enc_test)
    print(f"Embedding feature columns: {emb_train.shape[1]}")

    print("\n── Part 4: Class-conditional geo rates ──")
    geo_train, geo_test = compute_geo_class_rates(train_df, test_raw)

    print("\n── Part 5: Interaction features ──")
    interact_train, interact_test = add_interaction_features(
        train_df, test_raw, geo_train, geo_test
    )

    print("\n── Part 6: Assemble & save ──")
    X_train_emb, X_test_emb = assemble_features(
        emb_train,
        emb_test,
        geo_train,
        geo_test,
        train_df,
        test_raw,
        train_num,
        test_num,
        interact_train,
        interact_test,
    )
    X_train_emb.to_csv(PROCESSED_DIR / "X_train_embedded.csv", index=False)
    X_test_emb.to_csv(PROCESSED_DIR / "X_test_embedded.csv", index=False)

    print(f"\nX_train_embedded shape: {X_train_emb.shape}")
    print(f"X_test_embedded shape:  {X_test_emb.shape}")
    print("\nColumn list:")
    for col in X_train_emb.columns:
        print(f"  {col}")
    print(f"\nEmbedding complete. Ready for {RETRAIN_RUN_ID}.")

    print("\n── Part 7: Diagnostics ──")
    plot_embedding_diagnostics(model, train_df, encoders)

    print(f"\n── Part 8: RunManager {RETRAIN_RUN_ID} ──")
    register_run_004(X_train_emb.shape[1])

    desc = "Embeddings + Laplace-smoothed CV geo rates — fix overfit gap"
    print(f"\n── Part 9: Trigger retrain {RETRAIN_RUN_ID} ──")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / "retrain.py"),
            "--run-id",
            RETRAIN_RUN_ID,
            "--x-train",
            str(PROCESSED_DIR / "X_train_embedded.csv"),
            "--x-test",
            str(PROCESSED_DIR / "X_test_embedded.csv"),
            "--description",
            desc,
        ],
        check=True,
        cwd=str(ROOT),
    )

    print(f"\nTotal runtime: {time.time() - t0:.1f} seconds")


if __name__ == "__main__":
    main()
