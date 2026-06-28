#!/usr/bin/env python3
"""
SIMULATION OF 10,000 SYNTHETIC MOVIES FOR DEEP LEARNING TRAINING

Simulate 10,000 synthetic movies using the same Gaussian copula
methodologyscaled up for deep-learning training.

Two-level hierarchical class-conditional sampling using
gaussian copulas.
The copula preserves BOTH:
  - Each feature's marginal distribution (by mapping through the empirical CDF)
  - The correlation structure between features (by sampling correlated normals)

Outputs 3 files in ml_dataset/data/model_ready/movie_success_v7/:
  - scene_movie_metadata_v7_synthetic.csv   (10,000 rows)
  - participant_features_v7_synthetic.csv   (10,000 × 43 = 430,000 rows)
  - movie_features_v7_synthetic.csv         (10,000 rows)

All synthetic rows include is_synthetic=1.
"""
from __future__ import annotations
import csv
import math
import random
import statistics
from pathlib import Path
from collections import Counter, defaultdict
import warnings

import numpy as np

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR  = PROJECT / "ml_dataset" / "data" / "model_ready" / "movie_success_v6"
DATA_DIR   = PROJECT / "ml_dataset" / "data" / "model_ready" / "movie_success_v7"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SYNTHETIC = 10000
N_PARTICIPANTS = 43
SYNTHETIC_PARTICIPANT_ID_START = 100000
TIER_THRESHOLD = 7.5
CPI_BASE = 322.0
CPI = {1978: 65.2, 1979: 72.6, 1983: 99.6, 1988: 118.3, 1993: 144.5,
       1996: 156.9, 1998: 163.0, 1999: 166.6, 2006: 201.6, 2012: 229.6,
       2025: 322.0}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def phi(z):
    """Standard normal CDF (vectorised over numpy array)."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def phi_inv(p):
    """
    Inverse standard normal CDF using Beasley-Springer-Moro approximation.
    Vectorised over numpy array. Accuracy ~1e-9 in mid-range.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)

    a = [-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]

    p_low = 0.02425
    p_high = 1 - p_low
    out = np.empty_like(p)

    # Mid region
    mid = (p >= p_low) & (p <= p_high)
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        num = ((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]
        den = ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1
        out[mid] = num * q / den

    # Lower tail
    low = p < p_low
    if np.any(low):
        q = np.sqrt(-2 * np.log(p[low]))
        num = ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
        den = (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1
        out[low] = num / den

    # Upper tail (mirror of lower)
    high = p > p_high
    if np.any(high):
        q = np.sqrt(-2 * np.log(1 - p[high]))
        num = ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
        den = (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1
        out[high] = -num / den

    return out

# COPULA UTILITIES

def empirical_ranks(values):
    """
    Return rank-based percentiles in (0, 1), avoiding boundaries.
    Uses (rank + 0.5) / n convention; tied values get the same rank.
    """
    n = len(values)
    if n == 0:
        return np.array([])
    # argsort-of-argsort gives 0-indexed rank
    ranks = np.argsort(np.argsort(values)).astype(float)
    return (ranks + 0.5) / n


def inverse_empirical(percentiles, sorted_values):
    """Linear-interpolating inverse ECDF: percentile [0,1] → value in original units."""
    n = len(sorted_values)
    if n == 0:
        return np.zeros_like(percentiles)
    if n == 1:
        return np.full_like(percentiles, sorted_values[0])
    idx = percentiles * (n - 1)
    idx = np.clip(idx, 0, n - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def fit_copula(X, regularize=0.05):
    """
    Fit a Gaussian copula on real data X (n × p).
    Returns model with: cholesky factor, sorted real values per feature,
                        and a flag indicating constant features.
    """
    n, p = X.shape

    sorted_per_feat = []
    constant_value = []                  # for features with zero variance
    is_constant = np.zeros(p, dtype=bool)

    Z = np.zeros((n, p))
    for j in range(p):
        col = X[:, j]
        valid_mask = ~np.isnan(col)
        valid = col[valid_mask]

        if len(valid) == 0:
            sorted_per_feat.append(np.array([0.0]))
            constant_value.append(0.0)
            is_constant[j] = True
            continue

        sorted_col = np.sort(valid)
        sorted_per_feat.append(sorted_col)

        # Constant feature
        if sorted_col[0] == sorted_col[-1]:
            is_constant[j] = True
            constant_value.append(sorted_col[0])
            Z[:, j] = 0.0
            continue
        constant_value.append(0.0)

        # Compute z-scores via empirical CDF → Φ⁻¹
        valid_ranks = empirical_ranks(valid)
        Z_valid = phi_inv(valid_ranks)

        # Place Z_valid back into the full column; impute NaN as 0 (median in z-space)
        z_col = np.zeros(n)
        z_col[valid_mask] = Z_valid
        z_col[~valid_mask] = 0.0
        Z[:, j] = z_col

    # Correlation matrix of Z
    # Skip constant columns when computing correlation
    var = Z.var(axis=0)
    nonconst = var > 1e-10
    Z_nc = Z[:, nonconst]
    C_nc = np.corrcoef(Z_nc, rowvar=False) if Z_nc.shape[1] > 1 else np.eye(1)
    C_nc = np.nan_to_num(C_nc, nan=0.0)

    # Reconstruct full correlation matrix: identity for constant columns
    C = np.eye(p)
    nc_indices = np.where(nonconst)[0]
    for i_idx, i in enumerate(nc_indices):
        for j_idx, j in enumerate(nc_indices):
            C[i, j] = C_nc[i_idx, j_idx]

    # Regularise: shrink toward identity for numerical stability
    C = (1 - regularize) * C + regularize * np.eye(p)

    # Cholesky decomposition (with extra regularisation if needed)
    for extra in [0.0, 0.05, 0.2, 0.5]:
        try:
            C_try = (1 - extra) * C + extra * np.eye(p)
            L = np.linalg.cholesky(C_try)
            break
        except np.linalg.LinAlgError:
            continue
    else:
        raise RuntimeError("Cholesky failed even with heavy regularisation.")

    return {
        "L": L, "p": p, "n": n,
        "sorted_per_feat": sorted_per_feat,
        "is_constant": is_constant,
        "constant_value": constant_value,
    }


def sample_copula(model, n_samples):
    """Draw n_samples × p synthetic feature vectors from the fitted copula."""
    L = model["L"]
    p = model["p"]
    sorted_per_feat = model["sorted_per_feat"]
    is_constant = model["is_constant"]
    constant_value = model["constant_value"]

    # Step 1: correlated standard normals
    noise = np.random.standard_normal((n_samples, p))
    Z_new = noise @ L.T

    # Step 2: → uniform via Φ
    U_new = phi(Z_new)

    # Step 3: → original space via inverse ECDF
    X_new = np.empty((n_samples, p))
    for j in range(p):
        if is_constant[j]:
            X_new[:, j] = constant_value[j]
        else:
            X_new[:, j] = inverse_empirical(U_new[:, j], sorted_per_feat[j])

    return X_new


def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)

def to_float(s):
    if s is None or s == "" or s == "None":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")

# 1. LOAD REAL DATA
print("Loading real data...")

meta_cols, meta_rows = load_csv(INPUT_DIR / "scene_movie_metadata_v6.csv")
print(f"  Metadata: {len(meta_rows)} movies × {len(meta_cols)} cols")

ppt_cols, ppt_rows = load_csv(INPUT_DIR / "participant_features_v6.csv")
print(f"  Participants: {len(ppt_rows)} rows × {len(ppt_cols)} cols")

movie_imdb = {r["movie_id"]: float(r["imdb_rating"]) for r in meta_rows
              if r.get("imdb_rating")}
tier_of_movie = {mid: ("high" if rating >= TIER_THRESHOLD else "lower")
                 for mid, rating in movie_imdb.items()}

movies_in_tier = defaultdict(list)
for mid, t in tier_of_movie.items():
    movies_in_tier[t].append(mid)

print("\nTier assignment:")
for t in ["high", "lower"]:
    mids = movies_in_tier[t]
    ratings = [movie_imdb[m] for m in mids]
    print(f"  {t:6s}: {len(mids)} movies, IMDb {min(ratings):.1f}–{max(ratings):.1f}")

# 2. PREPARE PARTICIPANT-LEVEL FEATURES PER TIER

ID_COLS = {"participant_id", "condition", "movie_id"}
HAS_COLS = {c for c in ppt_cols if c.startswith("has_")}
SR_INTEGER_COLS = {c for c in ppt_cols
                   if c.startswith("sr_emo_") and not c.startswith("sr_emo_bc_")}
SR_INTEGER_COLS.update({"sr_sam_valence", "sr_sam_arousal", "sr_sam_motivation"})

feat_cols = [c for c in ppt_cols if c not in ID_COLS]
n_feat = len(feat_cols)

tier_data = defaultdict(list)
for r in ppt_rows:
    mid = r["movie_id"]
    if mid not in tier_of_movie:
        continue
    vec = np.array([to_float(r[c]) for c in feat_cols])
    tier_data[tier_of_movie[mid]].append(vec)

print(f"\nFeature matrix sizes:")
for t, vecs in tier_data.items():
    arr = np.vstack(vecs)
    print(f"  Tier '{t}': {arr.shape[0]} × {arr.shape[1]}, NaN rate {np.isnan(arr).mean():.1%}")

# 3. FIT GAUSSIAN COPULAS

print("\nFitting Gaussian copulas (preserves marginals + correlations)...")
tier_models = {}
for t, vecs in tier_data.items():
    X = np.vstack(vecs)
    model = fit_copula(X, regularize=0.05)
    tier_models[t] = model
    n_const = int(model["is_constant"].sum())
    print(f"  Tier '{t}': fitted on {model['n']} rows × {model['p']} features  "
          f"({n_const} constant features)")


# 4. METADATA SAMPLING DICTIONARIES

tier_metadata = defaultdict(lambda: defaultdict(list))
for r in meta_rows:
    mid = r["movie_id"]
    if mid not in tier_of_movie:
        continue
    for col in meta_cols:
        tier_metadata[tier_of_movie[mid]][col].append(r[col])

CATEGORICAL_SCENE = ["cut_count", "brightness", "motion_intensity", "audio_loudness",
                     "silence_ratio", "music_presence", "dialogue_density",
                     "face_screen_time_ratio", "lead_screen_time_ratio",
                     "budget_categorical"]


# 5. POST-PROCESS SAMPLED FEATURES

def post_process_row(vec, feat_cols):
    """Force has_* masks → 1, round integer self-report features."""
    out = vec.copy().astype(float)
    for i, c in enumerate(feat_cols):
        if c in HAS_COLS:
            out[i] = 1.0
        elif c.startswith("sr_emo_") and not c.startswith("sr_emo_bc_"):
            out[i] = round(np.clip(out[i], 1, 5))
        elif c in {"sr_sam_valence", "sr_sam_arousal", "sr_sam_motivation"}:
            out[i] = round(np.clip(out[i], 1, 9))
        elif c.startswith("sr_emo_bc_"):
            out[i] = round(np.clip(out[i], -4, 4))
    return out

# 6. GENERATE SYNTHETIC MOVIES

def sample_categorical(values):
    return random.choice(values)

n_per_tier = N_SYNTHETIC // 2
print(f"\nGenerating {N_SYNTHETIC} synthetic movies ({n_per_tier} per tier)...")

synthetic_metadata = []
synthetic_participants = []
synthetic_movie_features = []

syn_idx = 0
for tier in ["high", "lower"]:
    model = tier_models[tier]
    real_imdb = [float(v) for v in tier_metadata[tier]["imdb_rating"]]
    real_revenue = [int(v) for v in tier_metadata[tier]["worldwide_gross_revenue_usd_orignal_release"]]
    real_wom_log = [float(v) for v in tier_metadata[tier]["wom_multiplier_log"]]
    real_year = [int(v) for v in tier_metadata[tier]["release_year"]]
    real_duration = [int(v) for v in tier_metadata[tier]["clip_duration_s"]]

    imdb_min, imdb_max = min(real_imdb), max(real_imdb)
    rev_log_mean = statistics.mean([math.log10(v) for v in real_revenue])
    rev_log_std = statistics.stdev([math.log10(v) for v in real_revenue])
    wom_log_mean = statistics.mean(real_wom_log)
    wom_log_std = statistics.stdev(real_wom_log)

    for i in range(n_per_tier):
        syn_idx += 1
        syn_movie_id = f"SYN_M{syn_idx:03d}"
        syn_scene_id = f"SYN_S{syn_idx:03d}"

        # Sample 43 synthetic participants from the copula
        ppts = sample_copula(model, N_PARTICIPANTS)
        ppts = np.array([post_process_row(p, feat_cols) for p in ppts])

        syn_emotion = sample_categorical(tier_metadata[tier]["targeted_emotion"])
        syn_year = sample_categorical(real_year)
        syn_duration = sample_categorical(real_duration)
        syn_genre = sample_categorical(tier_metadata[tier]["genre_primary"])
        syn_genre2 = sample_categorical(tier_metadata[tier]["genre_secondary"])
        syn_country = sample_categorical(tier_metadata[tier]["country_of_origin"])
        syn_budget_cat = sample_categorical(tier_metadata[tier]["budget_categorical"])
        syn_imdb = round(random.uniform(imdb_min, imdb_max), 1)

        syn_rev = max(100_000, int(10 ** np.random.normal(rev_log_mean, rev_log_std)))
        syn_wom_log = np.random.normal(wom_log_mean, wom_log_std)
        syn_wom = 10 ** syn_wom_log
        syn_opening = max(10_000, int(syn_rev / syn_wom))

        cpi_year = CPI.get(syn_year, 200.0)
        inflate = CPI_BASE / cpi_year
        syn_rev_2025 = round(syn_rev * inflate, 2)
        syn_opn_2025 = round(syn_opening * inflate, 2)

        syn_categoricals = {col: sample_categorical(tier_metadata[tier][col])
                            for col in CATEGORICAL_SCENE}

        meta_row = {
            "scene_id": syn_scene_id,
            "movie_id": syn_movie_id,
            "movie_title": f"Synthetic_Movie_{syn_idx:03d}",
            "imdb_link": "",
            "targeted_emotion": syn_emotion,
            "clip_duration_s": syn_duration,
            "release_year": syn_year,
            "genre_primary": syn_genre,
            "genre_secondary": syn_genre2,
            "country_of_origin": syn_country,
            "imdb_rating": syn_imdb,
            "worldwide_gross_revenue_usd_orignal_release": syn_rev,
            "opening_weekend_usd": syn_opening,
            "revenue_usd_2025": syn_rev_2025,
            "opening_weekend_usd_2025": syn_opn_2025,
            "wom_multiplier": round(syn_wom, 4),
            "wom_multiplier_log": round(syn_wom_log, 4),
            "scene_notes": f"Synthetic — class-conditional copula sample from {tier} IMDb tier",
            "data_collection_notes": (
                f"Synthetic movie {syn_idx} of 40 (Gaussian copula sampling, "
                f"tier='{tier}', seed=42). Marginals match real distributions; "
                f"correlation structure preserved."
            ),
            "is_synthetic": 1,
        }
        meta_row.update(syn_categoricals)
        synthetic_metadata.append(meta_row)

        # Participant-level rows
        for p_i in range(N_PARTICIPANTS):
            syn_pid = SYNTHETIC_PARTICIPANT_ID_START + (syn_idx - 1) * N_PARTICIPANTS + p_i
            row = {"participant_id": syn_pid, "condition": syn_emotion, "movie_id": syn_movie_id}
            for j, c in enumerate(feat_cols):
                v = float(ppts[p_i, j])
                if abs(v) < 1e-6:
                    row[c] = 0
                else:
                    row[c] = round(v, 6)
            row["is_synthetic"] = 1
            synthetic_participants.append(row)

        # Movie-level aggregation
        m_row = {
            "movie_id": syn_movie_id,
            "condition": syn_emotion,
            "n_participants": N_PARTICIPANTS,
            "imdb_rating": syn_imdb,
            "wom_multiplier": round(syn_wom, 4),
            "wom_multiplier_log": round(syn_wom_log, 4),
            "is_synthetic": 1,
        }
        for j, c in enumerate(feat_cols):
            col_vals = ppts[:, j]
            m_row[f"{c}__mean"] = round(float(col_vals.mean()), 6)
            m_row[f"{c}__std"] = round(float(col_vals.std(ddof=1)), 6)
        synthetic_movie_features.append(m_row)

# 7. WRITE OUTPUTS

syn_meta_cols = list(meta_cols) + ["is_synthetic"]
out_meta = DATA_DIR / "scene_movie_metadata_v7_synthetic.csv"
with open(out_meta, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=syn_meta_cols, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(synthetic_metadata)
print(f"\nWrote {out_meta.name}: {len(synthetic_metadata)} rows × {len(syn_meta_cols)} cols")

syn_ppt_cols = list(ppt_cols) + ["is_synthetic"]
out_ppt = DATA_DIR / "participant_features_v7_synthetic.csv"
with open(out_ppt, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=syn_ppt_cols, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(synthetic_participants)
print(f"Wrote {out_ppt.name}: {len(synthetic_participants)} rows × {len(syn_ppt_cols)} cols")

mov_cols, _ = load_csv(INPUT_DIR / "movie_features_v6.csv")
syn_mov_cols = list(mov_cols)
if "is_synthetic" not in syn_mov_cols:
    syn_mov_cols.append("is_synthetic")
for c in ("imdb_rating", "wom_multiplier", "wom_multiplier_log"):
    if c not in syn_mov_cols:
        syn_mov_cols.append(c)

out_mov = DATA_DIR / "movie_features_v7_synthetic.csv"
with open(out_mov, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=syn_mov_cols, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(synthetic_movie_features)
print(f"Wrote {out_mov.name}: {len(synthetic_movie_features)} rows × {len(syn_mov_cols)} cols")

# 8. VALIDATION SUMMARY

print("\n" + "=" * 60)
print("  VALIDATION — synthetic vs real distributions")
print("=" * 60)

real_imdb_all = list(movie_imdb.values())
syn_imdb_all = [m["imdb_rating"] for m in synthetic_metadata]
print(f"\nIMDb rating:")
print(f"  Real      (n={len(real_imdb_all)}): mean={statistics.mean(real_imdb_all):.2f}, "
      f"std={statistics.stdev(real_imdb_all):.2f}, "
      f"range={min(real_imdb_all):.1f}–{max(real_imdb_all):.1f}")
print(f"  Synthetic (n={len(syn_imdb_all)}): mean={statistics.mean(syn_imdb_all):.2f}, "
      f"std={statistics.stdev(syn_imdb_all):.2f}, "
      f"range={min(syn_imdb_all):.1f}–{max(syn_imdb_all):.1f}")

real_wom_log = [float(r["wom_multiplier_log"]) for r in meta_rows]
syn_wom_log = [m["wom_multiplier_log"] for m in synthetic_metadata]
print(f"\nWOM multiplier (log10):")
print(f"  Real      (n={len(real_wom_log)}): mean={statistics.mean(real_wom_log):.2f}, "
      f"std={statistics.stdev(real_wom_log):.2f}, "
      f"range={min(real_wom_log):.2f}–{max(real_wom_log):.2f}")
print(f"  Synthetic (n={len(syn_wom_log)}): mean={statistics.mean(syn_wom_log):.2f}, "
      f"std={statistics.stdev(syn_wom_log):.2f}, "
      f"range={min(syn_wom_log):.2f}–{max(syn_wom_log):.2f}")

print(f"\nDONE. Combined corpus (real + synthetic) = 50 movies.")
