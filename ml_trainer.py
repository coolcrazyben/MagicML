"""
ml_trainer.py
Stage 4 — Machine Learning Layer

Builds a training dataset from optimization_report.json, trains an XGBoost
regression model to predict swap speed improvements, and exposes a
predict_swap_score() function for instant candidate scoring.

Usage
-----
  python ml_trainer.py          # build dataset + train + print report
"""

import json
import os
import re
import time

import requests
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPORT_PATH      = "data/optimization_report.json"
TRAINING_CSV     = "data/training_data.csv"
MODEL_PATH       = "data/combo_speed_model.json"
CARD_CACHE_DIR   = "data/cards"
RATE_LIMIT_DELAY = 0.12

# ---------------------------------------------------------------------------
# Cached model (loaded once per process)
# ---------------------------------------------------------------------------

_model_cache: XGBRegressor | None = None
_training_mean: float = 0.0
_training_std:  float = 1.0


# ---------------------------------------------------------------------------
# Scryfall full-card fetch (with local cache)
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_").lower()


def fetch_scryfall_card(name: str) -> dict:
    """
    Return full Scryfall card data for *name*.
    Results cached to data/cards/<name>.json so each card is fetched once.
    """
    os.makedirs(CARD_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CARD_CACHE_DIR, f"{_sanitize(name)}.json")

    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    try:
        response = requests.get(
            "https://api.scryfall.com/cards/named",
            params={"fuzzy": name},
            headers={"User-Agent": "MTGDeckOptimizer/1.0 (educational project)"},
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        time.sleep(RATE_LIMIT_DELAY)
        return data
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _primary_type(type_line: str) -> str:
    tl = type_line.lower()
    for t in ("creature", "instant", "sorcery", "artifact", "enchantment",
              "planeswalker", "land"):
        if t in tl:
            return t
    return "other"


def extract_features(added_data: dict, removed_data: dict) -> dict:
    """
    Extract the 15 model features for the ADDED card relative to the REMOVED card.
    """
    oracle    = (added_data.get("oracle_text") or "").lower()
    type_line = (added_data.get("type_line") or "").lower()
    keywords  = [k.lower() for k in (added_data.get("keywords") or [])]
    cmc_add   = float(added_data.get("cmc") or added_data.get("mana_value") or 0)
    cmc_rem   = float(removed_data.get("cmc") or removed_data.get("mana_value") or 0)

    added_type   = _primary_type(added_data.get("type_line") or "")
    removed_type = _primary_type(removed_data.get("type_line") or "")

    edhrec_raw  = added_data.get("edhrec_rank")
    edhrec_rank = float(edhrec_raw) if edhrec_raw is not None else 50000.0

    return {
        "cmc":                cmc_add,
        "is_creature":        int("creature"    in type_line),
        "is_instant":         int("instant"     in type_line),
        "is_sorcery":         int("sorcery"     in type_line),
        "is_artifact":        int("artifact"    in type_line),
        "is_enchantment":     int("enchantment" in type_line),
        "produces_mana":      int('add {' in oracle or "add {" in oracle),
        "draws_cards":        int("draw"                       in oracle),
        "is_tutor":           int("search your library"        in oracle),
        "has_flash":          int("flash"                      in keywords),
        "color_count":        len(added_data.get("color_identity") or []),
        "cmc_delta":          cmc_add - cmc_rem,
        "type_match":         int(added_type == removed_type),
        "edhrec_rank":        edhrec_rank,
        "is_free_spell":      int(
            cmc_add == 0 or "without paying its mana cost" in oracle
        ),
    }


FEATURE_COLS = [
    "cmc", "is_creature", "is_instant", "is_sorcery", "is_artifact",
    "is_enchantment", "produces_mana", "draws_cards", "is_tutor", "has_flash",
    "color_count", "cmc_delta", "type_match", "edhrec_rank", "is_free_spell",
]


# ---------------------------------------------------------------------------
# Step 1 — Build training dataset
# ---------------------------------------------------------------------------

def build_training_data() -> pd.DataFrame:
    """
    Load optimization_report.json, fetch Scryfall data for every swap pair,
    and return a DataFrame ready for training.

    Rows are APPENDED to data/training_data.csv across sessions.
    """
    if not os.path.exists(REPORT_PATH):
        raise FileNotFoundError(f"Report not found: {REPORT_PATH}")

    with open(REPORT_PATH, encoding="utf-8") as f:
        report = json.load(f)

    swaps = report.get("all_swaps_tested", [])
    if not swaps:
        raise ValueError("No swaps found in optimization_report.json")

    # Determine which (remove, add) pairs are already in the CSV so we only
    # append genuinely new rows.
    existing_pairs: set[tuple[str, str]] = set()
    if os.path.exists(TRAINING_CSV):
        try:
            existing_df = pd.read_csv(TRAINING_CSV)
            for _, row in existing_df.iterrows():
                existing_pairs.add((str(row["removed_card"]).lower(),
                                    str(row["added_card"]).lower()))
        except Exception:
            pass

    new_rows: list[dict] = []
    total = len(swaps)

    print(f"Processing {total} swap records from optimization report...")
    for i, swap in enumerate(swaps, 1):
        remove_name = swap.get("remove", "")
        add_name    = swap.get("add", "")
        score       = swap.get("score")

        if score is None:
            continue

        key = (remove_name.lower(), add_name.lower())
        if key in existing_pairs:
            continue  # already saved in a previous run

        print(f"  [{i}/{total}] Fetching Scryfall data for: {add_name} / {remove_name}",
              flush=True)

        added_data   = fetch_scryfall_card(add_name)
        removed_data = fetch_scryfall_card(remove_name)

        if not added_data or not removed_data:
            print(f"    Warning: skipping (Scryfall lookup failed)")
            continue

        feats = extract_features(added_data, removed_data)
        feats["removed_card"]      = remove_name
        feats["added_card"]        = add_name
        feats["speed_improvement"] = float(score)
        new_rows.append(feats)

    if not new_rows:
        print("  No new rows to append — dataset already up to date.")
        if os.path.exists(TRAINING_CSV):
            return pd.read_csv(TRAINING_CSV)
        return pd.DataFrame()

    new_df = pd.DataFrame(new_rows)
    os.makedirs("data", exist_ok=True)

    if os.path.exists(TRAINING_CSV):
        new_df.to_csv(TRAINING_CSV, mode="a", header=False, index=False)
        full_df = pd.read_csv(TRAINING_CSV)
    else:
        new_df.to_csv(TRAINING_CSV, index=False)
        full_df = new_df

    print(f"\nTraining CSV updated: {len(full_df)} total rows in {TRAINING_CSV}")
    return full_df


# ---------------------------------------------------------------------------
# Step 2 — Train model
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame | None = None) -> XGBRegressor:
    """
    Train an XGBRegressor on data/training_data.csv.
    Saves model to data/combo_speed_model.json.
    Returns the fitted model.
    """
    global _model_cache, _training_mean, _training_std

    if df is None:
        if not os.path.exists(TRAINING_CSV):
            raise FileNotFoundError(f"Training data not found: {TRAINING_CSV}")
        df = pd.read_csv(TRAINING_CSV)

    df = df.dropna(subset=["speed_improvement"])

    if len(df) < 20:
        print(
            "\nWarning: Not enough swap data to train reliably. "
            "Run the brute-force optimizer on 3-5 more decks first "
            "to build training data."
        )
        raise ValueError(f"Too few training rows: {len(df)}")

    print(f"\nTraining on {len(df)} swap records...")

    X = df[FEATURE_COLS]
    y = df["speed_improvement"]

    # Store mean/std for confidence labels in predict_swap_score
    _training_mean = float(y.mean())
    _training_std  = float(y.std()) if float(y.std()) > 0 else 1.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE on test set: {mae:.3f} turns")

    # Feature importances
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    print("\nFeature Importances:")
    for feat, score in ranked:
        print(f"  {feat:<20}: {score:.3f}")

    os.makedirs("data", exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    _model_cache = model
    return model


# ---------------------------------------------------------------------------
# Step 3 — Prediction function
# ---------------------------------------------------------------------------

def _load_model() -> XGBRegressor:
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run ml_trainer.py first."
        )
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    _model_cache = model

    # Reload mean/std from training CSV if available
    global _training_mean, _training_std
    if os.path.exists(TRAINING_CSV):
        try:
            df = pd.read_csv(TRAINING_CSV).dropna(subset=["speed_improvement"])
            _training_mean = float(df["speed_improvement"].mean())
            _training_std  = float(df["speed_improvement"].std()) or 1.0
        except Exception:
            pass

    return model


def predict_swap_score(add_name: str, remove_name: str) -> tuple[float, str]:
    """
    Predict the speed improvement for swapping *remove_name* -> *add_name*.

    Returns
    -------
    (predicted_improvement, confidence_label)
        predicted_improvement : float  — turns faster (positive = better)
        confidence_label      : str    — "High confidence" or "Medium confidence"
    """
    model       = _load_model()
    added_data  = fetch_scryfall_card(add_name)
    removed_data = fetch_scryfall_card(remove_name)

    feats = extract_features(added_data, removed_data)
    X = pd.DataFrame([feats])[FEATURE_COLS]
    pred = float(model.predict(X)[0])

    threshold = _training_mean + 2.0 * _training_std
    confidence = "High confidence" if pred > threshold else "Medium confidence"
    return pred, confidence


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("  ML TRAINER — Stage 4")
    print("=" * 72)

    # Step 1
    print("\n[Step 1] Building training dataset from optimization report...")
    try:
        df = build_training_data()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    if df.empty or len(df) < 20:
        print(
            "\nNot enough swap data to train reliably. "
            "Run the brute-force optimizer on 3-5 more decks first "
            "to build training data."
        )
        return

    # Data quality gate
    rows = len(df)
    print()
    if rows < 1000:
        print(f"WARNING: Only {rows} rows. Model will not be reliable.")
        print("Run data_pipeline.py to generate more training data.")
    elif rows < 10000:
        print(f"FAIR: {rows} rows. Model has basic pattern recognition.")
    elif rows < 50000:
        print(f"GOOD: {rows} rows. Model is reasonably reliable.")
    else:
        print(f"EXCELLENT: {rows} rows. Model is well trained.")
    print()

    # Step 2
    print("\n[Step 2] Training XGBoost model...")
    try:
        train_model(df)
    except ValueError as exc:
        print(f"Training skipped: {exc}")
        return

    print("\n" + "=" * 72)
    print("  Training complete. Model ready for use in ml_optimizer.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
