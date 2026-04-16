"""
ml_optimizer.py
Stage 4 — ML-Powered Optimizer

Replaces brute-force swap simulation with instant XGBoost predictions.
Queries 20 Scryfall candidates per swappable card and scores them all
in milliseconds using the trained model.

Usage
-----
  from ml_optimizer import get_ml_recommendations
  recs = get_ml_recommendations(decklist, matched_combos, color_identity)
"""

import time

from optimizer import (
    categorize_deck,
    _fetch_card_info,
    _fetch_scryfall_candidates,
    DB_PATH,
    MAX_CANDIDATES_PER_CARD,
)
from combo_detector import MatchedCombo
from ml_trainer import predict_swap_score, _load_model, FEATURE_COLS

# Double the candidate count for ML mode (simulation is free)
ML_CANDIDATES_PER_CARD = 20


def _top_feature_label(add_name: str, remove_name: str) -> str:
    """
    Return a human-readable explanation of the single most influential
    feature for this candidate using the loaded model's feature importances.
    """
    import pandas as pd
    from ml_trainer import fetch_scryfall_card, extract_features

    model       = _load_model()
    added_data  = fetch_scryfall_card(add_name)
    removed_data = fetch_scryfall_card(remove_name)
    feats = extract_features(added_data, removed_data)

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)

    # Map feature name → readable reason
    reason_map = {
        "cmc_delta":      f"Lower/different CMC (delta {feats['cmc_delta']:+.0f})",
        "is_tutor":       "Tutors for combo pieces",
        "produces_mana":  "Generates mana — accelerates combos",
        "cmc":            f"Low CMC ({feats['cmc']:.0f})",
        "draws_cards":    "Draws cards",
        "is_free_spell":  "Free or zero-cost spell",
        "has_flash":      "Flash enables instant-speed play",
        "edhrec_rank":    "Highly ranked on EDHREC",
        "is_creature":    "Creature synergy",
        "is_instant":     "Instant speed",
        "is_sorcery":     "Sorcery type synergy",
        "is_artifact":    "Artifact synergy",
        "is_enchantment": "Enchantment synergy",
        "type_match":     "Same type as removed card",
        "color_count":    "Color identity advantage",
    }

    for feat, _ in ranked:
        if feat in reason_map:
            return reason_map[feat]

    return "Overall profile matches fast combo decks"


def get_ml_recommendations(
    decklist:       list[str],
    matched_combos: list[MatchedCombo],
    color_identity: set[str],
    commanders:     list[str] | None = None,
    db_path:        str = DB_PATH,
) -> list[dict]:
    """
    Score swap candidates instantly using the trained XGBoost model.

    Returns a list of up to 10 recommendation dicts:
      {
        "remove":                str,
        "add":                   str,
        "predicted_improvement": float,
        "confidence":            str,
        "top_feature":           str,
      }
    """
    t_start = time.time()

    # Ensure model is loaded before we start (fails fast if not trained)
    _load_model()

    # ---- Categorize deck ------------------------------------------------
    locked, swappable = categorize_deck(
        decklist,
        commanders or [],
        matched_combos,
        db_path,
    )
    if not swappable:
        print("  No swappable cards found.")
        return []

    print(f"  Scoring candidates for {len(swappable)} swappable cards...")

    all_scored: list[dict] = []

    for card_name in swappable:
        card_info = _fetch_card_info(card_name, db_path)
        if card_info is None:
            continue

        # Temporarily raise candidate limit for ML mode
        import optimizer as _opt
        orig_limit = _opt.MAX_CANDIDATES_PER_CARD
        _opt.MAX_CANDIDATES_PER_CARD = ML_CANDIDATES_PER_CARD

        candidates = _fetch_scryfall_candidates(card_name, card_info, color_identity)

        _opt.MAX_CANDIDATES_PER_CARD = orig_limit

        for cand in candidates:
            add_name = cand["name"]
            try:
                pred, conf = predict_swap_score(add_name, card_name)
            except Exception:
                continue

            all_scored.append({
                "remove":                card_name,
                "add":                   add_name,
                "predicted_improvement": pred,
                "confidence":            conf,
                "_top_feature_pending":  True,  # computed lazily for top-10 only
            })

    # ---- Rank and return top 10 ----------------------------------------
    all_scored.sort(key=lambda x: x["predicted_improvement"], reverse=True)
    top_10 = all_scored[:10]

    # Fill in top_feature for the winners only (avoids N Scryfall calls upfront)
    for rec in top_10:
        rec["top_feature"] = _top_feature_label(rec["add"], rec["remove"])
        del rec["_top_feature_pending"]

    elapsed = time.time() - t_start
    print(f"  ML scoring complete in {elapsed:.1f}s  ({len(all_scored)} candidates scored)")

    return top_10
