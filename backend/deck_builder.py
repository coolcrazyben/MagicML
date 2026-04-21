"""
deck_builder.py
Commander deck generator using ML quality scoring.

Given a commander name and budget, builds a full 99-card Commander deck by
fetching a Scryfall card pool, training an XGBoost quality model, filling
deck slots by category, and exporting a .txt decklist.

Usage
-----
  python deck_builder.py
  python deck_builder.py --retrain
"""

import argparse
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests
from xgboost import XGBRegressor

# Reuse the cached Scryfall fetch utility from ml_trainer
from ml_trainer import fetch_scryfall_card

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

MODEL_PATH       = "data/deck_builder_model.json"
SCRYFALL_SEARCH  = "https://api.scryfall.com/cards/search"
RATE_LIMIT_DELAY = 0.1
MAX_POOL_SIZE    = 600
OUTPUT_DIR       = "outputs/generated_decks"
USER_AGENT       = "MTGDeckBuilder/1.0 (educational project)"
COLOR_ORDER      = "WUBRG"
SEP              = "=" * 72
SUBSEP           = "-" * 72

BASIC_LANDS: dict[str, str] = {
    "W": "Plains",
    "U": "Island",
    "B": "Swamp",
    "R": "Mountain",
    "G": "Forest",
}

DECK_FEATURE_COLS: list[str] = [
    "cmc", "is_creature", "is_instant", "is_sorcery", "is_artifact",
    "is_enchantment", "is_land", "produces_mana", "draws_cards", "is_tutor",
    "has_flash", "is_free_spell", "color_count", "edhrec_rank", "price_usd",
]

SLOT_TARGETS: dict[str, int] = {
    "Ramp":           12,
    "Card Draw":      10,
    "Tutors":          4,
    "Interaction":    10,
    "Win Conditions":  5,
    "Support/Value":  24,
    "Lands":          34,
}

TYPE_DISPLAY_ORDER: list[str] = [
    "Creatures", "Lands", "Enchantments", "Instants", "Sorceries", "Artifacts", "Other",
]


# ---------------------------------------------------------------------------
# Commander lookup
# ---------------------------------------------------------------------------

def fetch_commander(name: str) -> dict | None:
    """
    Fetch and validate commander card data from Scryfall (cached).
    Returns None if the card is not found or is not a valid commander.
    """
    data = fetch_scryfall_card(name)
    if not data:
        return None

    type_line = data.get("type_line", "")
    oracle    = (data.get("oracle_text") or "").lower()

    is_legendary_creature = "Legendary" in type_line and "Creature" in type_line
    can_be_commander      = "can be your commander" in oracle

    if not is_legendary_creature and not can_be_commander:
        print(f"  Error: '{data.get('name', name)}' is not a Legendary Creature.")
        return None

    return data


# ---------------------------------------------------------------------------
# Card pool fetch
# ---------------------------------------------------------------------------

def _oracle_text_from_card(card: dict) -> str:
    """Return oracle text, falling back to the first face for double-faced cards."""
    oracle = card.get("oracle_text")
    if oracle is not None:
        return oracle
    faces = card.get("card_faces") or []
    return faces[0].get("oracle_text", "") if faces else ""


def fetch_card_pool(
    color_identity: list[str],
    commander_name: str,
    budget: float,
) -> list[dict]:
    """
    Query Scryfall for Commander-legal cards matching the color identity.
    Paginates up to MAX_POOL_SIZE results.

    Excludes: basic lands, the commander, cards without USD price,
    and cards whose price exceeds the full budget.
    """
    sorted_ci = sorted(
        color_identity,
        key=lambda c: COLOR_ORDER.index(c) if c in COLOR_ORDER else 99,
    )
    color_str = "".join(sorted_ci) if sorted_ci else "C"

    headers      = {"User-Agent": USER_AGENT}
    first_params = {"q": f"legal:commander color<={color_str}", "order": "edhrec", "unique": "cards"}

    cards:    list[dict]    = []
    seen:     set[str]      = set()
    next_url: str | None    = SCRYFALL_SEARCH
    first    = True

    print(f"  Fetching card pool for color identity {{{color_str}}}...")

    while len(cards) < MAX_POOL_SIZE and next_url:
        try:
            time.sleep(RATE_LIMIT_DELAY)
            resp = (
                requests.get(next_url, params=first_params, headers=headers, timeout=15)
                if first
                else requests.get(next_url, headers=headers, timeout=15)
            )
            first = False
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  Warning: Scryfall request failed: {exc}")
            break

        for card in data.get("data", []):
            if len(cards) >= MAX_POOL_SIZE:
                break

            name = card.get("name", "")
            if name in seen:
                continue

            # Skip basics
            type_line = card.get("type_line", "")
            if re.search(r"\bBasic\b", type_line):
                continue

            # Skip the commander itself
            if name.lower() == commander_name.lower():
                continue

            # Must have a USD price within the full budget
            usd_raw = (card.get("prices") or {}).get("usd")
            if usd_raw is None:
                continue
            try:
                price_usd = float(usd_raw)
            except (ValueError, TypeError):
                continue
            if price_usd > budget:
                continue

            seen.add(name)
            cards.append({
                "name":           name,
                "type_line":      type_line,
                "oracle_text":    _oracle_text_from_card(card),
                "keywords":       card.get("keywords") or [],
                "cmc":            float(card.get("cmc") or 0),
                "mana_cost":      card.get("mana_cost") or "",
                "color_identity": card.get("color_identity") or [],
                "produced_mana":  card.get("produced_mana") or [],
                "edhrec_rank":    card.get("edhrec_rank"),
                "price_usd":      price_usd,
            })

        if not data.get("has_more"):
            break
        next_url = data.get("next_page")

    if len(cards) < 400:
        print(f"  Warning: Only {len(cards)} candidate cards found. Continuing.")
    else:
        print(f"  Found {len(cards)} candidate cards.")

    return cards


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_card_features(card: dict) -> dict:
    """
    Extract the 15 deck-builder features for a single card.
    Mirrors the logic in ml_trainer.extract_features() for standalone cards.
    """
    oracle    = (card.get("oracle_text") or "").lower()
    type_line = (card.get("type_line")   or "").lower()
    keywords  = [k.lower() for k in (card.get("keywords") or [])]
    cmc       = float(card.get("cmc") or 0)
    edhrec    = card.get("edhrec_rank")

    return {
        "cmc":            cmc,
        "is_creature":    int("creature"    in type_line),
        "is_instant":     int("instant"     in type_line),
        "is_sorcery":     int("sorcery"     in type_line),
        "is_artifact":    int("artifact"    in type_line),
        "is_enchantment": int("enchantment" in type_line),
        "is_land":        int("land"        in type_line),
        "produces_mana":  int("add {"       in oracle),
        "draws_cards":    int("draw"        in oracle),
        "is_tutor":       int("search your library" in oracle),
        "has_flash":      int("flash" in keywords),
        "is_free_spell":  int(cmc == 0),
        "color_count":    len(card.get("color_identity") or []),
        "edhrec_rank":    float(edhrec) if edhrec is not None else 99999.0,
        "price_usd":      float(card.get("price_usd") or 0),
    }


# ---------------------------------------------------------------------------
# Model: train or load
# ---------------------------------------------------------------------------

def _quality_score(edhrec_rank: float) -> float:
    """Lower EDHREC rank (= more popular) maps to a higher quality score in (0, 1]."""
    return 1.0 / (1.0 + math.log1p(edhrec_rank))


def train_deck_model(cards: list[dict], retrain: bool = False) -> XGBRegressor:
    """
    Train an XGBRegressor on *cards* using quality_score as the target,
    or load the saved model if it already exists and --retrain was not set.
    Saves to data/deck_builder_model.json.
    """
    if not retrain and os.path.exists(MODEL_PATH):
        print(f"  Loading existing deck builder model from {MODEL_PATH}")
        model = XGBRegressor()
        model.load_model(MODEL_PATH)
        return model

    print(f"  Training deck builder model on {len(cards)} candidate cards...")

    rows = [extract_card_features(c) for c in cards]
    df   = pd.DataFrame(rows)[DECK_FEATURE_COLS]
    df   = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    y    = np.array([_quality_score(r["edhrec_rank"]) for r in rows])

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(df, y)

    os.makedirs("data", exist_ok=True)
    model.save_model(MODEL_PATH)

    importances = dict(zip(DECK_FEATURE_COLS, model.feature_importances_))
    ranked      = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)

    print(f"  Deck builder model trained on {len(cards)} candidate cards.")
    print("  Top predictive features:")
    for feat, imp in ranked[:5]:
        print(f"    {feat:<20}: {imp:.3f}")

    return model


def score_and_enrich_cards(model: XGBRegressor, cards: list[dict]) -> list[dict]:
    """
    Score all cards using the model. Attaches quality_score and the extracted
    boolean features to each card dict. Returns cards sorted by score descending.
    """
    rows  = [extract_card_features(c) for c in cards]
    df    = pd.DataFrame(rows)[DECK_FEATURE_COLS]
    df    = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    preds = model.predict(df)

    for card, score, feats in zip(cards, preds, rows):
        card["quality_score"] = float(score)
        for k, v in feats.items():
            card[k] = v  # normalize and attach all computed fields

    return sorted(cards, key=lambda c: c["quality_score"], reverse=True)


# ---------------------------------------------------------------------------
# Slot filling
# ---------------------------------------------------------------------------

def _is_interaction(oracle: str) -> bool:
    ol = oracle.lower()
    return any(kw in ol for kw in ("destroy", "exile", "counter", "return", "sacrifice"))


def fill_slots(
    cards:          list[dict],
    color_identity: list[str],
    budget:         float,
) -> tuple[dict[str, list[dict]], float]:
    """
    Fill the 99 Commander deck slots by category, respecting the budget.

    Cards are drawn from *cards* (pre-sorted by quality_score descending).
    A per-card price cap (5× the average non-land cost) prevents a few expensive
    cards from exhausting the budget before all 65 non-land slots are filled.
    Returns (slots, total_cost_of_non_basic_cards).
    """
    # No single card may cost more than 5× the average non-land budget per slot.
    # This prevents 1-2 expensive staples from consuming half the budget.
    per_card_max = (budget / 65) * 5

    slots:        dict[str, list[dict]] = {cat: [] for cat in SLOT_TARGETS}
    slotted:      set[str]              = set()
    running_cost: float                 = 0.0

    def can_add(card: dict, price_cap: float | None = None) -> bool:
        price = card["price_usd"]
        if price_cap is not None and price > price_cap:
            return False
        return (
            card["name"] not in slotted
            and running_cost + price <= budget
        )

    def place(category: str, card: dict) -> None:
        nonlocal running_cost
        slots[category].append(card)
        slotted.add(card["name"])
        running_cost += card["price_usd"]

    # ---- Ramp: non-land mana producers, prefer CMC 2 --------------------
    ramp_pool = [
        c for c in cards
        if c.get("produces_mana") and not c.get("is_land")
    ]
    ramp_pool.sort(key=lambda c: (abs(c["cmc"] - 2), -c["quality_score"]))
    for card in ramp_pool:
        if len(slots["Ramp"]) >= SLOT_TARGETS["Ramp"]:
            break
        if can_add(card, per_card_max):
            place("Ramp", card)

    # ---- Card Draw -------------------------------------------------------
    draw_pool = sorted(
        [c for c in cards if c.get("draws_cards") and not c.get("is_land")],
        key=lambda c: -c["quality_score"],
    )
    for card in draw_pool:
        if len(slots["Card Draw"]) >= SLOT_TARGETS["Card Draw"]:
            break
        if can_add(card, per_card_max):
            place("Card Draw", card)

    # ---- Tutors ----------------------------------------------------------
    tutor_pool = sorted(
        [c for c in cards if c.get("is_tutor") and not c.get("is_land")],
        key=lambda c: -c["quality_score"],
    )
    for card in tutor_pool:
        if len(slots["Tutors"]) >= SLOT_TARGETS["Tutors"]:
            break
        if can_add(card, per_card_max):
            place("Tutors", card)

    # ---- Interaction: prefer instants ------------------------------------
    interact_pool = [
        c for c in cards
        if _is_interaction(c.get("oracle_text", "")) and not c.get("is_land")
    ]
    interact_pool.sort(key=lambda c: (-c.get("is_instant", 0), -c["quality_score"]))
    for card in interact_pool:
        if len(slots["Interaction"]) >= SLOT_TARGETS["Interaction"]:
            break
        if can_add(card, per_card_max):
            place("Interaction", card)

    # ---- Win Conditions: CMC >= 4, highest quality, not yet slotted -----
    win_pool = sorted(
        [c for c in cards if c["cmc"] >= 4 and not c.get("is_land")],
        key=lambda c: -c["quality_score"],
    )
    for card in win_pool:
        if len(slots["Win Conditions"]) >= SLOT_TARGETS["Win Conditions"]:
            break
        if can_add(card, per_card_max):
            place("Win Conditions", card)

    # ---- Support/Value: remaining high scorers, non-land ----------------
    for card in cards:
        if len(slots["Support/Value"]) >= SLOT_TARGETS["Support/Value"]:
            break
        if not card.get("is_land") and can_add(card, per_card_max):
            place("Support/Value", card)

    # ---- Second pass: fill underfilled slots without price cap ----------
    # Allows expensive-but-affordable cards that were skipped in the first pass.
    for category in list(SLOT_TARGETS.keys()):
        if category == "Lands":
            continue
        target = SLOT_TARGETS[category]
        if len(slots[category]) >= target:
            continue
        for card in cards:
            if len(slots[category]) >= target:
                break
            if not card.get("is_land") and can_add(card):
                place(category, card)

    # ---- Safety net: fill Support/Value with cheapest remaining cards ---
    # Ensures we always reach 65 non-land cards as long as budget allows.
    cheap_pool = sorted(
        [c for c in cards if not c.get("is_land") and c["name"] not in slotted],
        key=lambda c: c["price_usd"],
    )
    for card in cheap_pool:
        non_land = sum(len(slots[s]) for s in slots if s != "Lands")
        if non_land >= 65:
            break
        if can_add(card):
            place("Support/Value", card)

    # ---- Lands: non-basic utility lands then fill with basics -----------
    land_pool = sorted(
        [c for c in cards if c.get("is_land")],
        key=lambda c: -c["quality_score"],
    )
    for card in land_pool:
        if len(slots["Lands"]) >= SLOT_TARGETS["Lands"]:
            break
        if can_add(card):
            place("Lands", card)

    # Fill remaining land slots with proportional basic lands
    basics_needed = SLOT_TARGETS["Lands"] - len(slots["Lands"])
    if basics_needed > 0:
        for basic_name, qty in _distribute_basics(color_identity, basics_needed):
            for _ in range(qty):
                slots["Lands"].append({
                    "name":          basic_name,
                    "price_usd":     0.0,
                    "quality_score": 0.0,
                    "cmc":           0,
                    "is_basic":      True,
                })

    return slots, round(running_cost, 2)


def _distribute_basics(color_identity: list[str], count: int) -> list[tuple[str, int]]:
    """Distribute *count* basic land slots proportionally across colors."""
    colors = [c for c in color_identity if c in BASIC_LANDS]
    if not colors:
        return [("Wastes", count)]

    per_color = count // len(colors)
    remainder = count % len(colors)
    result: list[tuple[str, int]] = []
    for i, color in enumerate(colors):
        qty = per_color + (1 if i < remainder else 0)
        if qty > 0:
            result.append((BASIC_LANDS[color], qty))
    return result


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def _group_by_type(slots: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Re-group all slotted cards by card type for display."""
    groups: dict[str, list[dict]] = {cat: [] for cat in TYPE_DISPLAY_ORDER}
    for slot_name, card_list in slots.items():
        for card in card_list:
            if slot_name == "Lands" or card.get("is_land") or card.get("is_basic"):
                groups["Lands"].append(card)
            elif card.get("is_creature"):
                groups["Creatures"].append(card)
            elif card.get("is_enchantment"):
                groups["Enchantments"].append(card)
            elif card.get("is_instant"):
                groups["Instants"].append(card)
            elif card.get("is_sorcery"):
                groups["Sorceries"].append(card)
            elif card.get("is_artifact"):
                groups["Artifacts"].append(card)
            else:
                groups["Other"].append(card)
    return groups


def _color_display(color_identity: list[str]) -> str:
    name_map = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}
    parts = [name_map.get(c, c) for c in color_identity]
    return "/".join(parts) if parts else "Colorless"


def print_report(
    commander:      dict,
    color_identity: list[str],
    budget:         float,
    slots:          dict[str, list[dict]],
    total_cost:     float,
) -> None:
    total_cards = sum(len(v) for v in slots.values())

    print(f"\n{SEP}")
    print("  GENERATED COMMANDER DECK")
    print(SEP)
    print(f"  Commander     : {commander['name']}")
    print(f"  Colors        : {_color_display(color_identity)}")
    print(f"  Budget        : ${budget:.2f}")
    print(f"  Total Cost    : ${total_cost:.2f}")
    print(f"  Total Cards   : {total_cards}")
    print(SEP)

    type_groups = _group_by_type(slots)
    for category, card_list in type_groups.items():
        if not card_list:
            continue
        print(f"\n{category.upper()} ({len(card_list)})")
        print(SUBSEP)
        for idx, card in enumerate(card_list, 1):
            name      = card["name"]
            price_str = "—" if card.get("is_basic") else f"${card.get('price_usd', 0.0):.2f}"
            print(f"  {idx:>2}. {name:<44} {price_str:>7}")

    print(f"\n{SEP}")
    print("  STRATEGY NOTES")
    print(SEP)
    _print_strategy_notes(commander, slots)
    print(SEP)


def _print_strategy_notes(commander: dict, slots: dict[str, list[dict]]) -> None:
    oracle = (commander.get("oracle_text") or "").lower()
    name   = commander["name"]
    wins   = [c["name"] for c in slots.get("Win Conditions", []) if not c.get("is_basic")]
    win_str = ", ".join(wins[:3]) if wins else "your best threats"

    themes: list[str] = []
    if "draw" in oracle:
        themes.append("card advantage")
    if "token" in oracle or "create" in oracle:
        themes.append("token generation")
    if "+1/+1 counter" in oracle:
        themes.append("+1/+1 counter synergies")
    if "proliferate" in oracle:
        themes.append("proliferate")
    if "graveyard" in oracle or "dies" in oracle:
        themes.append("graveyard recursion")
    if "damage" in oracle:
        themes.append("direct damage")
    if "life" in oracle:
        themes.append("life gain")
    if "enters" in oracle or "enter" in oracle:
        themes.append("enters-the-battlefield triggers")
    if not themes:
        themes.append("value and tempo")

    theme_str = " and ".join(themes[:2])

    print(f"  This deck is built around {name}, leveraging {theme_str}.")
    print(f"  The ramp suite ensures {name} hits the battlefield ahead of schedule.")
    print(f"  Card draw and tutors maintain hand advantage and find key pieces.")
    print(f"  Primary win conditions include {win_str}.")
    print(
        "  Interaction and removal protect your board while"
        " disrupting your opponents' strategies."
    )


# ---------------------------------------------------------------------------
# Decklist export
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_").lower()


def export_decklist(
    commander:  dict,
    slots:      dict[str, list[dict]],
    budget:     float,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """Write the deck to outputs/generated_decks/{slug}_{budget}.txt and return the path."""
    os.makedirs(output_dir, exist_ok=True)

    slug     = _sanitize_filename(commander["name"])
    filepath = os.path.join(output_dir, f"{slug}_{int(budget)}.txt")

    lines: list[str] = [f"Commander: {commander['name']}", ""]

    # Non-land categories first
    for category, card_list in slots.items():
        if category == "Lands":
            continue
        for card in card_list:
            lines.append(f"1x {card['name']}")

    # Lands: non-basics individually, basics aggregated
    basic_counts: dict[str, int] = {}
    non_basics:   list[str]      = []
    for card in slots.get("Lands", []):
        if card.get("is_basic"):
            basic_counts[card["name"]] = basic_counts.get(card["name"], 0) + 1
        else:
            non_basics.append(card["name"])

    for name in non_basics:
        lines.append(f"1x {name}")
    for name, qty in sorted(basic_counts.items()):
        lines.append(f"{qty}x {name}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Commander deck with ML quality scoring."
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining the deck builder model even if one already exists.",
    )
    args = parser.parse_args()

    print(SEP)
    print("  MAGIC ML — COMMANDER DECK BUILDER")
    print(SEP)

    # ---- Step 1: Commander input & lookup --------------------------------
    print()
    commander_name = input("  Enter commander name: ").strip()
    if not commander_name:
        print("  Error: No commander name provided.")
        sys.exit(1)

    budget_raw = input("  Enter budget in USD (e.g. 150): ").strip()
    try:
        budget = float(budget_raw)
        if budget <= 0:
            raise ValueError
    except ValueError:
        print(f"  Error: Invalid budget '{budget_raw}'.")
        sys.exit(1)

    print(f"\n  Looking up '{commander_name}' on Scryfall...")
    commander = fetch_commander(commander_name)
    if commander is None:
        print(f"  Error: Could not find a valid commander for '{commander_name}'.")
        sys.exit(1)

    cmd_name       = commander["name"]
    color_identity = commander.get("color_identity") or []

    print(f"  Found    : {cmd_name}")
    print(f"  Type     : {commander.get('type_line', '')}")
    print(f"  Colors   : {_color_display(color_identity)}")

    # ---- Step 2: Card pool -----------------------------------------------
    print(f"\n[Step 2] Fetching card pool from Scryfall...")
    cards = fetch_card_pool(color_identity, cmd_name, budget)

    if not cards:
        print("  Error: No candidate cards found. Check your network connection.")
        sys.exit(1)

    # ---- Steps 3 & 4: Feature extraction + train/load model -------------
    print(f"\n[Step 3 & 4] Preparing deck quality model...")
    model = train_deck_model(cards, retrain=args.retrain)

    # ---- Score all cards -------------------------------------------------
    print(f"\n  Scoring {len(cards)} candidate cards...")
    cards = score_and_enrich_cards(model, cards)

    # ---- Step 5: Slot filling --------------------------------------------
    print(f"\n[Step 5] Building deck (budget: ${budget:.2f})...")
    slots, total_cost = fill_slots(cards, color_identity, budget)

    total_placed = sum(len(v) for v in slots.values())
    print(f"  Deck complete: {total_placed} cards | Total cost: ${total_cost:.2f}")

    # ---- Step 6: Printed report ------------------------------------------
    print_report(commander, color_identity, budget, slots, total_cost)

    # ---- Step 7: Export decklist -----------------------------------------
    filepath = export_decklist(commander, slots, budget)
    print(f"\n  Decklist saved to {filepath}")
    print(f"  Run 'python main.py' and load this file to analyze combos.")
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
