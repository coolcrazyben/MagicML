"""
backend/src/analyzer.py
Core analysis and deck-building logic extracted for API use.

Exposes two callable functions:
  analyze_decklist(decklist_text: str) -> dict
  build_deck(commander_name: str, budget: float) -> dict

Path resolution: all relative paths in the imported modules (DB, model
files) resolve against BACKEND_DIR so this works when run from any CWD.
"""

import os
import sys
import sqlite3

# ---------------------------------------------------------------------------
# Bootstrap: add backend directory to path so local modules are importable
# ---------------------------------------------------------------------------

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ---------------------------------------------------------------------------
# Path helpers (env-overridable for Railway)
# ---------------------------------------------------------------------------

COLOR_NAMES = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}


def _db_path() -> str:
    return os.environ.get(
        "COMBO_DB_PATH",
        os.path.join(BACKEND_DIR, "cards.db"),
    )


# ---------------------------------------------------------------------------
# Commander helpers (mirrors api/analyze.py, but path-aware)
# ---------------------------------------------------------------------------

def _lookup_commander_colors(name: str) -> set:
    try:
        conn = sqlite3.connect(_db_path())
        row = conn.execute(
            "SELECT color_identity FROM cards WHERE lower(name) = lower(?)", (name,)
        ).fetchone()
        conn.close()
    except Exception:
        return set()
    if not row:
        return set()
    return {c for c in (row[0] or "").split(",") if c}


def _detect_commander(unique_cards: set) -> str | None:
    try:
        conn = sqlite3.connect(_db_path())
    except Exception:
        return None
    candidates = []
    for name in unique_cards:
        row = conn.execute(
            "SELECT name, type_line, oracle_text FROM cards WHERE lower(name) = lower(?)",
            (name,),
        ).fetchone()
        if not row:
            continue
        _, type_line, oracle = row
        if "can be your commander" in (oracle or "").lower():
            conn.close()
            return row[0]
        if (
            "legendary" in (type_line or "").lower()
            and "creature" in (type_line or "").lower()
        ):
            candidates.append(row[0])
    conn.close()
    return candidates[0] if len(candidates) == 1 else None


# ---------------------------------------------------------------------------
# analyze_decklist
# ---------------------------------------------------------------------------

def analyze_decklist(decklist_text: str) -> dict:
    """
    Parse a decklist, run combo detection, speed analysis, and ML
    recommendations.  Returns a dict matching the /analyze response schema.

    If the ML model file does not exist the ml_recommendations key is
    returned as an empty list rather than raising an error.
    """
    from deck_parser import parse_decklist, unique_card_names
    from combo_detector import find_combos
    from speed_calculator import calculate_combo_speeds

    db_path = _db_path()

    try:
        decklist, commanders = parse_decklist(decklist_text)
    except Exception as exc:
        raise ValueError(f"Parse error: {exc}") from exc

    if not decklist:
        raise ValueError("Could not parse any cards — check the format")

    unique = unique_card_names(decklist)
    all_commanders = list(commanders)

    if not all_commanders:
        detected = _detect_commander(unique)
        if detected:
            all_commanders = [detected]

    commander_colors: set = set()
    for cmd in all_commanders:
        commander_colors |= _lookup_commander_colors(cmd)

    try:
        matched, near = find_combos(
            decklist,
            db_path=db_path,
            commander_colors=commander_colors or None,
        )
    except RuntimeError:
        matched, near = [], []

    speeds = []
    if matched:
        try:
            speeds = calculate_combo_speeds(decklist, matched, db_path=db_path)
        except Exception:
            pass

    # Aggregate speed summary
    speed_analysis: dict = {}
    if speeds:
        speed_analysis = {
            "fastest_combo_turns": float(
                min(s.estimated_fastest_turn for s in speeds)
            ),
            "average_combo_turns": round(
                sum(s.estimated_average_turn for s in speeds) / len(speeds), 1
            ),
        }

    # ML recommendations — gracefully omitted if model not ready
    ml_recs: list[dict] = []
    try:
        from ml_optimizer import get_ml_recommendations

        recs = get_ml_recommendations(
            decklist,
            matched,
            commander_colors,
            commanders=all_commanders,
        )
        ml_recs = [
            {
                "remove": r["remove"],
                "add": r["add"],
                "predicted_improvement": round(float(r["predicted_improvement"]), 3),
                "confidence": r["confidence"],
                "reason": r.get("top_feature", ""),
            }
            for r in recs
        ]
    except FileNotFoundError:
        pass  # model not trained yet — omit ml_recommendations
    except Exception:
        pass  # any other error — gracefully omit

    return {
        "commander": " + ".join(all_commanders) if all_commanders else None,
        "color_identity": sorted(commander_colors),
        "total_cards": len(decklist),
        "confirmed_combos": [
            {
                "combo_id": c.combo_id,
                "cards": list(c.cards_required),
                "result": c.result,
            }
            for c in matched
        ],
        "near_combos": [
            {
                "combo_id": n.combo_id,
                "cards_in_deck": [
                    c for c in n.cards_required if c != n.missing_card
                ],
                "missing_card": n.missing_card,
                "result": n.result,
            }
            for n in near
        ],
        "speed_analysis": speed_analysis,
        "ml_recommendations": ml_recs,
    }


# ---------------------------------------------------------------------------
# build_deck helpers
# ---------------------------------------------------------------------------

def _get_strategy_notes(commander: dict, slots: dict) -> str:
    """Build a strategy notes string from the commander's oracle text."""
    oracle = (commander.get("oracle_text") or "").lower()
    name = commander["name"]
    wins = [
        c["name"] for c in slots.get("Win Conditions", []) if not c.get("is_basic")
    ]
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
    return (
        f"This deck is built around {name}, leveraging {theme_str}. "
        f"The ramp suite ensures {name} hits the battlefield ahead of schedule. "
        f"Card draw and tutors maintain hand advantage and find key pieces. "
        f"Primary win conditions include {win_str}. "
        f"Interaction and removal protect your board while disrupting opponents."
    )


def _build_decklist_txt(commander_name: str, slots: dict) -> str:
    """Render the 99-card deck as a plain-text string."""
    lines: list[str] = [f"Commander: {commander_name}", ""]

    for category, card_list in slots.items():
        if category == "Lands":
            continue
        for card in card_list:
            lines.append(f"1x {card['name']}")

    # Lands: aggregate basics, list non-basics individually
    basic_counts: dict[str, int] = {}
    non_basics: list[str] = []
    for card in slots.get("Lands", []):
        if card.get("is_basic"):
            basic_counts[card["name"]] = basic_counts.get(card["name"], 0) + 1
        else:
            non_basics.append(card["name"])
    for name in non_basics:
        lines.append(f"1x {name}")
    for name, qty in sorted(basic_counts.items()):
        lines.append(f"{qty}x {name}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# build_deck
# ---------------------------------------------------------------------------

def build_deck(commander_name: str, budget: float) -> dict:
    """
    Build a Commander deck using ML quality scoring.
    Returns a dict matching the /build-deck response schema.
    """
    # deck_builder.py lives in BACKEND_DIR — chdir ensures its hardcoded
    # relative paths (MODEL_PATH, OUTPUT_DIR) resolve correctly.
    original_cwd = os.getcwd()
    os.chdir(BACKEND_DIR)
    try:
        return _build_deck_inner(commander_name, budget)
    finally:
        os.chdir(original_cwd)


def _build_deck_inner(commander_name: str, budget: float) -> dict:
    from deck_builder import (
        fetch_commander,
        fetch_card_pool,
        train_deck_model,
        score_and_enrich_cards,
        fill_slots,
    )

    commander = fetch_commander(commander_name)
    if commander is None:
        raise ValueError(
            f"Could not find a valid commander for '{commander_name}'"
        )

    cmd_name = commander["name"]
    color_identity = commander.get("color_identity") or []

    cards = fetch_card_pool(color_identity, cmd_name, budget)
    if not cards:
        raise ValueError("No candidate cards found — check network connection")

    model = train_deck_model(cards, retrain=False)
    cards = score_and_enrich_cards(model, cards)
    slots, total_cost = fill_slots(cards, color_identity, budget)

    categories = {
        cat: [
            {
                "name": c["name"],
                "price": round(c.get("price_usd", 0.0), 2),
                "score": round(c.get("quality_score", 0.0), 4),
            }
            for c in card_list
        ]
        for cat, card_list in slots.items()
    }

    return {
        "commander": cmd_name,
        "colors": color_identity,
        "total_cost": total_cost,
        "budget": budget,
        "categories": categories,
        "strategy_notes": _get_strategy_notes(commander, slots),
        "decklist_txt": _build_decklist_txt(cmd_name, slots),
    }
