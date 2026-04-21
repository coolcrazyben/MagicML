"""
speed_calculator.py
Estimates the theoretical assembly speed (turn number) for each confirmed
Commander combo in a decklist.

Combines:
  - mana_engine  : turn-by-turn mana profile
  - tutor_engine : per-piece tutor coverage
"""

import sqlite3
from dataclasses import dataclass, field

from mana_engine import build_mana_profile, FAST_MANA
from tutor_engine import build_tutor_coverage_multi
from combo_detector import MatchedCombo

DB_PATH = "cards.db"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ComboSpeed:
    combo_id: str
    cards_required: list[str]
    total_mana_needed: int
    mana_cost_display: str          # e.g. "3 (1UB)"
    tutor_coverage: dict[str, list[str]]  # {piece: [tutors]}
    estimated_fastest_turn: int
    estimated_average_turn: float
    speed_rating: str               # "turn 1-2" | "turn 2-3" | "turn 3-4" | "turn 4+"


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

def _fetch_cards(names: list[str], db_path: str) -> dict[str, dict]:
    conn = sqlite3.connect(db_path)
    cards: dict[str, dict] = {}
    for name in names:
        row = conn.execute(
            "SELECT name, mana_cost, mana_value, type_line FROM cards "
            "WHERE lower(name) = lower(?)",
            (name,),
        ).fetchone()
        if row:
            cards[name.lower()] = {
                "name": row[0],
                "mana_cost": row[1] or "",
                "mana_value": float(row[2] or 0),
                "type_line": row[3] or "",
            }
    conn.close()
    return cards


# ---------------------------------------------------------------------------
# Mana-cost display
# ---------------------------------------------------------------------------

def _mana_cost_display(cards: list[str], card_db: dict[str, dict]) -> str:
    """
    Return a human-readable mana cost summary, e.g. "3 (1UB)".
    Strips braces and combines mana costs of all combo pieces.
    """
    total_cmc = 0
    symbols: list[str] = []
    for name in cards:
        data = card_db.get(name.lower(), {})
        total_cmc += int(data.get("mana_value", 0))
        raw = data.get("mana_cost", "")
        # extract symbols from {X} braces
        import re
        for sym in re.findall(r"\{([^}]+)\}", raw):
            if sym not in ("T",):
                symbols.append(sym)

    symbol_str = "".join(symbols)
    if symbol_str:
        return f"{total_cmc} ({symbol_str})"
    return str(total_cmc)


# ---------------------------------------------------------------------------
# Speed rating helpers
# ---------------------------------------------------------------------------

def _fastest_turn(total_cmc: int, mana_profile: dict[str, float]) -> int:
    """
    Return the earliest turn where available mana >= total CMC needed.
    Caps at 6 (represented as "turn 4+" in the rating).
    """
    for turn in range(1, 6):
        if mana_profile.get(f"turn_{turn}", 0) >= total_cmc:
            return turn
    return 6  # couldn't assemble by turn 5


def _average_turn(fastest: int, tutor_coverage: dict[str, list[str]]) -> float:
    """
    Estimated average turn, accounting for tutor density.

    More tutors for each piece → smaller gap between fastest and average.
    Range: fastest + 0.5 (many tutors) to fastest + 1.5 (no tutors).
    """
    if not tutor_coverage:
        return fastest + 1.5

    # Minimum tutor coverage across all pieces (bottleneck piece)
    min_tutors = min(len(t_list) for t_list in tutor_coverage.values())

    # Scale: 0 tutors → +1.5,  5+ tutors → +0.5
    adjustment = max(0.5, 1.5 - min_tutors * 0.2)
    return round(fastest + adjustment, 1)


def _speed_rating(avg_turn: float) -> str:
    if avg_turn < 2.5:
        return "turn 1-2"
    if avg_turn < 3.5:
        return "turn 2-3"
    if avg_turn < 4.5:
        return "turn 3-4"
    return "turn 4+"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_combo_speeds(
    decklist: list[str],
    matched_combos: list[MatchedCombo],
    db_path: str = DB_PATH,
) -> list[ComboSpeed]:
    """
    Estimate assembly speed for every confirmed combo.

    Parameters
    ----------
    decklist : list[str]
        Full decklist with duplicates (needed for accurate land count).
    matched_combos : list[MatchedCombo]
        Confirmed combos from combo_detector.find_combos().
    db_path : str
        Path to cards.db.

    Returns
    -------
    list[ComboSpeed]
        One ComboSpeed per confirmed combo, sorted by fastest_turn.
    """
    if not matched_combos:
        return []

    unique_deck_names = list(dict.fromkeys(decklist))  # for tutor engine

    print("\n[Speed Calculator] Building mana profile...")
    mana_profile = build_mana_profile(decklist, db_path)

    print("[Speed Calculator] Mana profile:")
    for turn, mana in mana_profile.items():
        print(f"  Turn {turn}: {mana:.2f} mana available")

    # Collect all combo-piece names (deduplicated) for bulk DB fetch
    all_piece_names: list[str] = []
    for combo in matched_combos:
        for card in combo.cards_required:
            if card not in all_piece_names:
                all_piece_names.append(card)

    print("\n[Speed Calculator] Fetching combo piece data...")
    piece_db = _fetch_cards(all_piece_names, db_path)

    print("[Speed Calculator] Computing tutor coverage...")
    tutor_coverage_all = build_tutor_coverage_multi(
        unique_deck_names, all_piece_names, db_path
    )

    results: list[ComboSpeed] = []

    for combo in matched_combos:
        total_cmc = sum(
            int(piece_db.get(c.lower(), {}).get("mana_value", 0))
            for c in combo.cards_required
        )

        mana_display = _mana_cost_display(combo.cards_required, piece_db)

        # Tutor coverage for this combo's pieces only
        coverage = {
            card: tutor_coverage_all.get(card, [])
            for card in combo.cards_required
        }

        fastest = _fastest_turn(total_cmc, mana_profile)
        average = _average_turn(fastest, coverage)
        rating = _speed_rating(average)

        results.append(
            ComboSpeed(
                combo_id=combo.combo_id,
                cards_required=combo.cards_required,
                total_mana_needed=total_cmc,
                mana_cost_display=mana_display,
                tutor_coverage=coverage,
                estimated_fastest_turn=fastest,
                estimated_average_turn=average,
                speed_rating=rating,
            )
        )

    results.sort(key=lambda cs: (cs.estimated_fastest_turn, cs.estimated_average_turn))
    return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from combo_detector import find_combos

    deck = [
        "Thassa's Oracle", "Demonic Consultation",
        "Sol Ring", "Arcane Signet", "Dark Ritual",
        "Demonic Tutor", "Vampiric Tutor", "Imperial Seal", "Mystical Tutor",
        "Command Tower", "Swamp", "Island",
    ] + ["Swamp"] * 18 + ["Island"] * 12

    matched, _ = find_combos(deck, commander_colors={"W", "U", "B", "R", "G"})
    print(f"\nFound {len(matched)} complete combo(s).")
    speeds = calculate_combo_speeds(deck, matched)
    for cs in speeds:
        print(f"\nCombo [{cs.combo_id}]: {', '.join(cs.cards_required)}")
        print(f"  Fastest turn    : {cs.estimated_fastest_turn}")
        print(f"  Average turn    : {cs.estimated_average_turn}")
        print(f"  Speed rating    : {cs.speed_rating}")
        print(f"  Mana required   : {cs.mana_cost_display}")
        for piece, tutors in cs.tutor_coverage.items():
            print(f"  Tutors for {piece}: {tutors or ['(none)']}")
