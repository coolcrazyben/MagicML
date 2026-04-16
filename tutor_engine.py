"""
tutor_engine.py
Analyzes tutor chains in a Commander deck.

Scans every card's oracle text for "search your library" and categorises
tutors by what card types they can find.  For each combo piece, returns a
list of tutors in the deck that are capable of finding it.
"""

import re
import sqlite3

DB_PATH = "cards.db"

# Card types relevant to tutoring
ALL_TYPES = frozenset(
    ["instant", "sorcery", "creature", "artifact", "enchantment", "planeswalker"]
)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _fetch_deck_cards(unique_names: list[str], db_path: str) -> dict[str, dict]:
    conn = sqlite3.connect(db_path)
    cards: dict[str, dict] = {}
    for name in unique_names:
        row = conn.execute(
            "SELECT name, mana_cost, mana_value, type_line, oracle_text "
            "FROM cards WHERE lower(name) = lower(?)",
            (name,),
        ).fetchone()
        if row:
            cards[name.lower()] = {
                "name": row[0],
                "mana_cost": row[1] or "",
                "mana_value": float(row[2] or 0),
                "type_line": row[3] or "",
                "oracle_text": row[4] or "",
            }
    conn.close()
    return cards


# ---------------------------------------------------------------------------
# Tutor classification
# ---------------------------------------------------------------------------

def _classify_tutor(oracle_text: str) -> set[str] | None:
    """
    Determine what card types this card can tutor for.

    Returns
    -------
    set[str] | None
        A set containing one or more of: 'any', 'instant', 'sorcery',
        'creature', 'artifact', 'enchantment', 'planeswalker', 'land'.
        Returns None if the card is not a tutor.
    """
    oracle_lower = oracle_text.lower()
    if "search your library" not in oracle_lower:
        return None

    found: set[str] = set()

    # --- specific-type tutors ---
    if re.search(r"for (?:a |an )?instant\b", oracle_lower):
        found.add("instant")
    if re.search(r"for (?:a |an )?sorcery\b", oracle_lower):
        found.add("sorcery")
    if re.search(r"for (?:a |an )?instant or sorcery\b", oracle_lower):
        found.update(["instant", "sorcery"])
    if re.search(r"for (?:a |an )?creature\b", oracle_lower):
        found.add("creature")
    if re.search(r"for (?:a |an )?artifact\b", oracle_lower):
        found.add("artifact")
    if re.search(r"for (?:a |an )?enchantment\b", oracle_lower):
        found.add("enchantment")
    if re.search(r"for (?:a |an )?planeswalker\b", oracle_lower):
        found.add("planeswalker")
    # land tutors (usually not relevant for combos, but track anyway)
    if re.search(r"for (?:a |an )?(?:basic )?land\b", oracle_lower):
        found.add("land")

    # --- any-card tutors ---
    # Match "for a card" not immediately followed by a type word.
    # Negative lookahead keeps "for a creature card" from matching here.
    if re.search(
        r"for (?:up to \w+ )?(?:a |an )?cards?"
        r"(?!\s+named\b)(?!\s+with the same)"
        r"(?:\s*[,.\n]|\s+with\b|\s+that\b|\s+from\b|\s*$)",
        oracle_lower,
    ):
        # Only classify as "any" if no specific types were already found
        # (avoids tagging "search your library for a creature card" as any-card).
        if not found:
            found.add("any")

    return found if found else None


def _card_types(type_line: str) -> set[str]:
    """Extract lowercase spell types from a type line."""
    types: set[str] = set()
    tl = type_line.lower()
    for t in ALL_TYPES:
        if t in tl:
            types.add(t)
    return types


def _tutor_can_find(tutor_types: set[str], piece_types: set[str]) -> bool:
    """Return True if the tutor can find at least one of the piece's types."""
    if "any" in tutor_types:
        # 'any' tutors can find any non-land card
        return bool(piece_types)  # True as long as piece has a spell type
    return bool(tutor_types & piece_types)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_tutor_coverage(
    unique_deck_names: list[str],
    combo_cards: list[str],
    db_path: str = DB_PATH,
) -> dict[str, list[str]]:
    """
    For every combo piece card, return a list of deck cards that can tutor it.

    Parameters
    ----------
    unique_deck_names : list[str]
        All unique card names in the deck.
    combo_cards : list[str]
        Names of the combo pieces to analyse.

    Returns
    -------
    dict[str, list[str]]
        {combo_piece_name: [tutor_name, ...]}
    """
    print("  [Tutor Engine] Scanning oracle texts for tutors...")
    card_db = _fetch_deck_cards(unique_deck_names, db_path)
    piece_db = _fetch_deck_cards(combo_cards, db_path)

    # ---- identify tutors in the deck ------------------------------------
    tutors: list[tuple[str, set[str]]] = []   # (name, what_it_finds)
    for name in unique_deck_names:
        data = card_db.get(name.lower(), {})
        oracle = data.get("oracle_text", "")
        result = _classify_tutor(oracle)
        if result:
            tutors.append((data.get("name", name), result))

    print(f"  [Tutor Engine] {len(tutors)} tutors identified in the deck.")

    # ---- map tutors to each combo piece ---------------------------------
    coverage: dict[str, list[str]] = {}
    for piece_name in combo_cards:
        data = piece_db.get(piece_name.lower(), {})
        piece_types = _card_types(data.get("type_line", ""))
        matching = [
            t_name
            for t_name, t_types in tutors
            if t_name.lower() != piece_name.lower()   # don't count self
            and _tutor_can_find(t_types, piece_types)
        ]
        coverage[piece_name] = matching

    return coverage


def build_tutor_coverage_multi(
    unique_deck_names: list[str],
    all_combo_cards: list[str],
    db_path: str = DB_PATH,
) -> dict[str, list[str]]:
    """
    Efficient version for analysing many combos at once: fetches card data
    once and computes tutor coverage for all combo piece names supplied.
    """
    print("  [Tutor Engine] Building deck-wide tutor index...")
    all_names = list(dict.fromkeys(unique_deck_names + all_combo_cards))
    card_db = _fetch_deck_cards(all_names, db_path)

    # Build tutor index
    tutors: list[tuple[str, set[str]]] = []
    for name in unique_deck_names:
        data = card_db.get(name.lower(), {})
        oracle = data.get("oracle_text", "")
        result = _classify_tutor(oracle)
        if result:
            tutors.append((data.get("name", name), result))

    print(f"  [Tutor Engine] {len(tutors)} tutors found in deck.")

    coverage: dict[str, list[str]] = {}
    for piece_name in dict.fromkeys(all_combo_cards):  # deduplicate order-preserving
        data = card_db.get(piece_name.lower(), {})
        piece_types = _card_types(data.get("type_line", ""))
        matching = [
            t_name
            for t_name, t_types in tutors
            if t_name.lower() != piece_name.lower()
            and _tutor_can_find(t_types, piece_types)
        ]
        coverage[piece_name] = matching

    return coverage


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    deck = [
        "Demonic Tutor", "Vampiric Tutor", "Mystical Tutor", "Imperial Seal",
        "Fabricate", "Spellseeker", "Sol Ring", "Command Tower",
        "Thassa's Oracle", "Demonic Consultation", "Swamp", "Island",
    ]
    combo = ["Thassa's Oracle", "Demonic Consultation"]

    cov = build_tutor_coverage(deck, combo)
    print("\nTutor Coverage:")
    for piece, t_list in cov.items():
        print(f"  {piece}: {t_list or ['(none)']}")
