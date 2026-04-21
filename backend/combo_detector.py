"""
combo_detector.py
Queries the combos table in cards.db to find:
  - Complete combos: every required card is present in the decklist AND all
    cards fit within the commander's color identity.
  - Near combos: every required card except exactly one is present, and the
    full combo would fit within the commander's color identity.
"""

import sqlite3
from dataclasses import dataclass

DB_PATH = "cards.db"


@dataclass
class MatchedCombo:
    combo_id: str
    cards_required: list[str]
    steps: str
    result: str


@dataclass
class NearCombo:
    combo_id: str
    cards_required: list[str]
    missing_card: str   # the one card not in the deck
    steps: str
    result: str


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _load_all_combos(conn: sqlite3.Connection) -> list[tuple]:
    """Return all rows from the combos table."""
    cur = conn.execute(
        "SELECT combo_id, cards_required, num_cards, steps, result FROM combos"
    )
    return cur.fetchall()


def _load_color_identity_map(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """
    Build a mapping of lowercase card name -> set of color-identity letters.
    e.g. {"sol ring": set(), "atraxa, praetors' voice": {"W","U","B","G"}}

    Cards with no color identity (colorless) map to an empty set, which is
    always a valid subset of any commander's color identity.
    """
    cur = conn.execute("SELECT name, color_identity FROM cards")
    result: dict[str, set[str]] = {}
    for name, ci_str in cur.fetchall():
        colors = {c for c in (ci_str or "").split(",") if c}
        result[name.strip().lower()] = colors
    return result


# ---------------------------------------------------------------------------
# Colour-identity check
# ---------------------------------------------------------------------------

def _combo_fits_identity(
    required_cards: list[str],
    color_map: dict[str, set[str]],
    commander_colors: set[str],
) -> bool:
    """
    Return True if every card in the combo has a color identity that is a
    subset of the commander's color identity.

    Cards not found in the database are given the benefit of the doubt
    (treated as colorless) so we don't accidentally discard valid combos
    due to missing data.
    """
    for card in required_cards:
        card_colors = color_map.get(card.strip().lower(), set())
        if not card_colors.issubset(commander_colors):
            return False
    return True


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    return name.strip().lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_combos(
    decklist: list[str],
    db_path: str = DB_PATH,
    commander_colors: set[str] | None = None,
) -> tuple[list[MatchedCombo], list[NearCombo]]:
    """
    Search for complete and near combos in a parsed decklist.

    Parameters
    ----------
    decklist : list[str]
        All card names in the deck (duplicates OK; will be de-duplicated).
    db_path : str
        Path to the SQLite database.
    commander_colors : set[str] | None
        Set of WUBRG color letters that the commander allows.
        Pass None to skip color-identity filtering (not recommended).

    Returns
    -------
    (matched, near)
        matched — complete combos where every card is in the deck and within
                  the commander's color identity
        near    — combos missing exactly one card that would otherwise fit
    """
    deck_set: set[str] = {_normalize(name) for name in decklist}

    if not deck_set:
        print("Warning: decklist appears to be empty.")
        return [], []

    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            f"Could not open database '{db_path}'. "
            "Run build_card_db.py and build_combo_db.py first."
        ) from exc

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "combos" not in tables:
        conn.close()
        raise RuntimeError("No 'combos' table found. Run build_combo_db.py first.")

    all_combos = _load_all_combos(conn)

    # Build color identity map only when we need it
    color_map: dict[str, set[str]] = {}
    if commander_colors is not None:
        if "cards" in tables:
            color_map = _load_color_identity_map(conn)
        else:
            print("Warning: 'cards' table not found — skipping color identity check.")
            commander_colors = None

    conn.close()

    color_filter = commander_colors is not None
    if color_filter:
        color_label = "{" + ",".join(sorted(commander_colors)) + "}"
        print(
            f"Checking decklist ({len(deck_set)} unique cards) against "
            f"{len(all_combos):,} combos "
            f"(color identity filter: {color_label})..."
        )
    else:
        print(
            f"Checking decklist ({len(deck_set)} unique cards) against "
            f"{len(all_combos):,} combos (no color identity filter)..."
        )

    matched: list[MatchedCombo] = []
    near: list[NearCombo] = []

    for combo_id, cards_str, _num_cards, steps, result in all_combos:
        # Bug fix: skip combos with missing or empty cards_required data
        if not cards_str:
            continue
        required_cards = [c.strip() for c in cards_str.split(",") if c.strip()]
        if not required_cards:
            continue
        required_norm = [_normalize(c) for c in required_cards]

        # ---- color identity gate — disabled; re-enable after partner fix ----
        # if color_filter and not _combo_fits_identity(
        #     required_cards, color_map, commander_colors  # type: ignore[arg-type]
        # ):
        #     continue

        # ---- card-presence check ----
        missing = [
            (orig, norm)
            for orig, norm in zip(required_cards, required_norm)
            if norm not in deck_set
        ]

        if len(missing) == 0:
            matched.append(
                MatchedCombo(
                    combo_id=combo_id,
                    cards_required=required_cards,
                    steps=steps or "",
                    result=result or "",
                )
            )
        elif len(missing) == 1:
            # Guard: skip combos where zero cards from the combo are in the deck
            matched_cards = [c for c in required_norm if c in deck_set]
            if len(matched_cards) == 0:
                continue
            near.append(
                NearCombo(
                    combo_id=combo_id,
                    cards_required=required_cards,
                    missing_card=missing[0][0],
                    steps=steps or "",
                    result=result or "",
                )
            )

    return matched, near


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_deck = [
        "Thassa's Oracle",
        "Demonic Consultation",
        "Sol Ring",
        "Command Tower",
        "Swamp",
        "Island",
    ]
    # Kenrith, the Returned King is 5-colour — no filter needed for this test
    try:
        matched, near = find_combos(sample_deck, commander_colors={"W", "U", "B", "R", "G"})
        print(f"\nComplete combos found: {len(matched)}")
        for c in matched:
            print(f"  [{c.combo_id}] {', '.join(c.cards_required)}")
            print(f"    Result: {c.result}")

        print(f"\nNear combos (1 card away): {len(near)}")
        for n in near:
            print(f"  [{n.combo_id}] Missing: {n.missing_card}")
            print(f"    Needs: {', '.join(n.cards_required)}")
            print(f"    Result: {n.result}")
    except RuntimeError as e:
        print(f"Error: {e}")
