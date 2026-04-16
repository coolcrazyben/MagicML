"""
mana_engine.py
Models mana development across turns 1-5 for a Commander deck.

Estimates average available mana per turn assuming a conservative
opening hand (3 lands, proportional mana rocks).

Outputs a mana_profile dict: {1: float, 2: float, ..., 5: float}
representing average mana available for spending each turn.
"""

import re
import sqlite3

DB_PATH = "cards.db"

# ---------------------------------------------------------------------------
# Fast mana: hardcoded net mana gain (mana produced minus mana cost to play)
# These provide burst acceleration beyond normal land development.
# ---------------------------------------------------------------------------
FAST_MANA: dict[str, float] = {
    "mana crypt": 2.0,          # {0} → {C}{C}
    "mana vault": 2.0,          # {1} → {C}{C}{C}  (net +2 per activation)
    "chrome mox": 1.0,          # {0} → 1 colored  (exile a non-artifact card)
    "mox diamond": 1.0,         # {0} → 1 colored  (discard a land)
    "lotus petal": 1.0,         # {0} → 1 any, sac (one-shot burst)
    "lion's eye diamond": 2.0,  # {0} → {X}{X}{X}, sac + discard (combo piece, ~net 2)
    "dark ritual": 2.0,         # {B} → {B}{B}{B}  (net +2)
    "cabal ritual": 1.0,        # {1}{B} → {B}{B}{B} (net +1; threshold ignored)
}

# Hardcoded fast/accelerant lookup — checked before oracle-text scanning.
# Covers non-standard mana text that the regex parser would miss.
FAST_MANA_CARDS: dict[str, float] = {
    "sol ring": 2.0,
    "mana crypt": 2.0,
    "mana vault": 4.0,
    "chrome mox": 1.0,
    "mox diamond": 1.0,
    "lotus petal": 1.0,
    "lion's eye diamond": 3.0,
    "dark ritual": 2.0,
    "cabal ritual": 2.0,
    "arcane signet": 1.0,
    "fellwar stone": 1.0,
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _fetch_deck_cards(unique_names: list[str], db_path: str) -> dict[str, dict]:
    """Bulk-fetch card rows for every unique card name in the deck."""
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
# Oracle-text parsing
# ---------------------------------------------------------------------------

def _parse_rock_production(oracle_text: str) -> float:
    """
    Return the number of mana produced per tap from a mana rock's oracle text.

    Handles:
      {T}: Add {C}{C}.              → 2
      {T}: Add {W} or {U}.          → 1  (choice, pick one)
      {T}: Add {R}, {G}, or {W}.    → 1  (choice)
      {T}: Add {B}{B}{B}.           → 3
      {T}: Add one mana of any color → 1
    """
    oracle_lower = oracle_text.lower()

    if "add one mana of any color" in oracle_lower:
        return 1.0
    if "add two mana of any color" in oracle_lower:
        return 2.0

    max_prod = 0.0

    for line in oracle_text.split("\n"):
        ll = line.lower()
        if "add " not in ll:
            continue

        # Find all "add <mana symbols>" segments in this line.
        # Mana symbols are {X} where X is a colour letter, C, or a digit.
        segments = re.findall(
            r"add\s+((?:\{[^}]+\}(?:\s*(?:,\s*|\bor\b\s*))?)+)",
            ll,
        )

        for seg in segments:
            is_choice = bool(re.search(r"\bor\b", seg))
            if is_choice:
                # "or"-separated symbols: player picks one → produces 1 mana
                prod = 1.0
            else:
                symbols = re.findall(r"\{([^}]+)\}", seg)
                prod = 0.0
                for sym in symbols:
                    s = sym.strip()
                    if s.isdigit():
                        prod += int(s)
                    elif s.upper() in ("C", "W", "U", "B", "R", "G"):
                        prod += 1.0
            if prod > max_prod:
                max_prod = prod

    return max_prod


# ---------------------------------------------------------------------------
# Card classification
# ---------------------------------------------------------------------------

def _is_land(card: dict) -> bool:
    return "Land" in card.get("type_line", "")


def _is_mana_rock(card: dict) -> bool:
    tl = card.get("type_line", "")
    oracle = card.get("oracle_text", "")
    return (
        "Artifact" in tl
        and "Land" not in tl
        and "add {" in oracle.lower()
        and "{t}: add" in oracle.lower()   # must be a tap ability, not a ritual
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_mana_profile(
    deck_names: list[str],
    db_path: str = DB_PATH,
) -> dict[str, float]:
    """
    Model mana development across turns 1-5.

    Parameters
    ----------
    deck_names : list[str]
        Full decklist, duplicates included (needed for accurate land count).
        Typically the same list returned by parse_decklist().
    db_path : str
        Path to cards.db.

    Returns
    -------
    dict[str, float]
        {"turn_1": float, ..., "turn_5": float} — conservative estimate of
        average mana available to spend each turn.
    """
    print("  [Mana Engine] Fetching card data for mana analysis...")
    unique_names = list(dict.fromkeys(deck_names))  # for DB lookups
    card_db = _fetch_deck_cards(unique_names, db_path)

    DECK_SIZE = len(deck_names) or 100
    OPENER = 7  # cards in opening hand

    # ---- classify (use full list so basic-land duplicates are counted) ---
    lands = [n for n in deck_names
             if _is_land(card_db.get(n.lower(), {}))]

    fast_mana_cards = [n for n in unique_names
                       if n.lower() in FAST_MANA_CARDS]

    mana_rocks: list[dict] = []
    for name in unique_names:
        key = name.lower()
        if key in FAST_MANA_CARDS:
            continue  # already handled as fast mana
        data = card_db.get(key, {})
        if _is_mana_rock(data):
            prod = _parse_rock_production(data.get("oracle_text", ""))
            if prod > 0:
                mana_rocks.append({
                    "name": name,
                    "cmc": data.get("mana_value", 0),
                    "production": prod,
                })

    land_count = len(lands)
    print(
        f"  [Mana Engine] {land_count} lands | "
        f"{len(fast_mana_cards)} fast mana | "
        f"{len(mana_rocks)} mana rocks detected"
    )

    # ---- land model ------------------------------------------------------
    # Assume opener is kept if it has ~3 lands (conservative keep threshold).
    # After those 3 opener lands are used up, further lands enter at the deck's
    # natural draw rate.
    opener_lands = min(3, land_count)
    remaining_lands = max(0, land_count - opener_lands)
    # Rate at which we draw additional lands after the opener
    post_opener_land_rate = remaining_lands / (DECK_SIZE - OPENER)

    # ---- build profile ---------------------------------------------------
    profile: dict[str, float] = {}

    for turn in range(1, 6):
        # 1. Lands in play
        # Turn 1: play 1 opener land (not all 3 at once — 1 land drop per turn)
        # Each subsequent turn: play 1 more, either from opener or from draw
        draws_after_opener = turn - 1
        lands_in_play = min(float(turn),
                            opener_lands + draws_after_opener * post_opener_land_rate)
        land_mana = lands_in_play

        # 2. Fast mana bonus
        # These are 0-cost or near-0-cost cards that accelerate T1 mana.
        # Each copy has probability OPENER/DECK_SIZE of being in the opening hand.
        fast_bonus = 0.0
        for card in fast_mana_cards:
            net = FAST_MANA_CARDS[card.lower()]
            p_in_opener = OPENER / DECK_SIZE
            fast_bonus += net * p_in_opener

        # 3. Mana rock contribution
        # A rock with CMC=C can be played when ≥C mana is available (turn C at
        # the earliest for CMC ≥ 1; turn 1 for CMC 0).
        # Being an artifact, it can tap the same turn it enters (no summoning
        # sickness), contributing its production from turn max(1, C) onward.
        # We use a conservative model: the rock must be seen before or during
        # the turn it's played.  P(seen by turn T) ≈ (OPENER + T) / DECK_SIZE.
        rock_mana = 0.0
        for rock in mana_rocks:
            cmc = int(rock["cmc"])
            prod = rock["production"]
            play_turn = max(1, cmc)   # earliest turn it can be cast
            if turn >= play_turn:
                cards_seen = OPENER + turn  # opener + draws up to this turn
                p = min(1.0, cards_seen / DECK_SIZE)
                rock_mana += prod * p

        profile[f"turn_{turn}"] = round(land_mana + fast_bonus + rock_mana, 2)

    return profile


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = [
        "Command Tower", "Underground Sea", "Polluted Delta", "Flooded Strand",
        "Watery Grave", "Marsh Flats", "Swamp", "Island",
        "Sol Ring", "Arcane Signet", "Dark Ritual",
        "Thassa's Oracle", "Demonic Consultation", "Brainstorm", "Force of Will",
    ] + ["Swamp"] * 12 + ["Island"] * 10

    profile = build_mana_profile(sample)
    print("\nMana Profile:")
    for t, m in profile.items():
        print(f"  {t}: {m:.2f} mana")
