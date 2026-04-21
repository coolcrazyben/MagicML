"""
optimizer.py
Stage 3 — Variant Optimizer

Tests card substitutions in a Commander deck to find swaps that make
combos assemble faster. Uses the Scryfall API to find replacement candidates.

Usage
-----
  Instantiate DeckOptimizer and call .run() — see main.py for integration.
"""

import contextlib
import io
import json
import os
import re
import requests
import sqlite3
import time
import urllib.parse
import urllib.request
from datetime import datetime

from combo_detector import MatchedCombo, NearCombo
from mana_engine import FAST_MANA_CARDS
from speed_calculator import ComboSpeed, calculate_combo_speeds

DB_PATH = "cards.db"
CACHE_DIR = "data/candidates"
REPORT_PATH = "data/optimization_report.json"
SCRYFALL_SEARCH = "https://api.scryfall.com/cards/search"
RATE_LIMIT_DELAY = 0.12       # seconds between Scryfall requests (~8 req/sec)
MAX_CANDIDATES_PER_CARD = 10  # quality over quantity

# Cards that are always LOCKED regardless of whether they are combo pieces.
LOCKED_FAST_MANA: frozenset[str] = frozenset({
    "mana crypt",
    "mana vault",
    "sol ring",
    "chrome mox",
    "mox diamond",
    "lion's eye diamond",
    "dark ritual",
    "cabal ritual",
    "lotus petal",
})

# Spell types we recognise for type-matching queries.
SPELL_TYPES: frozenset[str] = frozenset(
    ["instant", "sorcery", "creature", "artifact", "enchantment", "planeswalker"]
)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _fetch_card_info(name: str, db_path: str) -> dict | None:
    """Return a single card's data from cards.db, or None if not found."""
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT name, mana_cost, mana_value, type_line, oracle_text, color_identity "
            "FROM cards WHERE lower(name) = lower(?)",
            (name,),
        ).fetchone()
        conn.close()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    # color_identity stored as JSON array string e.g. '["B"]' or as plain text
    raw_ci = row[5] or "[]"
    try:
        color_identity = json.loads(raw_ci)
    except (json.JSONDecodeError, TypeError):
        color_identity = [c.strip() for c in raw_ci.strip("[]").split(",") if c.strip()]
    return {
        "name":           row[0],
        "mana_cost":      row[1] or "",
        "mana_value":     float(row[2] or 0),
        "type_line":      row[3] or "",
        "oracle_text":    row[4] or "",
        "color_identity": color_identity,
    }


# ---------------------------------------------------------------------------
# Color identity helpers
# ---------------------------------------------------------------------------

def is_color_legal(card_data: dict, deck_color_identity: set[str]) -> bool:
    """Return True if every color in the card's color_identity is in the deck's."""
    card_colors = card_data.get("color_identity", [])
    for color in card_colors:
        if color not in deck_color_identity:
            return False
    return True


def _fetch_missing_card_color_identity(
    name: str,
    db_path: str,
) -> list[str]:
    """
    Return the color_identity list for a card, using cards.db first,
    then falling back to a Scryfall named lookup.
    Returns [] (colorless) if the card cannot be found.
    """
    info = _fetch_card_info(name, db_path)
    if info is not None:
        return info.get("color_identity", [])

    # Fallback: fetch from Scryfall /cards/named
    cache_dir = "data/cards"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{_sanitize_filename(name)}.json")

    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("color_identity", [])

    try:
        params = urllib.parse.urlencode({"fuzzy": name})
        url = f"https://api.scryfall.com/cards/named?{params}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MTGDeckOptimizer/1.0 (educational project)"},
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        time.sleep(RATE_LIMIT_DELAY)
        return data.get("color_identity", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Card classification
# ---------------------------------------------------------------------------

def _is_land(type_line: str) -> bool:
    return "Land" in type_line


def _card_types(type_line: str) -> list[str]:
    """Return the lowercase spell types present in a type line."""
    tl = type_line.lower()
    return [t for t in SPELL_TYPES if t in tl]


def categorize_deck(
    decklist:      list[str],
    commanders:    list[str],
    matched_combos: list[MatchedCombo],
    db_path:       str = DB_PATH,
) -> tuple[set[str], list[str]]:
    """
    Split non-land cards into LOCKED and SWAPPABLE sets.

    LOCKED  — combo pieces, the commander(s), and hardcoded fast mana.
    SWAPPABLE — every other non-land card with at least one recognisable type.

    Returns
    -------
    (locked_names_lower, swappable_names)
        locked_names_lower : set of lowercased card names
        swappable_names    : list of canonical card names (unique, order preserved)
    """
    combo_pieces = {c.lower() for combo in matched_combos for c in combo.cards_required}
    commander_set = {c.lower() for c in commanders}
    locked = combo_pieces | commander_set | LOCKED_FAST_MANA

    unique_names = list(dict.fromkeys(decklist))
    swappable: list[str] = []

    for name in unique_names:
        if name.lower() in locked:
            continue
        info = _fetch_card_info(name, db_path)
        if info is None:
            continue
        if _is_land(info["type_line"]):
            continue
        if not _card_types(info["type_line"]):
            continue  # unrecognised type (e.g. Dungeon)
        swappable.append(name)

    return locked, swappable


# ---------------------------------------------------------------------------
# Scryfall API helpers
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    """Convert a card name to a safe filename stem."""
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_").lower()


def _fetch_scryfall_candidates(
    card_name:      str,
    card_info:      dict,
    color_identity: set[str],
) -> list[dict]:
    """
    Query Scryfall for cards that could replace *card_name*.

    Results are cached to CACHE_DIR so repeat runs are instant.
    Respects Scryfall rate limits via RATE_LIMIT_DELAY.

    Returns a list of candidate dicts with keys:
      name, type_line, mana_value, mana_cost, color_identity
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{_sanitize_filename(card_name)}.json")

    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    # Extract primary type
    type_line = card_info.get("type_line", "")
    primary_type = None
    for t in ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Planeswalker"]:
        if t in type_line:
            primary_type = t.lower()
            break
    if primary_type is None:
        _cache_empty(cache_file)
        return []

    # Build color identity string for Scryfall (uppercase, e.g. "B" or "WUB")
    color_str = "".join(sorted(color_identity, key=lambda c: "WUBRG".index(c) if c in "WUBRG" else 99))
    if not color_str:
        color_str = "C"

    cmc = int(card_info.get("mana_value", 0))
    cmc_min = max(0, cmc - 1)
    cmc_max = cmc + 1

    query = f"type:{primary_type} cmc>={cmc_min} cmc<={cmc_max} id<={color_str} legal:commander"

    try:
        time.sleep(RATE_LIMIT_DELAY)
        response = requests.get(
            SCRYFALL_SEARCH,
            params={"q": query, "order": "edhrec", "page": 1},
            headers={"User-Agent": "MTGDeckOptimizer/1.0 (educational project)"},
            timeout=12,
        )

        if response.status_code == 400:
            print(f"\n    [Debug] Query: {query}")
            print(f"\n    [Debug] Response: {response.text[:300]}")
            _cache_empty(cache_file)
            return []

        response.raise_for_status()
        data = response.json()

        candidates: list[dict] = []
        for card in data.get("data", []):
            name = card.get("name", "")
            if name.lower() == card_name.lower():
                continue
            cand = {
                "name":           name,
                "type_line":      card.get("type_line", ""),
                "mana_value":     float(card.get("cmc", 0)),
                "mana_cost":      card.get("mana_cost", ""),
                "color_identity": card.get("color_identity", []),
            }
            # Secondary color-identity guard: filter explicitly after fetch.
            if not is_color_legal(cand, color_identity):
                continue
            candidates.append(cand)
            if len(candidates) >= MAX_CANDIDATES_PER_CARD:
                break

        _write_cache(cache_file, candidates)
        return candidates

    except Exception as exc:
        print(f"\n    [Scryfall] Warning: could not fetch candidates for '{card_name}': {exc}")
        _cache_empty(cache_file)
        return []


def _cache_empty(path: str) -> None:
    _write_cache(path, [])


def _write_cache(path: str, data: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Swap simulation helpers
# ---------------------------------------------------------------------------

def _run_speed_silently(
    decklist:       list[str],
    matched_combos: list[MatchedCombo],
    db_path:        str,
) -> list[ComboSpeed] | None:
    """Run calculate_combo_speeds with all stdout suppressed."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return calculate_combo_speeds(decklist, matched_combos, db_path)
    except Exception:
        return None


def _make_swapped_deck(decklist: list[str], remove: str, add: str) -> list[str]:
    """Return a copy of *decklist* with the first occurrence of *remove* replaced by *add*."""
    new_deck = list(decklist)
    for i, card in enumerate(new_deck):
        if card.lower() == remove.lower():
            new_deck[i] = add
            return new_deck
    return new_deck  # remove_card not found (shouldn't happen)


def _avg_turn(speeds: list[ComboSpeed]) -> float:
    """Mean of estimated_average_turn across all combos. Returns 999 if empty."""
    if not speeds:
        return 999.0
    return sum(cs.estimated_average_turn for cs in speeds) / len(speeds)


def _combos_improved(
    baseline: list[ComboSpeed],
    new_speeds: list[ComboSpeed],
) -> list[str]:
    """
    Return the IDs of combos that got faster after the swap.
    Also returns their human-readable card list for display.
    """
    new_by_id = {cs.combo_id: cs.estimated_average_turn for cs in new_speeds}
    improved: list[str] = []
    for cs in baseline:
        new_avg = new_by_id.get(cs.combo_id)
        if new_avg is not None and new_avg < cs.estimated_average_turn:
            improved.append(", ".join(cs.cards_required))
    return improved


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------

class DeckOptimizer:
    """
    Tests card substitutions to find swaps that improve combo speed.

    Parameters
    ----------
    decklist       : Full decklist with duplicates (same format as parse_decklist output).
    commanders     : List of commander names.
    matched_combos : Confirmed combos from combo_detector.find_combos().
    color_identity : The deck's combined color identity (set of WUBRG letters).
    db_path        : Path to cards.db.
    """

    def __init__(
        self,
        decklist:       list[str],
        commanders:     list[str],
        matched_combos: list[MatchedCombo],
        color_identity: set[str],
        db_path:        str = DB_PATH,
        near_combos:    list[NearCombo] | None = None,
    ) -> None:
        self.decklist       = decklist
        self.commanders     = commanders
        self.matched_combos = matched_combos
        self.color_identity = color_identity
        self.db_path        = db_path
        self.near_combos    = near_combos or []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the full optimization pipeline and return a report dict.

        Pipeline
        --------
        1. Categorize cards as LOCKED or SWAPPABLE.
        2. Compute baseline combo speeds.
        3. Fetch Scryfall candidates for each swappable card (cached).
        4. Test every swap; score = baseline_avg_turn - new_avg_turn.
        5. Rank and print the top 10 improvements.
        6. Save the full report to REPORT_PATH.
        """
        _sep = "=" * 72
        print(f"\n{_sep}")
        print("  DECK OPTIMIZER")
        print(_sep)

        # No confirmed combos — use near-combo completion scoring instead.
        if not self.matched_combos:
            return self._run_near_combo_mode()

        # ---- Step 1: Categorize ----------------------------------------
        print("\n[1/5] Categorizing cards...")
        locked, swappable = categorize_deck(
            self.decklist, self.commanders, self.matched_combos, self.db_path
        )
        print(f"      {len(locked)} LOCKED  (combo pieces, commander, fast mana)")
        print(f"      {len(swappable)} SWAPPABLE cards will be tested")

        if not swappable:
            print("\n  No swappable cards found — all non-land cards are locked.")
            return {}

        # ---- Step 2: Baseline ------------------------------------------
        print("\n[2/5] Computing baseline combo speeds...")
        baseline_speeds = _run_speed_silently(
            self.decklist, self.matched_combos, self.db_path
        )
        if not baseline_speeds:
            print("      ERROR: Could not compute baseline speeds.")
            return {}

        baseline_avg = _avg_turn(baseline_speeds)
        baseline_by_id = {cs.combo_id: cs.estimated_average_turn for cs in baseline_speeds}
        print(f"      Baseline average turn across all combos: {baseline_avg:.2f}")

        # ---- Step 3: Fetch candidates ----------------------------------
        print(f"\n[3/5] Fetching Scryfall candidates for {len(swappable)} swappable cards...")
        print("      (Cached results load instantly; new lookups respect rate limits.)")

        swap_pairs: list[tuple[str, dict]] = []  # (remove_card, candidate_dict)
        for idx, card_name in enumerate(swappable, 1):
            card_info = _fetch_card_info(card_name, self.db_path)
            if card_info is None:
                continue
            candidates = _fetch_scryfall_candidates(
                card_name, card_info, self.color_identity
            )
            for cand in candidates:
                swap_pairs.append((card_name, cand))
            print(
                f"      [{idx}/{len(swappable)}] {card_name}: "
                f"{len(candidates)} candidate(s) found",
                flush=True,
            )

        total_swaps = len(swap_pairs)
        print(f"\n      Total swaps to test: {total_swaps}")

        # ---- Step 4: Test swaps ----------------------------------------
        print(f"\n[4/5] Testing {total_swaps} swap(s)...")
        all_swaps: list[dict] = []

        for swap_num, (remove_card, candidate) in enumerate(swap_pairs, 1):
            add_card = candidate["name"]
            print(
                f"  Testing swap {swap_num}/{total_swaps}: "
                f"{remove_card} -> {add_card}...",
                end="",
                flush=True,
            )

            new_deck   = _make_swapped_deck(self.decklist, remove_card, add_card)
            new_speeds = _run_speed_silently(new_deck, self.matched_combos, self.db_path)

            if new_speeds is None:
                print(" [skip — simulation error]")
                continue

            new_avg = _avg_turn(new_speeds)
            score   = round(baseline_avg - new_avg, 3)
            improved_combos = (
                _combos_improved(baseline_speeds, new_speeds) if score > 0 else []
            )

            print(f" score={score:+.3f}")

            remove_info = _fetch_card_info(remove_card, self.db_path)
            all_swaps.append({
                "remove":           remove_card,
                "add":              add_card,
                "score":            score,
                "baseline_avg":     baseline_avg,
                "new_avg":          new_avg,
                "candidate_type":   candidate.get("type_line", ""),
                "candidate_cmc":    candidate.get("mana_value", 0),
                "original_cmc":     remove_info.get("mana_value", 0) if remove_info else 0,
                "improved_combos":  improved_combos,
            })

        # ---- Step 5: Rank & report -------------------------------------
        print("\n[5/5] Ranking results...")
        positive = [s for s in all_swaps if s["score"] > 0]
        positive.sort(key=lambda x: x["score"], reverse=True)
        top_10 = positive[:10]

        self._print_top_swaps(top_10, baseline_speeds)

        # ---- Save report -----------------------------------------------
        os.makedirs("data", exist_ok=True)
        report = {
            "baseline_speeds":   baseline_by_id,
            "baseline_avg_turn": baseline_avg,
            "top_swaps":         top_10,
            "all_swaps_tested":  all_swaps,
            "timestamp":         datetime.now().isoformat(),
        }
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to: {REPORT_PATH}")

        return report

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _print_top_swaps(
        self,
        top_swaps:      list[dict],
        baseline_speeds: list[ComboSpeed],
    ) -> None:
        sep = "-" * 72
        print()
        print("=" * 72)
        print("  TOP 10 RECOMMENDED SWAPS")
        print("=" * 72)

        if not top_swaps:
            print()
            print("  No speed improvements found in the candidate pool.")
            print("  Your deck is already highly optimized for the detected combos,")
            print("  or the Scryfall candidate search returned limited results.")
            print()
            return

        for rank, swap in enumerate(top_swaps, 1):
            remove  = swap["remove"]
            add     = swap["add"]
            score   = swap["score"]
            orig_cmc = int(swap.get("original_cmc", 0))
            cand_cmc = int(swap.get("candidate_cmc", 0))
            improved = swap.get("improved_combos", [])

            # Choose a data-driven reason
            if cand_cmc < orig_cmc:
                reason = (
                    f"Lower CMC ({cand_cmc} vs {orig_cmc}) reduces mana needed "
                    f"before combos can fire"
                )
            elif cand_cmc > orig_cmc:
                reason = (
                    f"Despite higher CMC ({cand_cmc} vs {orig_cmc}), enables "
                    f"faster tutor or mana acceleration paths"
                )
            else:
                reason = (
                    f"Same CMC ({cand_cmc}) with improved tutor coverage or "
                    f"mana-efficiency for combo assembly"
                )

            combo_display = improved[0] if improved else "(all combos)"

            print()
            print(f"  [{rank}] REMOVE: {remove}  ->  ADD: {add}")
            print(f"      Speed improvement : {score:+.2f} turns faster on average")
            print(f"      Affects combo     : {combo_display}")
            print(f"      Reason            : {reason}")

        print()
        print("=" * 72)
        print()

    # ------------------------------------------------------------------
    # Near-combo completion mode (no confirmed combos in the deck)
    # ------------------------------------------------------------------

    def _run_near_combo_mode(self) -> dict:
        """
        When there are no confirmed combos, rank candidate cards by how many
        near-combos they would complete if added to the deck.

        Score = number of near-combos where the candidate is the missing card.
        Does not require Scryfall — the missing cards are already known from
        the near-combo analysis.
        """
        print("\n  No confirmed combos detected.")
        print("  Switching to near-combo completion mode.")
        print("  Scoring candidates by how many near-combos they would complete.\n")

        if not self.near_combos:
            print("  No near-combos found either — nothing to optimise.")
            return {}

        # Tally how many near-combos each missing card would complete,
        # respecting the deck's color identity.
        counts: dict[str, int] = {}        # canonical_name -> count
        combos_for: dict[str, list[str]] = {}  # canonical_name -> [combo descriptions]
        illegal_cards: set[str] = set()    # cards skipped due to color identity

        for nc in self.near_combos:
            name = nc.missing_card
            key  = name.lower()

            # Check color identity before counting
            card_ci = _fetch_missing_card_color_identity(name, self.db_path)
            if not is_color_legal({"color_identity": card_ci}, self.color_identity):
                illegal_cards.add(name)
                continue

            counts[key]     = counts.get(key, 0) + 1
            combo_desc      = ", ".join(
                c for c in nc.cards_required if c.lower() != key
            )
            combos_for.setdefault(key, []).append(combo_desc)

        # Build ranked list (canonical name from the first near-combo seen)
        seen_keys: dict[str, str] = {}  # lower -> canonical
        for nc in self.near_combos:
            key = nc.missing_card.lower()
            if key not in seen_keys:
                seen_keys[key] = nc.missing_card

        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top_5 = ranked[:5]

        # Collect colorless near-combo cards for the fallback message
        colorless_options: list[str] = [
            seen_keys[k] for k in counts if not counts.get(k, 0) == 0
            and not any(c for c in _fetch_missing_card_color_identity(seen_keys[k], self.db_path))
        ]

        self._print_near_combo_adds(
            top_5, seen_keys, combos_for,
            deck_color_identity=self.color_identity,
            illegal_cards=illegal_cards,
            colorless_options=colorless_options,
        )

        report = {
            "mode":      "near_combo_completion",
            "top_adds":  [
                {
                    "add":              seen_keys[k],
                    "score":            v,
                    "completes_combos": combos_for.get(k, []),
                }
                for k, v in top_5
            ],
            "illegal_cards_filtered": sorted(illegal_cards),
            "timestamp": datetime.now().isoformat(),
        }
        os.makedirs("data", exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Full report saved to: {REPORT_PATH}")
        return report

    def _print_near_combo_adds(
        self,
        top_5:              list[tuple[str, int]],
        seen_keys:          dict[str, str],
        combos_for:         dict[str, list[str]],
        deck_color_identity: set[str] | None = None,
        illegal_cards:      set[str] | None = None,
        colorless_options:  list[str] | None = None,
    ) -> None:
        print("=" * 72)
        print("  TOP 5 RECOMMENDED ADDS  (near-combo completion)")
        print("=" * 72)
        print()

        if not top_5:
            ci_str = "".join(sorted(deck_color_identity or [], key=lambda c: "WUBRG".index(c) if c in "WUBRG" else 99)) or "C"
            print(f"  No legal near-combo completions found within color identity {{{ci_str}}}")
            if colorless_options:
                print(f"  Consider these colorless combo enablers instead: {colorless_options}")
            print()
            print("=" * 72)
            print()
            return

        for rank, (key, score) in enumerate(top_5, 1):
            canonical = seen_keys[key]
            combo_list = combos_for.get(key, [])
            combos_str = "; ".join(combo_list[:3])  # show up to 3 combos
            if len(combo_list) > 3:
                combos_str += f" (+{len(combo_list) - 3} more)"
            noun = "combo" if score == 1 else "combos"
            print(f"  [{rank}] ADD: {canonical}")
            print(f"      Completes  : {score} near-{noun}")
            print(f"      With cards : {combos_str}")
            print()

        if illegal_cards:
            print(f"  Note: {len(illegal_cards)} off-color card(s) excluded from recommendations.")

        print("=" * 72)
        print()
