"""
main.py
Entry point for the MTG Commander Combo Detector.

Usage
-----
  python main.py

Paste your decklist into the terminal (MTGO / Moxfield format).
Press Enter TWICE on a blank line when you're done.

The tool will:
  1. Parse the decklist and identify the commander.
  2. Look up the commander's color identity in cards.db.
  3. Find combos that are both present in the deck AND legal under that
     color identity.
  4. Print a formatted report.
"""

import os
import sqlite3
import sys
import textwrap
import traceback

from deck_parser import parse_decklist, unique_card_names
from combo_detector import find_combos, MatchedCombo, NearCombo
from speed_calculator import calculate_combo_speeds, ComboSpeed
from mana_engine import build_mana_profile
from tutor_engine import build_tutor_coverage
from optimizer import DeckOptimizer

DB_PATH = "cards.db"
WRAP_WIDTH = 72

# WUBRG color letter -> full name for pretty printing
COLOR_NAMES = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def read_multiline_input() -> str:
    """
    Read a pasted decklist from stdin.

    Stops when the user enters TWO consecutive blank lines, or on EOF
    (Ctrl+Z + Enter on Windows, Ctrl+D on Mac/Linux).

    A single blank line is kept as-is so section separators inside the
    decklist (e.g. between '// Commander' and '// Lands') are preserved.
    """
    print("Paste your decklist below.")
    print("Press Enter TWICE on a blank line when finished.\n")

    lines: list[str] = []
    consecutive_blanks = 0

    try:
        while True:
            line = input()
            if line.strip() == "":
                consecutive_blanks += 1
                if consecutive_blanks >= 2:
                    break           # two blank lines in a row → done
                lines.append("")   # keep the single blank line for the parser
            else:
                consecutive_blanks = 0
                lines.append(line)
    except EOFError:
        pass  # support: python main.py < decklist.txt

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commander colour identity lookup
# ---------------------------------------------------------------------------

def detect_commander_from_db(
    unique_cards: set[str], db_path: str
) -> str | None:
    """
    Fallback commander detection when no explicit section header was found.

    Checks (in order):
      1. Any card whose oracle text contains "can be your commander".
      2. If exactly one legendary creature is in the deck, treat it as
         the commander.
      3. If multiple legendary creatures exist, prompt the user to choose.
    """
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.OperationalError:
        return None

    candidates: list[str] = []

    for name in unique_cards:
        row = conn.execute(
            "SELECT name, type_line, oracle_text FROM cards WHERE lower(name) = lower(?)",
            (name,),
        ).fetchone()
        if row is None:
            continue
        _, type_line, oracle = row
        oracle_lower = (oracle or "").lower()
        type_lower = (type_line or "").lower()

        # 1. Eminence / partner / background commanders
        if "can be your commander" in oracle_lower:
            conn.close()
            return row[0]  # definitive match

        # Collect legendary creatures as potential commanders
        if "legendary" in type_lower and "creature" in type_lower:
            candidates.append(row[0])

    conn.close()

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        print(
            "\n  Multiple legendary creatures found. "
            "Please identify the commander:"
        )
        for i, name in enumerate(candidates, start=1):
            print(f"    {i}. {name}")
        print(f"    {len(candidates) + 1}. Skip (no commander filter)")
        try:
            choice = input("  Enter number: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except (ValueError, EOFError):
            pass

    return None


def lookup_commander_colors(commander_name: str, db_path: str) -> set[str] | None:
    """
    Look up the commander's color identity from cards.db.
    Returns a set of WUBRG letters, or None if the card isn't found.
    """
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT color_identity FROM cards WHERE lower(name) = lower(?)",
            (commander_name,),
        ).fetchone()
        conn.close()
    except sqlite3.OperationalError:
        return None

    if row is None:
        return None

    ci_str = row[0] or ""
    return {c for c in ci_str.split(",") if c}


def _has_partner(commander_name: str, db_path: str) -> bool:
    """Return True if the commander has the Partner keyword in oracle text."""
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT oracle_text FROM cards WHERE lower(name) = lower(?)",
            (commander_name,),
        ).fetchone()
        conn.close()
    except sqlite3.OperationalError:
        return False
    if row is None:
        return False
    return "partner" in (row[0] or "").lower()


def _get_legendary_creatures(
    unique_cards: set[str], db_path: str, exclude: str = ""
) -> list[str]:
    """Return names of all legendary creatures in the deck, excluding 'exclude'."""
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.OperationalError:
        return []
    candidates: list[str] = []
    for name in unique_cards:
        if exclude and name.lower() == exclude.lower():
            continue
        row = conn.execute(
            "SELECT name, type_line FROM cards WHERE lower(name) = lower(?)",
            (name,),
        ).fetchone()
        if row is None:
            continue
        type_lower = (row[1] or "").lower()
        if "legendary" in type_lower and "creature" in type_lower:
            candidates.append(row[0])
    conn.close()
    return sorted(candidates)


def format_color_identity(colors: set[str] | None) -> str:
    """Return a human-readable color identity string."""
    if colors is None:
        return "unknown"
    if not colors:
        return "Colorless"
    return ", ".join(COLOR_NAMES.get(c, c) for c in sorted(colors))


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _wrap(text: str, indent: str = "    ") -> str:
    if not text:
        return f"{indent}(none)"
    return textwrap.fill(
        text, width=WRAP_WIDTH, initial_indent=indent, subsequent_indent=indent
    )


def _sep(char: str = "-", width: int = WRAP_WIDTH) -> str:
    return char * width


def print_report(
    total_cards: int,
    unique_count: int,
    commander: str | None,
    commander_colors: set[str] | None,
    matched: list[MatchedCombo],
    near: list[NearCombo],
    speeds: list[ComboSpeed] | None = None,
) -> None:
    print()
    print(_sep("="))
    print("  MTG COMMANDER COMBO REPORT")
    print(_sep("="))
    print(f"  Total cards in deck  : {total_cards}")
    print(f"  Unique card names    : {unique_count}")
    print(f"  Commander            : {commander or '(not detected)'}")
    print(f"  Color identity       : {format_color_identity(commander_colors)}")
    print(_sep("="))

    # ---- Confirmed combos ----
    print(f"\n{'CONFIRMED COMBOS':^{WRAP_WIDTH}}")
    print(_sep())

    if not matched:
        print("  No complete combos detected in this decklist.")
    else:
        print(f"  Found {len(matched)} combo(s):\n")
        for idx, combo in enumerate(matched, start=1):
            print(f"  [{idx}] Combo ID: {combo.combo_id}")
            print(f"      Cards : {', '.join(combo.cards_required)}")
            print(f"      Result:")
            print(_wrap(combo.result))
            if combo.steps:
                print(f"      Steps:")
                for step_line in combo.steps.splitlines():
                    if step_line.strip():
                        print(_wrap(step_line.strip()))
            print()

    # ---- Near combos ----
    print(f"\n{'NEAR COMBOS  (1 card away)':^{WRAP_WIDTH}}")
    print(_sep())

    if not near:
        print("  No near-combos detected.")
    else:
        near_sorted = sorted(near, key=lambda n: len(n.cards_required), reverse=True)
        print(f"  Found {len(near_sorted)} near-combo(s):\n")
        for idx, combo in enumerate(near_sorted, start=1):
            print(f"  [{idx}] Combo ID: {combo.combo_id}")
            have = [c for c in combo.cards_required if c != combo.missing_card]
            print(f"      Cards in deck : {', '.join(have)}")
            print(f"      MISSING       : >>> {combo.missing_card} <<<")
            print(f"      Result:")
            print(_wrap(combo.result))
            print()

    # ---- Speed analysis ----
    if speeds:
        print(f"\n{'COMBO SPEED ANALYSIS':^{WRAP_WIDTH}}")
        print(_sep())
        for cs in speeds:
            # Use first card + result snippet as a label
            label = ", ".join(cs.cards_required)
            print(f"  [{cs.combo_id}] {label}")
            print(f"      Fastest theoretical turn : {cs.estimated_fastest_turn}")
            print(f"      Average expected turn    : {cs.estimated_average_turn}")
            print(f"      Speed rating             : {cs.speed_rating}")
            print(f"      Mana required            : {cs.mana_cost_display}")
            if any(cs.tutor_coverage.values()):
                print(f"      Tutors available         :")
                for piece, tutors in cs.tutor_coverage.items():
                    if tutors:
                        print(f"        - {piece} : {', '.join(tutors)}")
                    else:
                        print(f"        - {piece} : (no tutors in deck)")
            else:
                print(f"      Tutors available         : (none for any piece)")
            print()

    print(_sep("="))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database '{DB_PATH}' not found.")
        print("Run these first:")
        print("  python build_card_db.py")
        print("  python build_combo_db.py")
        sys.exit(1)

    # 1. Read decklist
    raw_text = read_multiline_input()
    if not raw_text.strip():
        print("No decklist provided. Exiting.")
        sys.exit(0)

    # 2. Parse — get card list AND commander(s)
    print("\nParsing decklist...")
    decklist, commanders = parse_decklist(raw_text)
    unique = unique_card_names(decklist)
    print(f"  {len(decklist)} total cards parsed ({len(unique)} unique names).")

    if not decklist:
        print("Could not parse any cards. Check the format and try again.")
        sys.exit(1)

    # 3. Resolve commanders and build combined color identity
    all_commanders: list[str] = list(commanders)

    if not all_commanders:
        # Fallback: DB-assisted detection when no commander section was found
        print("  No commander section found — attempting auto-detection...")
        detected = detect_commander_from_db(unique, DB_PATH)
        if detected:
            all_commanders = [detected]
            print(f"  Auto-detected commander: {detected}")
        else:
            print(
                "  Warning: Commander could not be determined — "
                "color identity filter disabled.\n"
                "  (Add '// Commander' or 'Commander (1)' before your commander's line.)"
            )

    # Look up and combine color identities for all commanders in the section
    commander_colors: set[str] = set()
    for cmd in all_commanders:
        colors = lookup_commander_colors(cmd, DB_PATH)
        if colors is not None:
            commander_colors |= colors
        else:
            print(f"  Warning: '{cmd}' not found in cards.db.")

    if all_commanders:
        ci_label = format_color_identity(commander_colors) if commander_colors else "unknown"
        print(f"  Commander(s): {' + '.join(all_commanders)} [{ci_label}]")

    # Partner check: if only 1 commander found and it has Partner, prompt for second
    if len(all_commanders) == 1 and _has_partner(all_commanders[0], DB_PATH):
        print(f"\n  '{all_commanders[0]}' has Partner — select your second commander:")
        cands = _get_legendary_creatures(unique, DB_PATH, exclude=all_commanders[0])
        if not cands:
            print("  No other legendary creatures found in deck.")
        else:
            for i, name in enumerate(cands, start=1):
                print(f"    {i}. {name}")
            print(f"    {len(cands) + 1}. Skip")
            try:
                choice = input("  Enter number: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(cands):
                    second = cands[idx]
                    second_colors = lookup_commander_colors(second, DB_PATH)
                    if second_colors is None:
                        print(f"  Warning: '{second}' not found in cards.db.")
                    else:
                        all_commanders.append(second)
                        commander_colors |= second_colors
                        print(
                            f"  Combined: {' + '.join(all_commanders)} "
                            f"[{format_color_identity(commander_colors)}]"
                        )
            except (ValueError, EOFError):
                pass

    # 4. Detect combos
    try:
        matched, near = find_combos(
            decklist, db_path=DB_PATH,
            commander_colors=commander_colors if commander_colors else None,
        )
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    # 5. Speed analysis for confirmed combos (or mana profile when none)
    speeds: list[ComboSpeed] = []
    if matched:
        print("\nRunning speed analysis...")
        try:
            speeds = calculate_combo_speeds(decklist, matched, db_path=DB_PATH)
        except Exception:
            print("\n[Speed Analysis ERROR — full traceback below]")
            traceback.print_exc()
            print("[Speed Analysis] Skipping speed analysis due to error above.\n")
    else:
        print("\nNo confirmed combos found — showing mana profile only")
        print("\nRunning mana analysis...")
        try:
            mana_profile = build_mana_profile(decklist, db_path=DB_PATH)
            print("\n  Mana Profile:")
            for key, mana in mana_profile.items():
                turn_num = key.split("_")[1]
                print(f"    Turn {turn_num}: {mana:.2f} mana available")
        except Exception:
            traceback.print_exc()

    # 6. Print report
    print_report(
        total_cards=len(decklist),
        unique_count=len(unique),
        commander=" + ".join(all_commanders) if all_commanders else None,
        commander_colors=commander_colors if commander_colors else None,
        matched=matched,
        near=near,
        speeds=speeds,
    )

    # 7. Optional: run deck optimizer (always offered)
    try:
        run_optimizer = input(
            "\nRun deck optimizer? This will test card substitutions. "
            "May take 2-3 minutes. (y/n): "
        ).strip().lower()
    except EOFError:
        run_optimizer = "n"

    if run_optimizer in ("y", "yes"):
        optimizer = DeckOptimizer(
            decklist=decklist,
            commanders=all_commanders,
            matched_combos=matched,
            near_combos=near,
            color_identity=commander_colors if commander_colors else set(),
            db_path=DB_PATH,
        )
        optimizer.run()
    else:
        print("Skipping optimizer.")

    # 8. Optional: ML-powered optimizer
    try:
        run_ml = input(
            "\nRun ML-powered optimizer? (uses trained model for instant predictions) (y/n): "
        ).strip().lower()
    except EOFError:
        run_ml = "n"

    if run_ml in ("y", "yes"):
        try:
            from ml_optimizer import get_ml_recommendations
            recs = get_ml_recommendations(
                decklist,
                matched,
                commander_colors if commander_colors else set(),
                commanders=all_commanders,
            )
            print("\n ML-POWERED RECOMMENDATIONS")
            print("=" * 72)
            if not recs:
                print("  No recommendations generated.")
            for i, rec in enumerate(recs, 1):
                print(f"  [{i}] REMOVE: {rec['remove']}  ->  ADD: {rec['add']}")
                print(f"      Predicted improvement : {rec['predicted_improvement']:+.3f} turns")
                print(f"      Confidence            : {rec['confidence']}")
                print(f"      Key reason            : {rec['top_feature']}")
            print("=" * 72)
        except FileNotFoundError as exc:
            print(f"\n  ML optimizer not ready: {exc}")
            print("  Run:  python ml_trainer.py")
        except Exception:
            traceback.print_exc()
    else:
        print("Skipping ML optimizer.")


if __name__ == "__main__":
    main()
