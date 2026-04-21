"""
deck_parser.py
Parses a decklist in standard MTGO / Moxfield / Archidekt text format and
returns a clean list of card names plus the identified commander.

Supported input formats
-----------------------
  MTGO / Moxfield (// prefix):
    // Commander
    1 Atraxa, Praetors' Voice

  Moxfield export (Word (N) header, no //):
    Commander (1)
    1 Atraxa, Praetors' Voice
    Deck (99)
    1 Sol Ring

  Archidekt / generic:
    Commander:
    1 Atraxa, Praetors' Voice

  Inline commander label (partners supported):
    Commander: Thrasios, Triton Hero
    Commander: Tymna the Weaver

  No-header format (commander alone at the bottom after a blank line):
    1 Sol Ring
    1 Command Tower
    ... (99 cards)

    1 Tinybones, Bauble Burglar

  Set codes and collector numbers are stripped from card names:
    1 Birds of Paradise [M10] #27   -> Birds of Paradise
    1 Lightning Bolt (2XM) 113      -> Lightning Bolt
"""

import re


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Noise appended after the card name:
#   [SET] or (SET)  — set code in brackets/parens
#   #NNN or bare NNN — collector number
_NOISE_PATTERN = re.compile(
    r"""
    \s*
    (?:[\(\[]\w[\w\s\-]*[\)\]]   # (SET) or [SET]  — allow spaces/hyphens inside
       \s*
    )?
    (?:\#?\s*\d+)?               # optional collector number
    \s*$
    """,
    re.VERBOSE,
)

# "QUANTITY CARD_NAME ..." — quantity is mandatory digits then whitespace
_QUANTITY_PATTERN = re.compile(r"^(\d+)\s+(.+)$")

# Inline commander label: "Commander: Card Name"
# Matches lines where the card name follows the colon on the same line.
# Distinct from the section-header form "Commander:" (nothing after colon).
_INLINE_COMMANDER = re.compile(r"^commander:\s+(.+)$", re.IGNORECASE)

# Moxfield-style section header: one or more words followed by (N)
# e.g. "Commander (1)", "Deck (99)", "Sideboard (0)", "Companion (1)"
_MOXFIELD_HEADER = re.compile(r"^[A-Za-z][\w\s]*\(\d+\)\s*$")

# Traditional section headers:
#   // anything
#   # anything
#   Anything: (with trailing colon)
_TRAD_HEADER = re.compile(
    r"""
    ^(?:
        //.*                  # // Commander, // Lands, etc.
        | \#.*                # # inline comments
        | [A-Za-z][\w\s]*:$  # "Sideboard:", "Commander:", etc.
    )$
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_section_header(line: str) -> bool:
    """Return True if the line is a section header or blank (should be skipped)."""
    stripped = line.strip()
    if not stripped:
        return True
    return bool(_TRAD_HEADER.match(stripped) or _MOXFIELD_HEADER.match(stripped))


def _section_name(line: str) -> str:
    """
    Extract a normalised section name from a header line.
    Returns a lowercase string like 'commander', 'lands', 'sideboard', etc.
    """
    s = line.strip()
    s = re.sub(r"^//\s*", "", s)          # strip leading //
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)  # strip trailing (N)
    s = re.sub(r":\s*$", "", s)           # strip trailing :
    return s.strip().lower()


def _clean_card_name(raw_name: str) -> str:
    """
    Strip set codes, collector numbers, and extra whitespace from a raw
    card-name string (after the leading quantity has already been removed).

    Examples:
      "Sol Ring [M10] #27"       -> "Sol Ring"
      "Lightning Bolt (2XM) 113" -> "Lightning Bolt"
      "Birds of Paradise"        -> "Birds of Paradise"
    """
    name = _NOISE_PATTERN.sub("", raw_name)
    return " ".join(name.split())  # collapse internal whitespace


# ---------------------------------------------------------------------------
# Fallback: detect commander isolated at the bottom
# ---------------------------------------------------------------------------

def _detect_trailing_commander(text: str) -> str | None:
    """
    Handle the headerless format where the commander is the only card in the
    last blank-line-separated paragraph:

        1 Sol Ring
        1 Command Tower
        ... (99 cards)
                          <- blank line
        1 Tinybones, Bauble Burglar

    Splits the text into paragraphs on blank lines. If the last paragraph
    contains exactly one card line with quantity 1, return its name.
    Returns None if the pattern is not matched.
    """
    # Split on one or more blank lines
    paragraphs = [p.strip() for p in re.split(r"\n[ \t]*\n", text.strip()) if p.strip()]

    if len(paragraphs) < 2:
        return None  # only one block — no separator to act on

    last_para = paragraphs[-1]
    last_lines = [l.strip() for l in last_para.splitlines() if l.strip()]

    if len(last_lines) != 1:
        return None  # multiple cards in the last block — not a clear commander

    line = last_lines[0]
    match = _QUANTITY_PATTERN.match(line)
    if match:
        quantity = int(match.group(1))
        name = _clean_card_name(match.group(2))
    else:
        quantity = 1
        name = _clean_card_name(line)

    # Only treat it as the commander if quantity is exactly 1
    if quantity == 1 and name:
        return name

    return None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_decklist(text: str) -> tuple[list[str], list[str]]:
    """
    Parse a decklist string and return (cards, commanders).

    Parameters
    ----------
    text : str
        Full decklist text (multiline).

    Returns
    -------
    cards : list[str]
        Every card in the deck, one entry per copy (e.g. 4x Sol Ring → 4 entries).
        Includes the commander(s).
    commanders : list[str]
        Commander card names. Empty if no commander section was found.
        Two entries for partner pairs listed under the same section.
    """
    cards: list[str] = []
    commanders: list[str] = []
    current_section: str = ""

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            continue

        # ---- Inline commander label: "Commander: Card Name" ----
        # Must be checked BEFORE the generic section-header test so that
        # "Commander: Thrasios, Triton Hero" is not treated as a blank header.
        inline_cmd = _INLINE_COMMANDER.match(stripped)
        if inline_cmd:
            name = _clean_card_name(inline_cmd.group(1))
            if name:
                cards.append(name)       # commander counts toward the 100-card deck
                commanders.append(name)
            continue

        # Detect section headers
        if _is_section_header(stripped):
            current_section = _section_name(stripped)
            continue

        # Parse the card line
        match = _QUANTITY_PATTERN.match(stripped)
        if match:
            quantity = int(match.group(1))
            name = _clean_card_name(match.group(2))
        else:
            # No leading quantity — treat as a single copy
            quantity = 1
            name = _clean_card_name(stripped)

        if not name:
            continue

        cards.extend([name] * quantity)

        # Record every card in the commander section (supports partner pairs)
        if current_section == "commander":
            commanders.append(name)

    # Fallback: if no explicit commander section was found, check whether the
    # deck uses the headerless "99 cards + blank line + 1 commander" format.
    if not commanders:
        trailing = _detect_trailing_commander(text)
        if trailing:
            commanders = [trailing]

    return cards, commanders


def unique_card_names(decklist: list[str]) -> set[str]:
    """Return the set of unique card names in the decklist."""
    return set(decklist)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    MTGO_SAMPLE = """
// Commander
1 Atraxa, Praetors' Voice

// Creatures
1 Birds of Paradise [M10] #27
1 Deepglow Skate (2XM) 113

// Artifacts
1 Sol Ring
4 Arcane Signet

// Lands
1 Command Tower

Sideboard:
1 Cyclonic Rift
"""

    MOXFIELD_SAMPLE = """
Commander (1)
1 Atraxa, Praetors' Voice

Deck (98)
1 Sol Ring
1 Command Tower
1 Birds of Paradise

Sideboard (0)
"""

    TRAILING_COMMANDER_SAMPLE = """
1 Sol Ring
1 Command Tower
1 Birds of Paradise
1 Arcane Signet

1 Tinybones, Bauble Burglar
"""

    for label, sample in [
        ("MTGO", MTGO_SAMPLE),
        ("Moxfield", MOXFIELD_SAMPLE),
        ("Trailing commander (no headers)", TRAILING_COMMANDER_SAMPLE),
    ]:
        cards, cmds = parse_decklist(sample)
        print(f"\n--- {label} format ---")
        print(f"Commander(s) : {', '.join(cmds) if cmds else '(none)'}")
        print(f"Total cards (with duplicates): {len(cards)}")
        print(f"Unique card names: {len(unique_card_names(cards))}")
        for card in cards:
            print(f"  {card}")
