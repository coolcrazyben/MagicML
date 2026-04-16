"""
build_combo_db.py
Downloads all known Commander combos from Commander Spellbook and stores
them in cards.db (same SQLite database as the card data).
"""

import sqlite3
import time

import requests

DB_PATH = "cards.db"
SPELLBOOK_API = "https://backend.commanderspellbook.com/variants/?format=json"


def create_combo_table(conn):
    """Create the combos table (drop and recreate for a clean build)."""
    conn.execute("DROP TABLE IF EXISTS combos")
    conn.execute("""
        CREATE TABLE combos (
            combo_id       TEXT PRIMARY KEY,
            cards_required TEXT NOT NULL,   -- comma-separated card names
            num_cards      INTEGER NOT NULL,
            steps          TEXT,            -- newline-separated step descriptions
            result         TEXT
        )
    """)
    conn.commit()
    print("Created combos table.")


def fetch_all_combos():
    """
    Paginate through the Commander Spellbook API and return a list of all variant objects.
    The API returns a paginated response with 'next' links.
    """
    combos = []
    url = SPELLBOOK_API
    page = 1

    while url:
        print(f"  Fetching page {page}...", end="\r", flush=True)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        combos.extend(results)

        url = data.get("next")  # None when we've hit the last page
        page += 1

    print()  # newline
    return combos


def parse_combo(variant):
    """
    Extract the fields we want from a Commander Spellbook variant object.

    The variant structure (simplified):
      {
        "id": "abc123",
        "uses": [
          {"card": {"name": "Card A"}},
          {"card": {"name": "Card B"}},
          ...
        ],
        "description": "Step 1: ...\nStep 2: ...",
        "results": [
          {"feature": {"name": "Infinite mana"}},
          ...
        ]
      }
    """
    combo_id = str(variant.get("id", ""))

    # Collect card names from the 'uses' list
    card_names = []
    for use in variant.get("uses", []):
        card_obj = use.get("card") or {}
        name = card_obj.get("name", "").strip()
        if name:
            card_names.append(name)

    if not card_names:
        return None  # skip malformed entries

    cards_required = ",".join(card_names)
    num_cards = len(card_names)

    # Steps are stored in the 'description' field as a block of text
    steps = variant.get("description", "") or ""

    # Results are a list of feature objects under 'produces'; join their names for a summary
    result_names = []
    for result in variant.get("produces", []):
        feature = result.get("feature") or {}
        feat_name = feature.get("name", "").strip()
        if feat_name:
            result_names.append(feat_name)
    result = "; ".join(result_names)

    return (combo_id, cards_required, num_cards, steps, result)


def build_combo_database():
    """Fetch all combos and insert them into the SQLite database."""
    print(f"Fetching combos from Commander Spellbook API...")
    start = time.time()

    raw_combos = fetch_all_combos()
    print(f"Retrieved {len(raw_combos):,} raw combo variants from API.")

    conn = sqlite3.connect(DB_PATH)
    create_combo_table(conn)

    inserted = 0
    skipped = 0
    rows = []

    for variant in raw_combos:
        parsed = parse_combo(variant)
        if parsed is None:
            skipped += 1
            continue
        rows.append(parsed)
        inserted += 1

    conn.executemany(
        "INSERT OR REPLACE INTO combos VALUES (?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()

    elapsed = time.time() - start
    print(f"\nInserted {inserted:,} combos into {DB_PATH}.")
    if skipped:
        print(f"Skipped {skipped:,} malformed/empty entries.")
    print(f"Total time: {elapsed:.1f}s")


def main():
    build_combo_database()


if __name__ == "__main__":
    main()
