"""
build_card_db.py
Downloads the Scryfall 'oracle_cards' bulk data and stores Commander-legal
cards in a local SQLite database (cards.db).
"""

import json
import os
import sqlite3
import sys
import time

import requests

DB_PATH = "cards.db"
BULK_DATA_URL = "https://api.scryfall.com/bulk-data"


def get_oracle_cards_download_url():
    """Fetch the Scryfall bulk-data index and return the oracle_cards download URL."""
    print("Fetching Scryfall bulk data index...")
    resp = requests.get(BULK_DATA_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    for entry in data["data"]:
        if entry["type"] == "oracle_cards":
            url = entry["download_uri"]
            size_mb = entry.get("size", 0) / (1024 * 1024)
            print(f"Found oracle_cards dataset ({size_mb:.1f} MB): {url}")
            return url

    raise RuntimeError("Could not find oracle_cards entry in Scryfall bulk data.")


def download_json(url, local_path="oracle_cards.json"):
    """Stream-download a large JSON file with progress reporting."""
    if os.path.exists(local_path):
        print(f"Using cached file: {local_path}")
        return local_path

    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 256  # 256 KB

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}% ({downloaded/1024/1024:.1f} MB)", end="", flush=True)

    print()  # newline after progress
    print(f"Download complete: {local_path}")
    return local_path


def create_tables(conn):
    """Create the cards table (drop and recreate for a clean build)."""
    conn.execute("DROP TABLE IF EXISTS cards")
    conn.execute("""
        CREATE TABLE cards (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            mana_cost     TEXT,
            mana_value    REAL,
            type_line     TEXT,
            oracle_text   TEXT,
            color_identity TEXT,
            keywords      TEXT
        )
    """)
    conn.commit()
    print("Created cards table.")


def is_commander_legal(card):
    """Return True if the card is legal in Commander format."""
    legalities = card.get("legalities", {})
    return legalities.get("commander") == "legal"


def extract_card_fields(card):
    """Extract the fields we want to store from a Scryfall card object."""
    name = card.get("name", "")
    mana_cost = card.get("mana_cost", "") or ""
    mana_value = card.get("cmc", 0.0)
    type_line = card.get("type_line", "") or ""
    oracle_text = card.get("oracle_text", "") or ""

    # color_identity is a list like ["W", "U"] — store as comma-separated string
    color_identity = ",".join(card.get("color_identity", []))

    # keywords is a list of strings
    keywords = ",".join(card.get("keywords", []))

    return (
        card.get("oracle_id") or card.get("id"),  # use oracle_id as stable PK
        name,
        mana_cost,
        mana_value,
        type_line,
        oracle_text,
        color_identity,
        keywords,
    )


def build_database(json_path):
    """Parse the oracle_cards JSON and insert Commander-legal cards into SQLite."""
    print(f"Loading JSON from {json_path} (this may take a moment)...")
    with open(json_path, encoding="utf-8") as f:
        cards = json.load(f)

    total = len(cards)
    print(f"Total cards in dataset: {total:,}")

    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    rows = []
    skipped = 0
    inserted = 0

    for i, card in enumerate(cards):
        if i % 5000 == 0:
            print(f"  Processing card {i:,}/{total:,}...", end="\r", flush=True)

        if not is_commander_legal(card):
            skipped += 1
            continue

        rows.append(extract_card_fields(card))
        inserted += 1

        # Batch insert every 1000 rows for efficiency
        if len(rows) >= 1000:
            conn.executemany(
                "INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?)", rows
            )
            conn.commit()
            rows.clear()

    # Insert any remaining rows
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?)", rows
        )
        conn.commit()

    print()  # newline after \r progress
    print(f"Done! Inserted {inserted:,} Commander-legal cards. Skipped {skipped:,} non-Commander cards.")
    conn.close()


def main():
    start = time.time()

    # Step 1: Get the download URL from Scryfall
    url = get_oracle_cards_download_url()

    # Step 2: Download (or use cached copy)
    json_path = download_json(url)

    # Step 3: Build the SQLite database
    build_database(json_path)

    elapsed = time.time() - start
    print(f"\nDatabase written to: {DB_PATH}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
