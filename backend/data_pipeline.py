"""
data_pipeline.py
Stage 5 - Automated Training Data Pipeline

Scrapes public Commander decklists from EDHREC and Moxfield, runs each
through the existing optimizer, and accumulates training rows overnight.

How the scrapers work
---------------------
* EDHRECScraper:
    1. Fetches popular commanders from Scryfall (sorted by EDHREC rank).
    2. Converts each name to an EDHREC slug and fetches the EDHREC
       commander page (individual pages are publicly accessible).
    3. Builds a synthetic "average deck" by picking the top cards from
       each category in the EDHREC cardlists (creatures, instants, …).
    4. Saves each deck to data/scraped_decks/{slug}_{index}.txt.

* MoxfieldScraper:
    Fetches popular public Commander decklists via Moxfield's API.
    Uses exponential back-off on HTTP 429 responses.
    Falls back to a second EDHREC pass when the API is unavailable.

Usage
-----
  python data_pipeline.py

Configuration
-------------
  DRY_RUN = True   -> Process 5 decks only, print what would happen,
                     no data written to training_data.csv
  DRY_RUN = False  -> Full overnight run, appends to training_data.csv
"""

import contextlib
import io
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DRY_RUN = False   # Set to False for the full overnight run

DB_PATH             = "cards.db"
SCRAPED_DECKS_DIR   = "data/scraped_decks"
EDHREC_CACHE_DIR    = "data/edhrec_cache"
MOXFIELD_CACHE_DIR  = "data/moxfield_cache"
TRAINING_CSV        = "data/training_data.csv"
ERRORS_LOG          = "data/pipeline_errors.log"
CHECKPOINT_FILE     = "data/pipeline_checkpoint.json"

SCRYFALL_DELAY = 0.12  # Scryfall asks for 50-100 ms between requests
EDHREC_DELAY   = 0.20  # polite delay between EDHREC requests
MOXFIELD_DELAY = 0.30
DRY_RUN_LIMIT  = 10    # decks to process in dry-run mode

# ---------------------------------------------------------------------------
# CSV schema - superset of ml_trainer.py FEATURE_COLS
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "deck_id", "commander", "color_identity",
    "removed_card", "added_card", "speed_improvement",
    "cmc", "cmc_delta", "is_creature", "is_instant",
    "is_sorcery", "is_artifact", "is_enchantment",
    "produces_mana", "draws_cards", "is_tutor", "has_flash",
    "color_count", "type_match", "edhrec_rank", "is_free_spell",
    "combo_count", "tutor_count", "avg_cmc_of_deck",
    "land_count", "fast_mana_count", "has_combo",
]

HTTP_HEADERS = {"User-Agent": "MTGDeckOptimizer/1.0 (educational project)"}
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, */*",
}


# ---------------------------------------------------------------------------
# Logging and checkpoint helpers
# ---------------------------------------------------------------------------

def _log_error(deck_id: str, exc: Exception) -> None:
    os.makedirs("data", exist_ok=True)
    with open(ERRORS_LOG, "a", encoding="utf-8") as f:
        ts = datetime.now().isoformat()
        f.write(f"[{ts}] {deck_id}: {type(exc).__name__}: {exc}\n")
        f.write(traceback.format_exc())
        f.write("\n")


def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_completed_index": -1, "total_rows": 0,
            "decks_processed": 0, "errors": 0}


def _save_checkpoint(last_idx: int, total_rows: int,
                     decks_processed: int, errors: int) -> None:
    os.makedirs("data", exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "last_completed_index": last_idx,
            "total_rows":           total_rows,
            "decks_processed":      decks_processed,
            "errors":               errors,
        }, f, indent=2)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _ensure_csv_schema() -> None:
    """
    If training_data.csv already exists with the old ml_trainer schema
    (fewer columns), upgrade it by adding missing columns filled with
    empty strings so new pipeline rows append consistently.
    """
    if not os.path.exists(TRAINING_CSV):
        return
    try:
        df = pd.read_csv(TRAINING_CSV)
    except Exception:
        return

    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if not missing:
        return

    for col in missing:
        df[col] = ""
    all_cols = CSV_COLUMNS + [c for c in df.columns if c not in CSV_COLUMNS]
    df[all_cols].to_csv(TRAINING_CSV, index=False)
    print(f"  [Pipeline] Migrated training_data.csv -> "
          f"added {len(missing)} column(s), {len(df)} rows preserved.")


def _count_csv_rows() -> int:
    if not os.path.exists(TRAINING_CSV):
        return 0
    try:
        return len(pd.read_csv(TRAINING_CSV))
    except Exception:
        return 0


def _append_rows(rows: list[dict]) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    os.makedirs("data", exist_ok=True)
    if os.path.exists(TRAINING_CSV):
        new_df.to_csv(TRAINING_CSV, mode="a", header=False, index=False)
    else:
        new_df.to_csv(TRAINING_CSV, index=False)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_json(url: str, cache_path: str | None = None,
              delay: float = 0.2,
              headers: dict | None = None) -> dict | None:
    """
    Fetch JSON with retry + exponential back-off on 429.
    Caches response to cache_path when provided.
    Returns None on 404, persistent failure, or invalid JSON.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    hdrs = headers or HTTP_HEADERS

    for attempt in range(3):
        try:
            time.sleep(delay)
            resp = requests.get(url, headers=hdrs, timeout=15)
            if resp.status_code == 404:
                return None
            if resp.status_code in (403, 401):
                return None  # blocked - no point retrying
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"    [Rate limited] Waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if cache_path:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)),
                            exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
            return data
        except requests.HTTPError:
            return None
        except Exception as exc:
            if attempt == 2:
                print(f"    [HTTP] {url}: {exc}")
                return None
            time.sleep(2 ** attempt)
    return None


def _deep_get(data, *keys, default=None):
    for key in keys:
        if data is None:
            return default
        if isinstance(key, int):
            data = data[key] if isinstance(data, list) and key < len(data) else default
        else:
            data = data.get(key, default) if isinstance(data, dict) else default
    return data


# ---------------------------------------------------------------------------
# EDHREC slug conversion
# ---------------------------------------------------------------------------

def _to_edhrec_slug(name: str) -> str:
    """
    Convert a Magic card name to the EDHREC URL slug.
    e.g. "Atraxa, Praetors' Voice" -> "atraxa-praetors-voice"
    """
    s = name.lower()
    s = re.sub(r"[',\\.!\"?]", "", s)          # remove punctuation
    s = re.sub(r"[^a-z0-9\s\-]", "", s)        # keep alphanum/space/hyphen
    s = re.sub(r"\s+", "-", s.strip())          # spaces -> hyphens
    return re.sub(r"-+", "-", s)                # collapse multiple hyphens


# ---------------------------------------------------------------------------
# EDHREC Scraper
# ---------------------------------------------------------------------------

# How many cards to take from each EDHREC category for the synthetic deck.
_CATEGORY_TARGETS: dict[str, int] = {
    "creatures":        20,
    "instants":         10,
    "sorceries":         8,
    "utilityartifacts":  5,
    "enchantments":      5,
    "planeswalkers":     2,
    "manaartifacts":    10,
    "utilitylands":      5,
    "lands":            34,
}


class EDHRECScraper:
    """
    Builds synthetic Commander decklists from EDHREC card-category data.

    Steps
    -----
    1. Fetch popular commanders from Scryfall (sorted by EDHREC rank).
    2. Convert each name to an EDHREC slug.
    3. Fetch the EDHREC commander page and pick top cards per category.
    4. Save the resulting 99-card deck to data/scraped_decks/.
    """

    COMMANDER_URL = "https://json.edhrec.com/pages/commanders/{slug}.json"

    def __init__(self, target: int = 300, dry_run: bool = False):
        self.target  = DRY_RUN_LIMIT if dry_run else target
        self.dry_run = dry_run
        self.saved: list[str] = []
        os.makedirs(SCRAPED_DECKS_DIR, exist_ok=True)
        os.makedirs(EDHREC_CACHE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Commander list via Scryfall
    # ------------------------------------------------------------------

    def _fetch_commanders_from_scryfall(self) -> list[dict]:
        """
        Return popular commanders from Scryfall, sorted by EDHREC rank.
        Fetches enough pages to cover self.target (with a 2x buffer).
        """
        commanders: list[dict] = []
        needed = self.target * 2  # fetch extra to survive slug mismatches
        page = 1

        while len(commanders) < needed:
            url = (
                "https://api.scryfall.com/cards/search"
                f"?q=is%3Acommander+legal%3Acommander&order=edhrec&page={page}"
            )
            data = _get_json(
                url,
                cache_path=os.path.join(EDHREC_CACHE_DIR,
                                        f"_scryfall_commanders_p{page}.json"),
                delay=SCRYFALL_DELAY,
                headers=HTTP_HEADERS,
            )
            if not data:
                break
            for card in data.get("data", []):
                ci     = card.get("color_identity", [])
                ci_str = "".join(
                    sorted(ci,
                           key=lambda c: "WUBRG".index(c) if c in "WUBRG" else 99)
                ) or "C"
                commanders.append({
                    "name":           card["name"],
                    "slug":           _to_edhrec_slug(card["name"]),
                    "color_identity": ci_str,
                })
            if not data.get("has_more"):
                break
            page += 1

        print(f"  [EDHREC] {len(commanders)} commanders from Scryfall "
              f"({page} page(s)).")
        return commanders

    # ------------------------------------------------------------------
    # Color diversity
    # ------------------------------------------------------------------

    @staticmethod
    def _diversify_commanders(commanders: list[dict]) -> list[dict]:
        """
        Round-robin across color-identity buckets to ensure the final
        commanders list spans all 5 colors rather than clustering at the
        top of EDHREC's popularity ranking (which is dominated by a few
        colour combinations).

        1. Group commanders by color_identity string.
        2. Shuffle within each group (prevents the same popular commander
           from always being chosen first within a color).
        3. Interleave groups: take one from each bucket in sorted order,
           cycling until all commanders are emitted.
        """
        from collections import defaultdict

        buckets: dict[str, list[dict]] = defaultdict(list)
        for cmd in commanders:
            buckets[cmd.get("color_identity", "C")].append(cmd)

        # Shuffle within each colour bucket for variety across runs
        for bucket in buckets.values():
            random.shuffle(bucket)

        # Sort bucket keys so the interleaving order is deterministic
        sorted_keys = sorted(buckets.keys())
        pointers    = {k: 0 for k in sorted_keys}
        diversified: list[dict] = []

        while len(diversified) < len(commanders):
            added = False
            for k in sorted_keys:
                idx = pointers[k]
                if idx < len(buckets[k]):
                    diversified.append(buckets[k][idx])
                    pointers[k] += 1
                    added = True
            if not added:
                break

        return diversified

    # ------------------------------------------------------------------
    # EDHREC per-commander page
    # ------------------------------------------------------------------

    def _fetch_page(self, slug: str) -> dict | None:
        url   = self.COMMANDER_URL.format(slug=slug)
        cache = os.path.join(EDHREC_CACHE_DIR, f"{slug}.json")
        return _get_json(url, cache_path=cache, delay=EDHREC_DELAY,
                         headers=BROWSER_HEADERS)

    def _build_deck(self, page_data: dict, commander_name: str) -> str | None:
        """
        Construct a synthetic deck from EDHREC category cardlists.
        Returns formatted decklist text, or None if too few cards.
        """
        cardlists = _deep_get(page_data, "container", "json_dict", "cardlists") or []
        if not isinstance(cardlists, list):
            return None

        lines = [f"Commander: {commander_name}"]
        seen: set[str] = set()
        total = 0

        for cl in cardlists:
            if not isinstance(cl, dict):
                continue
            tag = cl.get("tag", "")
            n   = _CATEGORY_TARGETS.get(tag, 0)
            if n == 0:
                continue
            for cv in cl.get("cardviews", [])[:n]:
                if not isinstance(cv, dict):
                    continue
                name = cv.get("name")
                if name and name.lower() != commander_name.lower() and name not in seen:
                    seen.add(name)
                    lines.append(f"1 {name}")
                    total += 1

        return "\n".join(lines) if total >= 20 else None

    # ------------------------------------------------------------------

    def _save(self, text: str, slug: str, index: int) -> str:
        path = os.path.join(SCRAPED_DECKS_DIR, f"{slug}_{index}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def scrape(self) -> list[str]:
        print(f"\n[EDHREC] Scraping up to {self.target} decklists "
              f"(Scryfall commanders -> EDHREC pages)...")

        commanders = self._fetch_commanders_from_scryfall()
        if not commanders:
            print("  [EDHREC] No commanders found.")
            return []

        # Diversify: round-robin across color-identity buckets so we don't
        # pull the same mono-black / mono-red top-10 every run.
        commanders = self._diversify_commanders(commanders)
        print(f"  [EDHREC] After color diversification: "
              f"{len(commanders)} commanders queued.")

        index = 0
        for cmd in commanders:
            if len(self.saved) >= self.target:
                break

            slug, name = cmd["slug"], cmd["name"]

            # Skip if already saved from a previous run
            existing = os.path.join(SCRAPED_DECKS_DIR, f"{slug}_{index}.txt")
            if os.path.exists(existing):
                self.saved.append(existing)
                index += 1
                continue

            print(f"  [EDHREC] {name}...", end=" ", flush=True)

            page = self._fetch_page(slug)
            if not page:
                print("skip")
                continue

            text = self._build_deck(page, name)
            if not text:
                print("skip (insufficient card data)")
                continue

            path = self._save(text, slug, index)
            self.saved.append(path)
            index += 1
            card_count = text.count("\n")
            print(f"saved ({card_count} cards)")

        print(f"  [EDHREC] Done - {len(self.saved)} decklists.\n")
        return self.saved


# ---------------------------------------------------------------------------
# Moxfield Scraper
# ---------------------------------------------------------------------------

class MoxfieldScraper:
    """
    Attempts to fetch popular Commander decklists from Moxfield's API.
    Falls back to fetching a second batch of commanders from EDHREC when
    Moxfield's API is unavailable (returns 403).
    """

    SEARCH_URL = "https://api2.moxfield.com/v2/decks/search"
    DECK_URL   = "https://api2.moxfield.com/v2/decks/all/{deck_id}"

    def __init__(self, pages: int = 5, dry_run: bool = False):
        self.pages   = pages
        self.dry_run = dry_run
        self.saved: list[str] = []
        os.makedirs(SCRAPED_DECKS_DIR, exist_ok=True)
        os.makedirs(MOXFIELD_CACHE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Moxfield API
    # ------------------------------------------------------------------

    def _api_available(self) -> bool:
        """Return True if Moxfield's API responds with JSON."""
        r = _get_json(
            f"{self.SEARCH_URL}?pageNumber=1&pageSize=1"
            "&sortType=views&sortDirection=Descending&fmt=commander",
            delay=MOXFIELD_DELAY,
            headers={
                "User-Agent": BROWSER_HEADERS["User-Agent"],
                "Accept": "application/json",
                "Referer": "https://www.moxfield.com/",
                "Origin": "https://www.moxfield.com",
            },
        )
        return r is not None

    def _search_page(self, page_num: int) -> list[dict]:
        params = (f"pageNumber={page_num}&pageSize=64"
                  "&sortType=views&sortDirection=Descending&fmt=commander")
        data = _get_json(
            f"{self.SEARCH_URL}?{params}",
            delay=MOXFIELD_DELAY,
            headers={
                "User-Agent": BROWSER_HEADERS["User-Agent"],
                "Accept": "application/json",
                "Referer": "https://www.moxfield.com/",
            },
        )
        return (data or {}).get("data", [])

    def _fetch_deck(self, deck_id: str) -> dict | None:
        cache = os.path.join(MOXFIELD_CACHE_DIR, f"{deck_id}.json")
        return _get_json(
            self.DECK_URL.format(deck_id=deck_id),
            cache_path=cache,
            delay=MOXFIELD_DELAY,
            headers={
                "User-Agent": BROWSER_HEADERS["User-Agent"],
                "Accept": "application/json",
            },
        )

    def _deck_to_text(self, deck_data: dict) -> str | None:
        cmdr_section = (deck_data.get("commanders")
                        or deck_data.get("commander") or {})
        main_section = (deck_data.get("mainboard")
                        or deck_data.get("main") or {})

        commanders: list[str] = []
        for entry in cmdr_section.values():
            name = ((entry.get("card") or {}).get("name")
                    if isinstance(entry, dict) else str(entry))
            if name:
                commanders.append(name)

        if not commanders:
            return None

        lines = [f"Commander: {n}" for n in commanders]
        for entry in main_section.values():
            if not isinstance(entry, dict):
                continue
            name = ((entry.get("card") or {}).get("name")
                    or entry.get("name"))
            qty  = entry.get("quantity") or 1
            if name:
                lines.append(f"{qty} {name}")

        return "\n".join(lines) if len(lines) >= 20 else None

    # ------------------------------------------------------------------
    # Fallback: second EDHREC pass (less-popular commanders, pages 2+)
    # ------------------------------------------------------------------

    def _edhrec_fallback(self, target: int) -> list[str]:
        """
        When Moxfield is unavailable, scrape a second batch of commanders
        from EDHREC starting from page 2 of the Scryfall rankings.
        """
        print("  [Moxfield] API unavailable - falling back to EDHREC "
              f"for {target} more decklists.")
        # Re-use EDHRECScraper but request a larger batch starting offset
        # by fetching commanders from Scryfall page 2+
        saved: list[str] = []
        page  = 2
        index = 10000   # high offset to avoid filename collisions

        while len(saved) < target:
            data = _get_json(
                "https://api.scryfall.com/cards/search"
                f"?q=is:commander+legal:commander&order=edhrec&page={page}",
                cache_path=os.path.join(EDHREC_CACHE_DIR,
                                        f"_scryfall_commanders_p{page}.json"),
                delay=SCRYFALL_DELAY,
                headers=HTTP_HEADERS,
            )
            if not data:
                break

            scraper = EDHRECScraper(target=target - len(saved),
                                    dry_run=self.dry_run)
            # Process this page of commanders directly
            for card in data.get("data", []):
                if len(saved) >= target:
                    break
                name  = card["name"]
                slug  = _to_edhrec_slug(name)
                fname = os.path.join(SCRAPED_DECKS_DIR, f"{slug}_{index}.txt")
                if os.path.exists(fname):
                    saved.append(fname)
                    index += 1
                    continue
                page_data = scraper._fetch_page(slug)
                if not page_data:
                    continue
                text = scraper._build_deck(page_data, name)
                if not text:
                    continue
                path = scraper._save(text, slug, index)
                saved.append(path)
                index += 1
                print(f"    [EDHREC-fallback] {name} saved")

            if not data.get("has_more"):
                break
            page += 1

        return saved

    # ------------------------------------------------------------------

    def scrape(self) -> list[str]:
        target = DRY_RUN_LIMIT if self.dry_run else self.pages * 64
        print(f"\n[Moxfield] Attempting to fetch up to {target} decklists...")

        if not self._api_available():
            print("  [Moxfield] API not accessible (403).")
            self.saved = self._edhrec_fallback(target)
            print(f"  [Moxfield] Fallback done - {len(self.saved)} decklists.\n")
            return self.saved

        # Moxfield is available - proceed normally
        for page_num in range(1, self.pages + 1):
            if len(self.saved) >= target:
                break
            print(f"  [Moxfield] Page {page_num}/{self.pages}...", flush=True)

            for summary in self._search_page(page_num):
                if len(self.saved) >= target:
                    break
                deck_id  = summary.get("publicId") or summary.get("id")
                if not deck_id:
                    continue
                out_path = os.path.join(SCRAPED_DECKS_DIR,
                                        f"moxfield_{deck_id}.txt")
                if os.path.exists(out_path):
                    self.saved.append(out_path)
                    continue
                deck_data = self._fetch_deck(deck_id)
                if not deck_data:
                    continue
                text = self._deck_to_text(deck_data)
                if not text:
                    continue
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
                self.saved.append(out_path)
                print(f"    Saved moxfield_{deck_id}.txt")

        print(f"  [Moxfield] Done - {len(self.saved)} decklists.\n")
        return self.saved


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def _deduplicate(deck_files: list[str]) -> list[str]:
    """Drop decks that share >80% of cards with an already-accepted deck."""
    from deck_parser import parse_decklist

    accepted: list[tuple[str, set]] = []

    for path in deck_files:
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            cards, _ = parse_decklist(text)
            card_set  = {c.lower() for c in cards}
        except Exception:
            accepted.append((path, set()))
            continue

        if not any(_jaccard(card_set, s) > 0.80 for _, s in accepted):
            accepted.append((path, card_set))

    return [p for p, _ in accepted]


# ---------------------------------------------------------------------------
# Deck-level context statistics
# ---------------------------------------------------------------------------

def _compute_deck_stats(decklist: list[str], matched_combos: list,
                        db_path: str = DB_PATH) -> dict:
    """Return combo_count, tutor_count, avg_cmc_of_deck, land_count, fast_mana_count."""
    from mana_engine import FAST_MANA_CARDS

    unique = list(dict.fromkeys(c.lower() for c in decklist))
    land_count = 0
    non_land_cmcs: list[float] = []
    tutor_count = 0
    fast_mana_count = 0

    try:
        conn = sqlite3.connect(db_path)
        for name in unique:
            row = conn.execute(
                "SELECT mana_value, type_line, oracle_text "
                "FROM cards WHERE lower(name) = lower(?)",
                (name,),
            ).fetchone()
            if not row:
                continue
            mv, tl, oracle = row
            tl_lower     = (tl or "").lower()
            oracle_lower = (oracle or "").lower()
            if "land" in tl_lower:
                land_count += 1
            else:
                non_land_cmcs.append(float(mv or 0))
            if "search your library" in oracle_lower:
                tutor_count += 1
            if name in FAST_MANA_CARDS:
                fast_mana_count += 1
        conn.close()
    except Exception:
        pass

    return {
        "combo_count":     len(matched_combos),
        "tutor_count":     tutor_count,
        "avg_cmc_of_deck": round(
            sum(non_land_cmcs) / len(non_land_cmcs) if non_land_cmcs else 0.0,
            3,
        ),
        "land_count":      land_count,
        "fast_mana_count": fast_mana_count,
    }


# ---------------------------------------------------------------------------
# Proxy speed helpers (for non-combo decks)
# ---------------------------------------------------------------------------

def _card_proxy_stats_from_db(card_name: str, db_path: str = DB_PATH) -> dict:
    """
    Return proxy-relevant attributes for a single card.
    Used to compute delta proxy speed when a swap adds this card.
    """
    from mana_engine import FAST_MANA_CARDS

    name_lower = card_name.lower()
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT mana_value, type_line, oracle_text FROM cards WHERE lower(name) = lower(?)",
            (card_name,),
        ).fetchone()
        conn.close()
    except Exception:
        row = None

    if not row:
        return {
            "is_land": False,
            "is_fast_mana": name_lower in FAST_MANA_CARDS,
            "is_mana_rock": False,
            "cmc": 0.0,
        }

    mv, tl, oracle = row
    tl_lower     = (tl     or "").lower()
    oracle_lower = (oracle or "").lower()
    is_land      = "land"     in tl_lower
    is_fast      = name_lower in FAST_MANA_CARDS
    is_rock      = ("artifact" in tl_lower and "add" in oracle_lower and not is_land
                    and not is_fast)
    return {
        "is_land":     is_land,
        "is_fast_mana": is_fast,
        "is_mana_rock": is_rock,
        "cmc":          float(mv or 0),
    }


def _compute_proxy_baseline(decklist: list[str], db_path: str = DB_PATH) -> tuple:
    """
    Compute proxy-speed aggregates for a non-combo deck in a single DB pass.

    Returns
    -------
    (proxy_speed, fast_count, rock_count, threat_sum, threat_count, per_card_stats)
      proxy_speed  : float — lower = faster deck  (10 - mana_accel + avg_cmc_threats)
      fast_count   : int   — # of fast-mana cards
      rock_count   : int   — # of mana rocks
      threat_sum   : float — cumulative CMC of threat cards
      threat_count : int   — # of threat cards
      per_card_stats : dict[str, dict] — cached stats keyed by lowercase card name
    """
    from mana_engine import FAST_MANA_CARDS

    unique_lower = list(dict.fromkeys(c.lower() for c in decklist))
    fast_count   = 0
    rock_count   = 0
    threat_sum   = 0.0
    threat_count = 0
    per_card: dict[str, dict] = {}

    try:
        conn = sqlite3.connect(db_path)
        for name_lower in unique_lower:
            row = conn.execute(
                "SELECT mana_value, type_line, oracle_text "
                "FROM cards WHERE lower(name) = lower(?)",
                (name_lower,),
            ).fetchone()
            if not row:
                per_card[name_lower] = {
                    "is_land": False, "is_fast_mana": False,
                    "is_mana_rock": False, "cmc": 0.0,
                }
                continue
            mv, tl, oracle = row
            tl_lower     = (tl     or "").lower()
            oracle_lower = (oracle or "").lower()
            is_land  = "land"     in tl_lower
            is_fast  = name_lower in FAST_MANA_CARDS
            is_rock  = ("artifact" in tl_lower and "add" in oracle_lower
                        and not is_land and not is_fast)
            cmc      = float(mv or 0)
            per_card[name_lower] = {
                "is_land": is_land, "is_fast_mana": is_fast,
                "is_mana_rock": is_rock, "cmc": cmc,
            }
            if is_land:
                continue
            if is_fast:
                fast_count += 1
            elif is_rock:
                rock_count += 1
            else:
                threat_sum   += cmc
                threat_count += 1
        conn.close()
    except Exception:
        pass

    mana_accel  = fast_count * 2 + rock_count
    avg_threats = threat_sum / threat_count if threat_count else 3.0
    proxy_speed = 10.0 - mana_accel + avg_threats
    return proxy_speed, fast_count, rock_count, threat_sum, threat_count, per_card


def _proxy_after_swap(
    base_fast:        int,
    base_rocks:       int,
    base_threat_sum:  float,
    base_threat_count: int,
    remove_stats:     dict,
    add_stats:        dict,
) -> float:
    """
    Return the new proxy speed after swapping one card (delta calculation).
    Avoids re-scanning the whole deck for every swap.
    """
    nf  = base_fast
    nr  = base_rocks
    ns  = base_threat_sum
    nc  = base_threat_count

    # Remove old card's contribution
    if not remove_stats["is_land"]:
        if remove_stats["is_fast_mana"]:
            nf -= 1
        elif remove_stats["is_mana_rock"]:
            nr -= 1
        else:
            ns -= remove_stats["cmc"]
            nc -= 1

    # Add new card's contribution
    if not add_stats["is_land"]:
        if add_stats["is_fast_mana"]:
            nf += 1
        elif add_stats["is_mana_rock"]:
            nr += 1
        else:
            ns += add_stats["cmc"]
            nc += 1

    mana_accel  = nf * 2 + nr
    avg_threats = ns / nc if nc else 3.0
    return 10.0 - mana_accel + avg_threats


# ---------------------------------------------------------------------------
# HeadlessOptimizer
# ---------------------------------------------------------------------------

class HeadlessOptimizer:
    """
    Runs the full optimizer pipeline on a saved decklist file.

    All sub-call output is suppressed. Returns the number of swap rows
    written (or that would have been written in dry-run mode).
    """

    def __init__(self, deck_file: str, db_path: str = DB_PATH):
        self.deck_file = deck_file
        self.deck_id   = os.path.splitext(os.path.basename(deck_file))[0]
        self.db_path   = db_path

    @staticmethod
    def _silent(fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            return fn(*args, **kwargs)

    def _resolve_colors(self, commanders: list[str]) -> set[str]:
        colors: set[str] = set()
        try:
            conn = sqlite3.connect(self.db_path)
            for name in commanders:
                row = conn.execute(
                    "SELECT color_identity FROM cards "
                    "WHERE lower(name)=lower(?)", (name,),
                ).fetchone()
                if row:
                    colors |= {c for c in (row[0] or "").split(",") if c}
            conn.close()
        except Exception:
            pass
        return colors

    def run(self, dry_run: bool = False) -> int:
        """
        Run the pipeline on self.deck_file.

        Parameters
        ----------
        dry_run : bool
            True -> compute but do NOT write to CSV.

        Returns
        -------
        int - number of swap records (would-be-)written.
        """
        from deck_parser      import parse_decklist
        from combo_detector   import find_combos
        from optimizer        import (
            categorize_deck,
            _fetch_card_info,
            _fetch_scryfall_candidates,
            _run_speed_silently,
            _make_swapped_deck,
            _avg_turn,
            MAX_CANDIDATES_PER_CARD,
        )
        from ml_trainer       import extract_features, fetch_scryfall_card

        # 1. Parse --------------------------------------------------------
        with open(self.deck_file, encoding="utf-8") as f:
            raw = f.read()

        decklist, commanders = parse_decklist(raw)
        if not decklist:
            raise ValueError("Parsed 0 cards.")
        if not commanders:
            raise ValueError("No commander detected.")

        # 2. Color identity -----------------------------------------------
        color_identity = self._resolve_colors(commanders)
        color_str = "".join(
            sorted(color_identity,
                   key=lambda c: "WUBRG".index(c) if c in "WUBRG" else 99)
        )

        # 3. Find combos --------------------------------------------------
        matched, _ = self._silent(
            find_combos, decklist,
            db_path=self.db_path,
            commander_colors=color_identity or None,
        )
        has_combo = 1 if matched else 0

        # 4. Baseline metric ----------------------------------------------
        if has_combo:
            baseline_speeds = self._silent(
                _run_speed_silently, decklist, matched, self.db_path
            )
            if not baseline_speeds:
                raise ValueError("Could not compute baseline speeds.")
            baseline_avg = _avg_turn(baseline_speeds)
            proxy_state  = None          # not used in combo path
        else:
            # Non-combo deck: use proxy speed as the metric
            (baseline_avg,
             _pf, _pr, _pts, _ptc, _ppc) = _compute_proxy_baseline(
                decklist, self.db_path
            )
            proxy_state = (_pf, _pr, _pts, _ptc, _ppc)
            print(f"  [Proxy] No combos — proxy speed baseline = "
                  f"{baseline_avg:.2f} (lower = faster)")

        # 5. Categorise cards ---------------------------------------------
        locked, swappable = self._silent(
            categorize_deck, decklist, commanders, matched, self.db_path
        )
        if not swappable:
            raise ValueError("No swappable cards found.")

        # 6. Deck-level context stats -------------------------------------
        deck_stats = _compute_deck_stats(decklist, matched, self.db_path)

        # 7. Fetch candidates, test swaps, extract features ---------------
        new_rows: list[dict] = []

        # 8 candidates in dry-run (was 3); full MAX in production.
        max_cands = 8 if dry_run else MAX_CANDIDATES_PER_CARD

        n_swappable = len(swappable)
        for card_idx, card_name in enumerate(swappable, 1):
            print(f"  [Optimizer] card {card_idx}/{n_swappable}: {card_name}...",
                  end=" ", flush=True)

            card_info = self._silent(_fetch_card_info, card_name, self.db_path)
            if not card_info:
                print("skip (not in db)")
                continue

            candidates = self._silent(
                _fetch_scryfall_candidates, card_name, card_info, color_identity
            )
            candidates = candidates[:max_cands]

            # Fetch the removed card's Scryfall data once, outside the loop.
            removed_sf = self._silent(fetch_scryfall_card, card_name)
            if not removed_sf:
                print("skip (no scryfall data)")
                continue

            print(f"{len(candidates)} candidates")

            for cand in candidates:
                add_name = cand["name"]

                # --- score the swap ---
                if has_combo:
                    new_deck   = _make_swapped_deck(decklist, card_name, add_name)
                    new_speeds = self._silent(
                        _run_speed_silently, new_deck, matched, self.db_path
                    )
                    if new_speeds is None:
                        continue
                    score = round(baseline_avg - _avg_turn(new_speeds), 4)
                else:
                    pf, pr, pts, ptc, ppc = proxy_state
                    remove_stats = ppc.get(
                        card_name.lower(),
                        {"is_land": False, "is_fast_mana": False,
                         "is_mana_rock": False, "cmc": 0.0},
                    )
                    add_stats = _card_proxy_stats_from_db(add_name, self.db_path)
                    new_proxy = _proxy_after_swap(
                        pf, pr, pts, ptc, remove_stats, add_stats
                    )
                    score = round(baseline_avg - new_proxy, 4)

                added_sf = self._silent(fetch_scryfall_card, add_name)
                if not added_sf:
                    continue

                feats = extract_features(added_sf, removed_sf)
                new_rows.append({
                    "deck_id":           self.deck_id,
                    "commander":         " + ".join(commanders),
                    "color_identity":    color_str,
                    "removed_card":      card_name,
                    "added_card":        add_name,
                    "speed_improvement": score,
                    "cmc":               feats["cmc"],
                    "cmc_delta":         feats["cmc_delta"],
                    "is_creature":       feats["is_creature"],
                    "is_instant":        feats["is_instant"],
                    "is_sorcery":        feats["is_sorcery"],
                    "is_artifact":       feats["is_artifact"],
                    "is_enchantment":    feats["is_enchantment"],
                    "produces_mana":     feats["produces_mana"],
                    "draws_cards":       feats["draws_cards"],
                    "is_tutor":          feats["is_tutor"],
                    "has_flash":         feats["has_flash"],
                    "color_count":       feats["color_count"],
                    "type_match":        feats["type_match"],
                    "edhrec_rank":       feats["edhrec_rank"],
                    "is_free_spell":     feats["is_free_spell"],
                    **deck_stats,
                    "has_combo":         has_combo,
                })

        if not new_rows:
            raise ValueError("Pipeline produced 0 swap records.")

        if not dry_run:
            _append_rows(new_rows)

        return len(new_rows)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("  MTG TRAINING DATA PIPELINE - Stage 5")
    print("=" * 72)

    if DRY_RUN:
        print("\n  *** DRY RUN MODE - training_data.csv will NOT be modified ***")
        print(f"  Processing up to {DRY_RUN_LIMIT} decks for demonstration.\n")

    # ---- Resume / start-fresh prompt ------------------------------------
    checkpoint = _load_checkpoint()
    start_index = 0

    if checkpoint["decks_processed"] > 0 and not DRY_RUN:
        try:
            answer = input(
                f"\nResume from deck {checkpoint['decks_processed']}? "
                f"({checkpoint['total_rows']} rows so far) (y/n): "
            ).strip().lower()
        except EOFError:
            answer = "y"
        if answer in ("y", "yes"):
            start_index = checkpoint["last_completed_index"] + 1
            print(f"  Resuming from index {start_index}.")
        else:
            checkpoint = {"last_completed_index": -1, "total_rows": 0,
                          "decks_processed": 0, "errors": 0}

    if not DRY_RUN:
        _ensure_csv_schema()

    initial_rows = _count_csv_rows()

    # ---- Step 1: EDHREC -------------------------------------------------
    edhrec_files = EDHRECScraper(target=300, dry_run=DRY_RUN).scrape()

    # ---- Step 2: Moxfield (with EDHREC fallback) -----------------------
    moxfield_files = MoxfieldScraper(pages=5, dry_run=DRY_RUN).scrape()

    # ---- Step 3: Deduplicate -------------------------------------------
    all_files = edhrec_files + moxfield_files
    print(f"[Dedup] {len(all_files)} total decks before deduplication...")
    unique_files = _deduplicate(all_files)
    print(f"[Dedup] {len(unique_files)} unique decks "
          f"({len(all_files) - len(unique_files)} duplicates removed).\n")

    to_process = unique_files[start_index:]
    if DRY_RUN:
        to_process = to_process[:DRY_RUN_LIMIT]

    total_decks        = len(unique_files)
    decks_done         = checkpoint["decks_processed"]
    errors_count       = checkpoint["errors"]
    total_rows         = checkpoint["total_rows"]
    new_swaps_this_run = 0
    deck_times: list[float] = []   # rolling window for time-remaining estimate

    # ---- Step 4: HeadlessOptimizer loop --------------------------------
    print(f"[Optimizer] Processing {len(to_process)} deck(s)...\n")

    for local_idx, deck_file in enumerate(to_process):
        global_idx   = start_index + local_idx
        deck_display = os.path.basename(deck_file)

        try:
            with open(deck_file, encoding="utf-8") as f:
                first = f.read(400)
            m = re.search(r"Commander:\s*(.+)", first)
            if m:
                deck_display = m.group(1).strip()
        except Exception:
            pass

        # --- Time-remaining estimate ---
        if deck_times:
            avg_sec  = sum(deck_times) / len(deck_times)
            remaining_sec = avg_sec * (total_decks - (global_idx + 1))
            hrs_left = remaining_sec / 3600
            eta_str  = f" | ~{hrs_left:.1f} hrs remaining | {total_rows:,} rows so far"
        else:
            eta_str  = ""

        print(f"Processing deck {global_idx + 1}/{total_decks}: "
              f"{deck_display}...{eta_str}")

        deck_start = time.time()
        try:
            n_swaps = HeadlessOptimizer(deck_file, db_path=DB_PATH).run(
                dry_run=DRY_RUN
            )
            total_rows         += n_swaps
            new_swaps_this_run += n_swaps
            decks_done         += 1
            print(f"  -> {n_swaps} swaps recorded (total: {total_rows:,} rows)")

        except Exception as exc:
            errors_count += 1
            _log_error(os.path.basename(deck_file), exc)
            print(f"  -> SKIPPED: {exc}")

        elapsed = time.time() - deck_start
        deck_times.append(elapsed)
        if len(deck_times) > 5:
            deck_times.pop(0)

        if not DRY_RUN and (local_idx + 1) % 10 == 0:
            _save_checkpoint(global_idx, total_rows, decks_done, errors_count)
            print(f"  [Checkpoint saved at deck {global_idx + 1}]")

    if not DRY_RUN:
        _save_checkpoint(
            start_index + len(to_process) - 1,
            total_rows, decks_done, errors_count,
        )

    # ---- Step 5: Summary -----------------------------------------------
    final_rows = _count_csv_rows() if not DRY_RUN else initial_rows + new_swaps_this_run

    if final_rows >= 50000:
        quality = "EXCELLENT (>50k rows)"
    elif final_rows >= 10000:
        quality = "GOOD (>10k rows)"
    elif final_rows >= 1000:
        quality = "FAIR (>1k rows)"
    else:
        quality = f"LOW ({final_rows} rows - run again for more data)"

    print()
    print("=" * 50)
    print("  PIPELINE COMPLETE" + ("  [DRY RUN]" if DRY_RUN else ""))
    print("-" * 50)
    print(f"  Decks processed    : {decks_done}")
    print(f"  Decks skipped      : {errors_count} (errors)")
    print(f"  Swap records added : {new_swaps_this_run:,}")
    print(f"  Training data size : {final_rows:,} total rows")
    print(f"  Estimated quality  : {quality}")
    if DRY_RUN:
        print()
        print("  Set DRY_RUN = False and run again for the overnight pass.")
    print("=" * 50)

    # ---- Step 6: Auto-retrain when pipeline finishes -------------------
    if not DRY_RUN and new_swaps_this_run > 0:
        print("\n[Pipeline] Triggering ml_trainer.py to retrain...")
        try:
            result = subprocess.run(
                [sys.executable, "ml_trainer.py"],
                capture_output=False,
                check=False,
            )
            if result.returncode == 0:
                print("[Pipeline] Retraining complete.")
            else:
                print(f"[Pipeline] ml_trainer.py exited with code "
                      f"{result.returncode}.")
        except Exception as exc:
            print(f"[Pipeline] Could not launch ml_trainer.py: {exc}")


if __name__ == "__main__":
    main()
