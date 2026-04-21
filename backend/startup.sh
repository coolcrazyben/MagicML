#!/usr/bin/env bash
# Railway startup script.
# Builds cards.db on first boot (slow ~10-20 min), then starts the API server.
# On a Railway persistent volume the DB survives redeployments.
set -e

cd "$(dirname "$0")"

if [ ! -f cards.db ]; then
  echo "==> cards.db not found — building from Scryfall (first boot, takes ~10-20 min)..."
  python3 build_card_db.py
  python3 build_combo_db.py
  echo "==> Database built."
else
  echo "==> cards.db found, skipping build."
fi

exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
