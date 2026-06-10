"""
Disk-caching layer over the StatsBomb open data (via statsbombpy).

Every network call is wrapped so that the downloaded JSON is written to
cache/ on first access and read from disk on every rerun. This makes the
whole pipeline reproducible offline and keeps us from re-hitting the
StatsBomb GitHub mirror.

We bypass statsbombpy's own (optional) caching and store our own canonical
JSON artifacts, because the progress report needs the raw event JSON to be
auditable and because we want exact control over what is on disk.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pandas as pd

# Silence statsbombpy's "you are using the free version" credentials warning.
warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy")
os.environ.setdefault("SB_DISABLE_WARNINGS", "1")

from statsbombpy import sb  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE = PROJECT_ROOT / "cache"
COMP_CACHE = CACHE / "competitions"
MATCH_CACHE = CACHE / "matches"
LINEUP_CACHE = CACHE / "lineups"
EVENT_CACHE = CACHE / "events"
for _d in (COMP_CACHE, MATCH_CACHE, LINEUP_CACHE, EVENT_CACHE):
    _d.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, obj) -> None:
    # Per-process unique temp name so concurrent writers (e.g. a background
    # download running while another script reads/writes the cache) cannot
    # consume each other's .tmp file before the atomic rename.
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def competitions() -> pd.DataFrame:
    """All free competitions/seasons. Cached as competitions.json."""
    path = COMP_CACHE / "competitions.json"
    if path.exists():
        return pd.DataFrame(_read_json(path))
    df = sb.competitions()
    _write_json(path, df.to_dict(orient="records"))
    return df


def matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """All matches for a (competition, season). Cached per pair."""
    path = MATCH_CACHE / f"comp{competition_id}_season{season_id}.json"
    if path.exists():
        return pd.DataFrame(_read_json(path))
    df = sb.matches(competition_id=competition_id, season_id=season_id)
    _write_json(path, df.to_dict(orient="records"))
    return df


def events(match_id: int) -> pd.DataFrame:
    """All events for a match. Cached per match as raw records."""
    path = EVENT_CACHE / f"match{match_id}.json"
    if path.exists():
        return pd.DataFrame(_read_json(path))
    df = sb.events(match_id=match_id)
    # store as JSON records; dict/list columns survive round-trip fine
    _write_json(path, df.to_dict(orient="records"))
    return df


def lineups(match_id: int) -> dict:
    """
    Lineups for a match, keyed by team name -> DataFrame of players.
    Cached per match.
    """
    path = LINEUP_CACHE / f"match{match_id}.json"
    if path.exists():
        raw = _read_json(path)
        return {team: pd.DataFrame(rows) for team, rows in raw.items()}
    data = sb.lineups(match_id=match_id)  # dict: team -> DataFrame
    _write_json(path, {team: df.to_dict(orient="records") for team, df in data.items()})
    return data
