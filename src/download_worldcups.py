"""
Background pre-download for Phase 2 / Problem 2 (Saturday): cache ALL event
JSON for the men's FIFA World Cup 2018 (comp 43, season 3) and 2022
(comp 43, season 106). Pure download-and-cache; no analysis here.

Idempotent: events already in cache/events/ are skipped, so this can be
re-run safely and resumes where it left off.
"""
from __future__ import annotations

import sys
import time

from sb_cache import matches, events, EVENT_CACHE

TARGETS = [
    (43, 3, "World Cup 2018"),
    (43, 106, "World Cup 2022"),
]


def main():
    for comp, season, name in TARGETS:
        md = matches(comp, season)
        ids = sorted(int(m) for m in md["match_id"].tolist())
        print(f"[{name}] {len(ids)} matches", flush=True)
        done = 0
        for i, mid in enumerate(ids, 1):
            cached = (EVENT_CACHE / f"match{mid}.json").exists()
            try:
                events(mid)  # downloads + caches if missing
                done += 1
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR match {mid}: {e}", flush=True)
                continue
            tag = "cached" if cached else "downloaded"
            if i % 10 == 0 or not cached:
                print(f"  [{name}] {i}/{len(ids)} ({tag} {mid})", flush=True)
        print(f"[{name}] complete: {done}/{len(ids)} events available",
              flush=True)
    print("ALL WORLD CUP DOWNLOADS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
