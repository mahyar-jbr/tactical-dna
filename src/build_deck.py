"""
Build the premium, fully-offline presentation deck (presentation.html).

A single self-contained HTML file: inlines plotly.min.js, embeds the chart data
(deck_data.json) and figure images (deck_figs.json), and renders a custom
slide engine with York University theming, animations, a speaker timer, and
LIVE interactive charts (P2 archetype UMAP, P3 player-role UMAP with neighbour
highlight, animated Clasico network).

Run:  PYTHONPATH=src python src/build_deck.py
Out:  outputs/viz/presentation.html
"""
from __future__ import annotations

import json
from pathlib import Path

from sb_cache import PROJECT_ROOT

VIZ = PROJECT_ROOT / "outputs" / "viz"
PLOTLY = (PROJECT_ROOT / ".venv" / "lib" / "python3.13" / "site-packages"
          / "plotly" / "package_data" / "plotly.min.js")


def load():
    data = json.loads((VIZ / "deck_data.json").read_text())
    # Figure PNGs are no longer embedded — slides 6 & 7 now render native
    # on-theme Plotly charts instead of matplotlib images.
    figs = {}
    plotly_js = PLOTLY.read_text() if PLOTLY.exists() else ""
    return data, figs, plotly_js


def build():
    data, figs, plotly_js = load()
    from deck_template import render
    html = render(data, figs, plotly_js)
    out = VIZ / "presentation.html"
    out.write_text(html, encoding="utf-8")
    mb = out.stat().st_size / 1e6
    print(f"Wrote {out.relative_to(PROJECT_ROOT)} ({mb:.2f} MB)")
    return out


if __name__ == "__main__":
    build()
