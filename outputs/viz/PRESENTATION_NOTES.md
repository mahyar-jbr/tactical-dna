# Tactical DNA — Speaker Prep Guide

**File:** `presentation.html` (open by double-click; works offline). **14 slides · target 8:00.**
Author: Mahyar Jaberi · EECS 4414, York University.

---

## 0. How to run it on the day
| Key / action | Effect |
|---|---|
| `F` | Fullscreen — **do this first** |
| `→` / Space / click right side | Next slide |
| `←` / click left side | Previous slide |
| `Home` / `End` | First / last slide |
| Bottom dots | Jump to any slide |

Test before you present: fullscreen, click through once, and run the **slide-10 search demo** (type "Xavi") so you know it works on the room's machine. The whole deck is one offline file — no internet needed.

---

## 1. The 30-second core message (memorize this)
> "I turn a football team's passing into a graph, then ask three questions of it: **who coached it, what style is it, and what role does each player play.** The headline finding is honest — a coach's 'fingerprint' is mostly the *squad*, not the coach — but the player-role embedding works cleanly: it recovers position with 0.88 purity and finds real structural look-alikes across leagues."

If you forget everything else, land that. Everything in the talk supports it.

---

## 2. Slide-by-slide script + timing

> **Format:** ⏱ target time · **what's on screen** · *what you say* (talking points, not a word-for-word script) · ▶ transition.

### Slide 1 — Title ⏱ 0:20
- York logo, Vari Hall photo, your name.
- *"Hi, I'm Mahyar. This is Tactical DNA — using passing networks to read coaches, styles, and player roles."*
- ▶ "Let me start with the question that motivated it."

### Slide 2 — The Hook ⏱ 0:35
- Allianz Arena, "Pundits **say** 'possession football.' Can we **measure** it?"
- *"Football is full of rich tactical language — gegenpressing, style of play, a coach's fingerprint — but it's almost never quantified. My goal: turn that language into something measurable."*
- ▶ "The object I use to measure it is a passing network."

### Slide 3 — The Idea (live Clásico network) ⏱ 0:45
- Xavi portrait + the live network, G=(V,E,w).
- *"A passing network is a directed weighted graph: players are nodes, a completed pass is an edge, the weight is how often. This is the 2010 Clásico, Barça 5–0. Node size is PageRank — and notice **Xavi sits dead-centre**, the biggest node. The maths picks out the metronome the eye would pick."*
- 👉 Optional: hover a node to show its PageRank live.
- ▶ "From one graph like this, I ask three questions."

### Slide 4 — Three Questions ⏱ 0:30
- Three cards: P1 coach, P2 style, P3 role.
- *"P1 — supervised: can I predict the coach? P2 — unsupervised: do tactical styles cluster on their own? P3 — embedding: can I place players in a role space."*
- ▶ "First, the data behind all three."

### Slide 5 — Data ⏱ 0:35
- 775 / 42 / 271 / 11 + team crests.
- *"775 passing networks, 42 teams, from StatsBomb's open data — every Barça La Liga match across 16 seasons and 8 coaches, both recent World Cups, and Bayern. 271 players embedded. Every network is fixed at exactly 11 nodes — I'll explain why next."*
- ▶ "Here's how a graph becomes features."

### Slide 6 — Methods & Features ⏱ 0:50  *(the technical slide — slow down here)*
- Left: construction + notation. Right: the 50 features in 3 families.
- *"Each team-match is a graph G=(V,E,w). To keep them comparable I take only the passes **before the first substitution**, so every graph has exactly 11 nodes. I then compute a 50-dimensional fingerprint in three families: **20 centrality** features — degree, betweenness, eigenvector, PageRank, each summarised four ways; **16 motif** features — the triadic census of 3-node patterns; and **8 global** metrics like density, clustering, the Fiedler value and spectral radius."*
- 🎓 If asked about cost: *"k-means and modularity are NP-hard, so I use Lloyd and Ward heuristics; fixing motifs to 3 nodes keeps the census polynomial."*
- ▶ "With the fingerprint defined, Problem 1: identifying the coach."

### Slide 7 — P1 Results ⏱ 1:00
- Table (baseline 0.263 → RF 0.358) + confusion matrix.
- *"Against a 0.263 majority baseline, a Random Forest hits 0.358 — a real lift. **But here's the honest catch:** when I switch to a leakage-free, season-grouped split, it collapses to 0.250. The model was partly memorising *which season* a match came from, not *which coach*. In the confusion matrix, Guardiola is the one clearly identifiable signature; the single-season coaches just collapse into the majority class."*
- ▶ "So is the signal really the coach? I tested that directly."

### Slide 8 — P1 Transfer (the marquee test) ⏱ 1:00
- Table + bar chart + the red callout.
- *"The real test: train on Barça, then test three of its coaches managing a **different team**. Guardiola at Bayern: recovered 0 of 2 — his Bayern networks read as *other Barça coaches*. Across all three, mean recovery is about 0.18, barely above chance. **The conclusion: the fingerprint is largely club- and roster-bound, not coach-bound.** The one partial exception is Luis Enrique, whose possession identity survives to Spain — XGBoost gets 3 of 4."*
- ▶ "That was supervised. Now without any labels — do styles emerge on their own?"

### Slide 9 — P2 Archetypes (live) ⏱ 1:00
- Live UMAP, blue "direct" vs orange "possession", landmarks.
- *"I clustered all 774 networks with no labels. The silhouette picks k=2, and the axis it finds is exactly **possession versus direct** — and it cuts across competitions: possession-minded national teams sit with Barça. The 2010 Clásico is deep in the possession region; the 2022 World Cup Final sits near the boundary. The clusters blend, which tells us styles are a **continuum**, not discrete boxes."*
- 👉 Optional: hover a few points to show team/competition.
- ▶ "The clearest result, though, is at the player level."

### Slide 10 — P3 Player Roles (live demo — your wow moment) ⏱ 1:20
- Search box + role-space UMAP.
- *"Here I embed players instead of matches. Goalkeepers form a totally separate island; forwards, mids, defenders separate cleanly. Position purity is **0.88** among the 5 nearest neighbours — three times chance."*
- 🎬 **DEMO:** type **"Xavi"** → *"Watch — Xavi's nearest neighbours in role space light up: Iniesta, Busquets, Thiago, Cesc — the deep playmakers you'd name by hand."* Then type **"Modric"** → *"And Modrić, who only appears for Croatia, matches creators from completely different teams — a genuine cross-context analogue. That's the recruitment use case: find players who fill a structural role."*
- ▶ "Quick look under the hood at the whole pipeline."

### Slide 11 — ML Pipeline ⏱ 0:45
- Graphs → features → models → insight + 3 boxes.
- *"To tie it together: every problem runs the same pipeline — graph, 50 features, a model. P1 is supervised (LR, Random Forest, XGBoost). P2 is k-means and Ward with UMAP. P3 is PCA and UMAP with k-NN retrieval. And one finding recurs across all three: **centrality, not motifs, carries the signal.**"*
- ▶ "So what did we actually learn?"

### Slide 12 — Three Findings ⏱ 0:40
- Three cards: Club > coach · A continuum · Roles are real.
- *"Three takeaways. One: club beats coach — the fingerprint is mostly the squad. Two: styles are a continuum, recovered with no labels. Three: roles are real — 0.88 purity and tactically coherent neighbours."*
- ▶ "Where does this go?"

### Slide 13 — Impact & Future ⏱ 0:40
- Applications today / Future work.
- *"Today it's a scouting tool — recruitment by role, opponent-style summaries, cross-league analogues. Next: de-bias the corpus with more clubs, temporal networks for in-match adaptation, and graph neural nets to learn the embedding end-to-end instead of hand-building it."*
- ▶ "To wrap up."

### Slide 14 — Close ⏱ 0:10
- "Tactical DNA — Qualitative tactics, made measurable. Thank you."
- *"Qualitative tactics, made measurable. Thank you — happy to take questions."*

**Running total ≈ 8:30.** If you're over, the easiest 60s to trim: shorten slides 2, 4, 11.

---

## 3. Anticipated Q&A (prep answers)
- **"Why only 11 nodes / cut at the first sub?"** → To keep graphs the same size; centrality and motif stats are sensitive to node count. The pre-sub window guarantees all 11 starters are on the pitch.
- **"Isn't 0.358 accuracy weak?"** → Yes, deliberately — and that's the point. The honest, leakage-free number is 0.250, near baseline. The story is that coach signal is weak *because* it's mostly the squad. Reporting the inflated number would be dishonest.
- **"Why is the corpus so Barça-heavy?"** → The open data only has Real Madrid/Atlético as opponents, not as their own teams. Barça is the only club with rich within-club coaching variation. It's the main limitation; de-biasing needs more clubs.
- **"Why UMAP and not t-SNE / PCA only?"** → UMAP preserves local neighbourhood structure better for the k-NN retrieval; I report PCA too for interpretability (it's in the report).
- **"Centrality beats motifs — why?"** → Centrality captures *how hierarchical* the passing is (who the hubs are), which varies more across styles than the local 3-node patterns do.
- **"Is the Enrique result real or noise?"** → Honest answer: suggestive, not conclusive. n≤4, and only XGBoost recovers it. I flag it as a hypothesis (stylistic extremity may be more portable) worth testing on a bigger corpus.
- **"What's the live demo actually computing?"** → k-nearest-neighbours in the standardized 14-dimensional ego-feature space — the exact same neighbours reported in the paper, not the 2-D projection.

---

## 4. Rehearsal checklist
- [ ] Run through once out loud with a timer — note where you run long.
- [ ] Practice the **slide-10 demo** 3×: type Xavi, then Modric, then Clear. Smooth it out.
- [ ] Memorize the **3 headline numbers**: 0.358 → 0.250 (leakage), 0.18 (transfer), 0.88 (purity).
- [ ] Have the one-line core message (Section 1) ready as your opener and closer.
- [ ] Decide your two "cut if long" slides in advance (2 and 11).
- [ ] On the day: fullscreen (`F`), start clock, breathe, drive with `→`.

## 5. Delivery tips
- The **honesty is your strength** — lean into "the naive number was misleading, here's the real one." Examiners reward that.
- Slides 7–10 are the substance (4+ min). Don't rush them; the front half (1–6) can move quickly.
- The slide-10 demo is your memorable moment — pause, let it land, narrate the names as they light up.
