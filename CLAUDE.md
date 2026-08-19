# PokerDom Microlimits Analysis — handoff notes (2026-08-11)

This repo is the offline analysis half of a two-repo project. **The main,
detailed handoff doc — current strategy-bot state, what's actively being
worked on, an unresolved test-harness bug, GitHub status, and how this user
likes work run — lives in the sibling repo:**
`../PokerDom_Practice_App/CLAUDE.md`. Read that first.

## What this repo is

Offline analysis of a real 3.56M-hand microstakes poker dataset (PokerStars
NL25, Zenodo/phh-dataset DOI 10.5281/zenodo.10796885, CC-BY 4.0): hand
parsing, archetype labeling (Nit/TAG/LAG/Maniac/Station/Loose-passive), EV
models, real-showdown hand-range extraction, opponent-adaptation research
(within-session adaptation to Nit-styled and frequent-bluffer opponents
specifically).

This repo's outputs are consumed directly by the sibling
`PokerDom_Practice_App` repo:
- `src.*` modules imported at runtime (card engine, hand rankings, board
  texture, implied ranges) via a hardcoded sibling-path `sys.path` insert —
  **both repos must be cloned as siblings with these exact directory names**
  or Practice_App won't import.
- `data/reference/*.csv` tables (`archetype_facing_bet.csv`,
  `archetype_vs_raise.csv`, `archetype_position_vpip.csv`, etc.) — read
  directly by `backend/bots/abc_bot.py` at runtime for several of its
  opponent-aware rules (see the sibling CLAUDE.md's flag table).

## Recently built here, not yet wired into the bot

- `find_frequent_bluffers.py` — per-player bluff-frequency labeling.
- `check_within_session_adaptation.py` / `check_extreme_opponent_adaptation.py`
  / `check_bluffer_adaptation.py` / `check_pairwise_adaptation.py` — real,
  measured within-session adaptation findings (players genuinely adapt to
  Nit-styled and frequent-bluffer opponents specifically, not to generic
  archetype labels).
- `check_tilt_after_cooler.py` (2026-08-18) — **confirmed real**: players
  play looser/more aggressively for ~10 hands after losing a big pot
  (VPIP +11.75pp, postflop aggression +5.76pp, bigger bets, more thin
  calls, all p~0 on 768k post-cooler hands vs 18.4M baseline). Survives a
  stack-matched re-check and shows a decay curve matching real
  psychological tilt, not a stack-depth artifact — full numbers in the
  script's own docstring. Same "not wired into the bot" status as the
  items above, for the same reason (see next paragraph).

All three of these are flagged in the sibling repo's CLAUDE.md as the
highest-leverage next features to wire into `abc_bot.py` — currently the bot
only ever uses static archetype labels, never per-player bluff frequency,
generic within-session adaptation, or recent-big-pot-loss signal, even
though all three are already computed/confirmed here. The blocker for all
three is the same: `choose_abc_action` only ever sees the CURRENT hand plus
static per-seat archetype labels, no session history at all -- wiring any
of them in needs (a) session-scoped per-seat state passed into the bot
(Practice_App's `backend/dossier.py::TableDossier` already tracks
per-seat session stats live, a natural place to extend) and (b) a way for
`probe_chance_enumeration.py` to A/B test a rule that depends on a
sequence of hands, not one fresh hand at a time -- it currently can't.

## GitHub

Meant to be pushed public under account `le-melon1`. Check current state —
as of the last check this repo may not be pushed yet at all (see sibling
CLAUDE.md's GitHub section, don't trust it blindly, re-verify with
`git remote -v` / `git ls-remote origin`).
