# PokerDom Microlimits Analysis

Post-session (no live HUD) analysis pipeline for microlimit (NL2-equivalent,
1/2 RUB) 6-8max tables: parse raw hand history logs, compute population
tendencies, and find exploitable patterns (fold equity of small sizings,
bluff-catch profitability, BB defense, villain archetype reliability,
within-session opponent adaptation).

## Companion repo

The findings and parsed data here feed a second, sibling repo:
[`PokerDom_Practice_App`](https://github.com/le-melon1/PokerDom_Practice_App)
— a local web app to play against ML bots trained on this project's parsed
hands, plus a hand-coded strategy bot whose rules were derived and A/B-tested
from the tables in `data/reference/` below. That app imports this project's
`src.*` modules directly (not a packaged dependency — a hardcoded relative
`sys.path` insert), so **clone both repos as siblings under the same parent
directory**, with these exact names, if you want the practice app to run:

```
some-folder/
├── PokerDom_Practice_App/
└── PokerDom_Microlimits_Analysis/   (this repo)
```

This repo works standalone on its own for anything not related to that app
(parsing, the Excel report, the adaptation-analysis scripts below).

## Data sources: what's actually real, and what NL2 gap remains

- **Zenodo — "A Dataset of Poker Hand Histories"** ([DOI 10.5281/zenodo.10796885](https://zenodo.org/doi/10.5281/zenodo.10796885),
  mirrored per-file on [GitHub](https://github.com/uoftcprg/phh-dataset)):
  real, verified iPoker Network hands from July 2009 in a TOML-based "PHH"
  format, CC-BY 4.0. **Its lowest stakes tier is NL25 (PokerStars,
  $0.10/$0.25), not NL2** — there is no NL2/NL5/NL10 tier in this archive.
  `data/raw/ps_nl25/` holds the full PokerStars NL25 portion of the archive:
  **4,379 files, 3,564,757 hands, 34,587,959 actions.** `data/raw/ipn_nl100/`
  holds a smaller iPoker-format sample. Not shipped in this repo (raw data
  is excluded via `.gitignore`, both for size and because it isn't this
  project's to redistribute) — download it yourself from the DOI above and
  place it under `data/raw/` in the same layout. The full archive also has
  unused PartyPoker (8.3M hands), iPoker (6.0M), and Absolute
  Poker/Full Tilt/Ongame portions if this is ever revisited for more data —
  currently only the PokerStars NL25 portion is used.
- **KingsHands / HHDealer** (the datamining services named in the original
  brief): checked directly. KingsHands is paid-subscription-only, no free
  samples. HHDealer's advertised free-sample page lists a `WPN_NL2` freebie
  zip, but the link is dead (2018 file, no longer hosted).
- **Forums (2+2, GipsyTeam)**: searched, no downloadable NL2 archive surfaced.
- **Bottom line**: a genuine bulk free NL2 dataset was not found. NL25 is the
  closest available and is still commonly bucketed as "microlimits" (NL2-NL25),
  per this project's own premise that weak-player tendencies are consistent
  across this bracket. If real PokerDom logs become available, use the iPoker
  XML parser below — that's the room's actual network format.

## Three hand-history formats, three parsers

| Parser | Format | Status |
|---|---|---|
| `src/parser/ipoker_xml_parser.py` | iPoker Network XML (`<session><game>...`) | **Verified against real data** — this is PokerDom's actual network format. Action-type codes confirmed against the reference C# parser source; pot arithmetic hand-checked against real 3-bet/all-in/ante hands (see `tests/test_ipoker_xml_parser.py`). |
| `src/parser/phh_parser.py` | HandHQ/Zenodo TOML ("PHH") | **Verified against real data** — used for the NL25/NL100 sample above. Player identities are anonymized hashes; stacks are always `inf` and showdown cards are masked in this public "OBFUSCATED" release *except* where a hand actually reached a real showdown with unmasked cards (1.2M+ such reveals extracted — see below), so winrate (bb/100) isn't computable, but showdown-range and action-based stats are. |
| `src/parser/hand_history_parser.py` | PokerStars-style text | **Unverified guess**, kept for reference/future use if a real PokerStars-format sample turns up. Don't trust it over the other two. |

`main.py` sniffs `.txt` files (iPoker XML vs. PokerStars-text) and routes
`.phhs` files to the PHH parser automatically.

## Pipeline

```
data/raw/**/*.txt, *.phhs   -- raw hand history logs (download separately, see above)
        |
        v  src/parser/{ipoker_xml,phh,hand_history}_parser.py
list[Hand]  (processed in bounded batches -- see note below)
        |
        v  src/pipeline/preprocess.py
hands_df, actions_df (pandas, categorical dtypes)
        |
        +--> src/pipeline/archetypes.py       -- VPIP/PFR/AF -> archetype label
        |                                        (gated by MIN_HANDS_FOR_LABEL)
        |
        +--> src/pipeline/decision_points.py  -- bet-sizing -> did villain fold
        |           v  src/modeling/fold_equity_model.py (CatBoost)
        |    predicted fold% vs breakeven fold% -> profitable sizings
        |
        +--> scripts/extract_showdowns.py, extract_hand_timestamps.py
        |    -> data/processed/{showdowns,hand_outcomes,hand_timestamps}.parquet
        |
        +--> scripts/build_archetype_tables.py, find_frequent_bluffers.py,
        |    find_repeat_opponents.py
        |    -> data/reference/*.csv  (the ABC bot's real-data-derived rules
        |       and the adaptation-analysis scripts below both read these)
        |
        v  src/reporting/excel_report.py
reports/analysis.xlsx (color-coded)
```

Run: `python main.py --raw-dir data/raw --out reports/analysis.xlsx`

**Batched/streaming, not one-shot**: this machine has 8GB RAM, and an
earlier one-shot version (parse every file into one big list of `Hand`
objects before processing anything) silently OOM-crashed at the full
3.56M-hand dataset size — the process vanished mid-run with no traceback.
`main.py` and every `scripts/*.py` file above now parse raw files in bounded
batches (100-200 files at a time) and stream each batch's output straight
into the target parquet via `pyarrow.parquet.ParquetWriter`, instead of
accumulating the whole dataset in memory. Peak memory stays under ~800MB.
**Use this pattern for any new script that re-parses raw files at this
dataset's size** — a one-shot list-then-write approach will not survive.

## Within-session opponent adaptation (2026-08)

Investigated whether real players and the sibling app's ML bots adjust
their play against a specific opponent as a session goes on.

- **Real players**: long-term (across different sessions, same opponent
  pair) — no measurable adaptation. **Within a single session** — real,
  statistically significant adaptation, but *only* against two specific,
  distinctive opponent signals: players labeled **Nit** (`scripts/
  check_extreme_opponent_adaptation.py`) and players independently
  identified as **frequent river bluffers** (`scripts/
  find_frequent_bluffers.py` — shrinkage-estimator ranking toward the
  population bluff-proxy rate, `scripts/check_bluffer_adaptation.py`: fold%
  62.61% -> 61.68%, p=0.0046, n=88,452 events / 7,957 sessions). Generic
  archetype labels (TAG/LAG/Maniac/Station/Loose-passive) showed no such
  effect — the adaptation is to a distinctive, learnable *pattern*, not to
  "how aggressive is this seat" in general.
- **ML bots** (in the sibling app): architecturally cannot do this at all —
  `behavior_clone.py`'s feature set has no opponent-history/memory features.
  Confirmed both by code inspection and by a dedicated simulation in the
  sibling repo (`scripts/check_donk_bluff_reaction.py`, p=0.44, flat across
  deciles). Teaching the ML bots this one specific Nit/frequent-bluffer
  adaptation pattern is scoped as a real follow-up, not yet built.

Session boundaries are detected from real timestamps
(`scripts/extract_hand_timestamps.py`, 45-minute gap ends a session);
adaptation is measured by pooling early-half vs. late-half events across
many real sessions (not one huge session) for statistical power.

## Known approximations (documented, not bugs)

- Board-per-street is inferred from board length (3/4/5 cards), not full
  street-by-street text state.
- Archetype labels (`src/pipeline/archetypes.py`) are rule-based thresholds on
  VPIP/PFR/AF, not a fitted clustering model — flagged `reliable=False` below
  `MIN_HANDS_FOR_LABEL` (100) hands per this project's own calibration
  requirement. Population mix at the current dataset size (26,797
  confidently-labeled players): Loose-passive 29.9%, Station 26.1%, LAG
  14.6%, Maniac 12.5%, TAG 9.5%, Nit 7.4%.
- A handful of real hands have `big_blind == 0` (missed/dead blind edge
  case) — skipped in `decision_points.py` rather than crashing.
- `outcome_known` is only `True` when a hand's winner can be determined
  exactly (uncontested pot, or every live player at a real showdown has
  unmasked cards) — 88.7% of all hands; the other 11.3% reached a real
  showdown with at least one mucked/masked hand and are correctly left
  "unknown," not guessed.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/          # 8 tests
```

## What's still open

- The 148M `data/reference/repeat_opponent_pairs.csv` is excluded from git
  (GitHub's 100MB hard limit) — regenerate with
  `python scripts/find_repeat_opponents.py` if you need it.
- Other room formats in the same Zenodo archive (PartyPoker, iPoker,
  Absolute Poker/Full Tilt/Ongame) are downloaded-but-unused — a path to a
  bigger/more varied dataset if revisited.
- Teaching the sibling app's ML bots the real Nit/frequent-bluffer
  within-session adaptation pattern found above (scoped, not built).
