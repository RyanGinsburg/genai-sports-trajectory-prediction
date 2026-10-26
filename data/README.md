# Data

Raw datasets live in this folder **locally only**. Everything in `data/` except
this README is git-ignored. The datasets are large (about 1.8 GB and 2.5 GB, with
single files over GitHub's 100 MB limit) and come with their own terms of use.
Each team member downloads them separately.

```text
data/
├── README.md            # tracked
├── pff_world_cup/       # local only, git-ignored
├── idsse_bundesliga/    # local only, git-ignored
└── cache/               # created by the loaders, git-ignored, safe to delete
```

## PFF / Gradient 2022 FIFA World Cup

**What it is:** broadcast-video player and ball tracking (29.97 fps), event data
with PFF grades, match metadata and rosters for the 2022 FIFA World Cup. The
public announcement says the release covers all 64 matches. **Our local copy has
tracking for only 22 of them**, so it is probably an incomplete download.

**Access:** free, but you have to request it through the form on PFF FC's
announcement page:
https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset

The same announcement also appears on Gradient Sports' site:
https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset

There is no anonymous download link, so it can't be scripted.

**Expected layout:** unzip so that this folder contains:

```text
data/pff_world_cup/
├── Tracking Data/<match_id>.jsonl.bz2
├── Event Data/          (per-match JSON; our copy also has sub-folders "March 14, 2025/" and "May 1, 2025/")
├── Metadata/<match_id>.json
├── Rosters/<match_id>.json
└── competitions.csv
```

**Restrictions:** the announcement page states no license or redistribution
terms. Treat the data as **not redistributable**: don't commit it, re-host it, or
share copies. Each person should request access themselves and follow the terms
shown when they do. *Note:* an earlier commit of this repository (`dfff662`)
included the raw PFF files, and they remain in the Git history.

## IDSSE / Sportec Bundesliga

**What it is:** official DFL match information, event data and position data for
7 matches of the 2022/23 Bundesliga and 2. Bundesliga. Positions come from
TRACAB Gen5 multi-camera optical tracking in the stadium (25 Hz), for all
players and the ball.

Reference: Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of
spatiotemporal and event data in elite soccer", *Scientific Data*,
https://doi.org/10.1038/s41597-025-04505-y

**Download:** figshare, https://doi.org/10.6084/m9.figshare.28196177. It is also
available through the `floodlight` Python package (`floodlight.io.datasets`).

**Expected layout:** put the 21 XML files directly in the folder. There are three
per match: `DFL_02_01_matchinformation_*`, `DFL_03_02_events_raw_*` and
`DFL_04_03_positions_raw_observed_*`.

```text
data/idsse_bundesliga/DFL_0*_*_DFL-COM-*_DFL-MAT-*.xml
```

**License:** CC BY 4.0, released with the DFL's authorization. Attribution is
required: cite the paper above.
