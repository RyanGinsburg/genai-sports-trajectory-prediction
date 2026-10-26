# Dataset comparison: PFF World Cup 2022 vs. IDSSE Bundesliga

**Bottom line:** we propose **IDSSE Bundesliga as the primary dataset**, with PFF
World Cup as a secondary dataset (fully visible data only). PFF is larger; IDSSE
is cleaner. For trajectory prediction, clean and fully observed movement matters
more than size. The reasons are in §4.

All numbers were measured on our **local** copies with:

```bash
python -m src.data_analysis.pff_inspection     # PFF
python -m src.data_analysis.idsse_inspection   # IDSSE
python -m src.data_analysis.compare_datasets   # side by side
```

Some PFF checks use 6 of the 22 matches to stay fast (3850, 3832, 10513, 3819,
3835, 10512).

---

## 1. PFF / Gradient World Cup 2022 (`data/pff_world_cup/`)

| | |
|---|---|
| **Matches locally** | **22 with tracking**, of the 64 listed in `competitions.csv`. The public release reportedly covers all 64, so our download is probably incomplete. About 2,300 min and 843 player-hours |
| **Tracking frequency** | 29.97 fps in every match, with no frame gaps |
| **Effective coordinate updates** | **Raw x/y change only every other frame (~15 Hz)**. Smoothed x/y change every frame |
| **Players** | 11 per side in every frame, keyed by shirt number. Player IDs exist for 16 of 22 matches. **Only 37% of positions are `VISIBLE`**; the rest are `ESTIMATED` (off camera) |
| **Ball** | Missing in 35% of frames (raw) and 47% (smoothed). No in-play flag |
| **Events / metadata** | Rich PFF events (15 of 22 matches, in three overlapping releases). Metadata 6/22, rosters 6/22, both only 3/22. 4 matches have no auxiliary file at all |
| **Missing data** | 15,750 duplicated frame lines (exact copies, removed by the loader); about 31 player-frames per match without a smoothed position. Otherwise complete, because estimation fills every gap |
| **Coordinates** | Metres from the centre spot, pitch 105 × 68. Teams swap ends each period; direction can be inferred from kick-off positions and matched metadata in all 11 checkable matches |
| **Loading** | 0.97 GB of bz2 JSON (~800 MB per match uncompressed). About 18.5 s per match to parse. Moderate complexity: duplicate frames, shirt-number keys, patchy auxiliary files |

### Raw, smoothed and estimated behaviour: what we tested

**1. Raw coordinates update at about 15 Hz.** Exactly 50% of raw frame-to-frame
steps are zero, in a strict `0101…` pattern (99.9% alternation in every match
tested). Differencing raw positions therefore gives impossible speeds: p99 is
15.8 m/s, and 2% of steps exceed 12 m/s. *Confidence: high.*

**2. Estimated positions appear to use future information.** We ran three
checks:
- Over the middle of each estimated (off-camera) stretch, the estimated path
  stays closest to a **straight line towards where the player re-appears**. That
  reference needs the future. References using only the past (last velocity
  continued, last position held) are further away, e.g. 2.9 m vs. 5.4 m and 7.9 m
  for 8–16 s stretches (26,833 stretches).
- On ~8 s stretches, the first second of estimated movement already points at the
  final position (mean cosine **0.75**, against 0.48 for visible PFF movement and
  0.56 for IDSSE). Estimated paths are also nearly straight: path length ÷
  straight-line distance is 1.04, against about 1.20 for both kinds of real
  movement.
- At the same speed, estimated tracks show about half the acceleration and half
  the heading change of visible ones.

*Confidence: moderate.* This is what offline interpolation between known
positions would produce. Players far from the ball may also genuinely move more
calmly, and without ground truth for off-camera players this cannot be fully
proven.

**3. Smoothing appears to look ahead in time.** The error between smoothed(t)
and raw(t + lag) is lowest at lag 0 and symmetric in both directions, in all 6
matches tested. A filter that used only past frames would lag behind the raw
signal. Raw and smoothed positions also differ by 0.6–1.5 m per match, and the
difference drifts slowly, which suggests position accuracy of roughly a metre.
*Confidence: moderate-to-high* that the smoother is centred; the size of the
look-ahead is not measured.

**Why this matters for trajectory prediction:** a forecasting model must see only
the past.
- A history window containing an estimated stretch, or the last frames of a
  centred smoother, already holds information about the future.
- A target window containing an estimated stretch is PFF's interpolation, not
  real movement.
- Smooth, straight interpolated segments are also easier to predict and to
  compress into tokens.

Any of these would make models look better than they are, and could favour one
method over another for the wrong reason. Estimated positions are 63% of PFF
positions, and the majority for goalkeepers (visible 15% of the time) and for
players more than 30 m from the ball.

**Strengths:** 3× the matches of IDSSE and many more teams (and up to 64 matches
if we re-download). Rich event data. A different competition from IDSSE, so it
can serve as a generalization test.

**Concerns:** the issues above. Fully visible movement is limited to about
214 player-hours of 6 s windows, biased towards the ball. Coverage of auxiliary
files is patchy. Redistribution terms are unclear.

---

## 2. IDSSE / Sportec Bundesliga 2022/23 (`data/idsse_bundesliga/`)

| | |
|---|---|
| **Matches locally** | **7**: 2 Bundesliga (Köln – Bayern, Bochum – Leverkusen) and **5 2. Bundesliga, all Fortuna Düsseldorf home games**. 10 teams, 668 min, 243 player-hours |
| **Tracking frequency** | 25 Hz. **Positions change every frame**: 0% repeated steps while a player moves faster than 2 m/s |
| **Players** | All 22 in every frame (except after two real red cards). Stable `PersonId` across matches. No estimated or visibility flag |
| **Ball** | Present in every frame, with height, possession, and an **in-play flag** (in play 48–59% of the time) |
| **Events / metadata** | Official DFL events in all 7 matches (1,363–1,715 per match, 23–29 types, including `Delete` retractions). Line-ups, positions, result |
| **Missing data** | None: no frame gaps, duplicates or missing values |
| **Coordinates** | Metres from the centre spot, 1 cm precision, pitch 105 × 68. Teams swap ends at half-time. **Events use a corner origin (0–105, 0–68) and local-time timestamps**, while positions use UTC |
| **Loading** | 2.6 GB of XML (~370 MB per match). About 5.2 s per match to parse. Low complexity: one uniform format |

**Tracking quality:** the IDSSE paper describes TRACAB Gen5 multi-camera optical
tracking in the stadium. Our checks agree: every player is present continuously,
and movement looks like PFF's *visible* movement on the 8 s test (0.56 / 1.20),
not like its estimated movement. We cannot test whether the positions were
filtered, because there is no second stream to compare against.

At the same speed, IDSSE shows 1.5–2× the acceleration and heading change of PFF
visible tracks. That is partly PFF's heavier smoothing, and partly IDSSE's 1 cm
rounding and measurement jitter.

**Strengths:** every position is an observation, of every player including
goalkeepers. True 25 Hz updates. Complete ball data with an in-play flag.
Uniform files, easy to load. Clear license (CC BY 4.0).

**Concerns:** only 7 matches, 5 involving Düsseldorf. Any split by match puts the
same players and team on both sides unless we design the split carefully. Small
test sets mean noisier conclusions.

---

## 3. Comparison table

| | PFF World Cup (local) | IDSSE Bundesliga |
|---|---|---|
| Matches | **22** (of 64 released) | 7 (5 Düsseldorf) |
| Player-hours: total / actually observed | **843** / 310 | 243 / **243** |
| Fully observed 6 s windows | 214 player-hours, biased towards the ball | **243 player-hours, all players** |
| Tracking quality | Broadcast video; 63% estimated; accuracy ~1 m | **In-stadium optical; all observed; 1 cm precision** |
| Temporal resolution | 29.97 fps nominal; raw updates ~15 Hz | **25 Hz, real updates** |
| Player tracking completeness | Complete only because of estimation | **Complete and observed** |
| Ball tracking | Missing 35–47% | **Every frame, plus in-play flag** |
| Event data | **Rich (PFF grades)**, 15/22 matches, 3 releases | Official DFL, 7/7 matches |
| Ease of loading | ~18.5 s per match; moderate | **~5.2 s per match; simple** |
| Preprocessing complexity | Higher: dedupe, raw vs. smoothed, visibility masking, patchy metadata | **Lower**: event coordinate and timezone conversion only |
| Leakage concerns | **Estimated positions and smoothing appear to use future frames** | No evidence; Düsseldorf overlap across matches |
| Suitability for trajectory prediction | Usable only on fully visible windows, with care | **Good** |
| Suitability for movement tokenization | Estimated stretches would teach the vocabulary artificially smooth movement | **Good**: real movement texture for every player |

---

## 4. Recommendation

**Propose IDSSE Bundesliga as the primary dataset at the first meeting. Keep PFF
World Cup as a secondary dataset for testing generalization, using only fully
visible windows.**

**The trade-off:** PFF is *larger*. It has 3× the matches (more if we re-download
the full release), many more teams, and 843 vs. 243 player-hours. IDSSE is
*cleaner*: every position is observed at a true 25 Hz. Once PFF is restricted to
data a forecasting model can safely use (fully visible windows), the volumes are
similar (214 vs. 243 player-hours), and PFF's are biased towards the ball. Size
is PFF's advantage only if we accept its estimated positions, and those are what
appear to carry future information.

Top reasons:
1. **No future-information leakage risk found in IDSSE.** PFF's estimated
   positions and smoothing appear to look ahead, which could inflate any
   forecasting result.
2. **Real movement for every player.** That matters both for learning a movement
   vocabulary and for prediction targets.
3. **Simpler, cleaner pipeline.** True 25 Hz, complete ball data with an in-play
   flag, stable player IDs, no duplicates, 3–4× faster loading, and a clear
   CC BY 4.0 license.

**For the team to decide:**
- how to split 7 matches given the Düsseldorf overlap (for example, test on the
  two top-division matches);
- whether to re-download the full 64-match PFF release;
- whether to look for more in-stadium tracking data.
