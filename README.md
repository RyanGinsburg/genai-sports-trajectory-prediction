# GenAI Sports Trajectory Prediction

### Research question

Soccer trajectory prediction models usually work on **continuous** coordinates:
sequences of `(x, y)` positions, velocities and so on. We want to test whether
short segments of player movement can instead be represented as **discrete
movement tokens**, a learned vocabulary of motion patterns such as "accelerate
forward" or "cut left". We then want to know whether predicting the next tokens
works as well as, or better than, predicting continuous coordinates.

> Can learned discrete movement tokens provide a useful representation for
> soccer player trajectory prediction compared with conventional
> continuous-coordinate representations?

Details: [PROJECT_PLAN.md](PROJECT_PLAN.md).

### Current status

**Current phase: Dataset Selection & Initial Data Exploration**

We have obtained and inspected two candidate datasets. No tokenization,
prediction models or experiments have been implemented yet.

### Candidate datasets

- **PFF / Gradient 2022 FIFA World Cup:** broadcast-video tracking; 22 matches
  with tracking in our local copy.
- **IDSSE / Sportec Bundesliga 2022/23:** in-stadium optical tracking; 7 matches.

Findings and the proposed primary dataset:
[docs/DATASET_COMPARISON.md](docs/DATASET_COMPARISON.md).

### Planned high-level pipeline

```text
Tracking Data
     ↓
Data Pipeline
     ↓
Standardized Player Trajectories
     ↓
 ┌──────────────────┬──────────────────┐
 │ Continuous Model │ Movement Tokens  │
 └──────────────────┴──────────────────┘
             ↓
      Trajectory Prediction
             ↓
          Evaluation
```

Only data loading and inspection exist today. Standardized trajectories and
everything after them are planned future work.

### Team

A 3-person subteam with three **proposed** workstreams. Owners for 2 and 3 will
be decided at the first meeting.

1. **Data Pipeline & Preprocessing**: Ryan (subteam lead)
2. **Movement Tokenization**: TBD
3. **Prediction Models & Evaluation**: TBD

### Setup

Tested with Python 3.14 and the package versions pinned in `requirements.txt`.
The pins need a recent Python; 3.12 or newer is recommended.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The datasets are **not in Git**. Download them and place them in
`data/pff_world_cup/` and `data/idsse_bundesliga/` as described in
[data/README.md](data/README.md).

Run from the repository root:

```bash
.venv/bin/python -m src.data_pipeline.pff_loader        # parse PFF tracking once (~70 s, all cores)
.venv/bin/python -m src.data_pipeline.idsse_loader      # parse IDSSE positions once (~7 s)
.venv/bin/python -m src.data_analysis.pff_inspection    # PFF findings
.venv/bin/python -m src.data_analysis.idsse_inspection  # IDSSE findings
.venv/bin/python -m src.data_analysis.compare_datasets  # side-by-side comparison
```

`notebooks/dataset_exploration.ipynb` loads one match from each dataset and plots
one player's trajectory.

```text
data/                 local datasets (git-ignored) + README
docs/                 DATASET_COMPARISON.md
notebooks/            dataset_exploration.ipynb
src/data_pipeline/    pff_loader.py, idsse_loader.py
src/data_analysis/    pff_inspection.py, idsse_inspection.py, compare_datasets.py
```
