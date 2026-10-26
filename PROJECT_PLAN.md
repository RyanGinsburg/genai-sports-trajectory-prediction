# GenAI Sports Trajectory Prediction: Project Plan

> **Current stage: dataset selection and initial data exploration.**
> Nothing beyond data loading and dataset inspection has been implemented.
> Everything from the pipeline section onward is planned work, to be agreed at
> the first subteam meeting.

## Project goal

We want to test whether soccer player movement can be represented as a sequence
of **discrete movement tokens**, and whether those tokens can be used to predict
future player trajectories as effectively as, or better than, standard
continuous-coordinate representations.

Instead of modelling a trajectory only as continuous coordinates:

`(x1, y1) -> (x2, y2) -> (x3, y3) -> ...`

we want to convert short movement segments into learned tokens:

`TOKEN_17 -> TOKEN_42 -> TOKEN_42 -> TOKEN_8 -> ...`

Each token would ideally represent a recurring movement pattern, such as
accelerating forward, cutting left, slowing down, or making a curved run.

**Research question:**

> **Can learned discrete movement tokens provide a useful representation for
> soccer player trajectory prediction compared with conventional
> continuous-coordinate representations?**

---

## Datasets (current stage)

We have two candidate tracking datasets locally and have inspected both:

- **PFF / Gradient 2022 FIFA World Cup:** broadcast-video tracking. Our local
  copy has tracking for 22 matches.
- **IDSSE / Sportec Bundesliga 2022/23:** in-stadium optical tracking, 7 matches.

The findings and a proposed choice of primary dataset are in
[docs/DATASET_COMPARISON.md](docs/DATASET_COMPARISON.md). **The team will
confirm the primary dataset at the first meeting.** How to get the data is
described in [data/README.md](data/README.md).

---

## Planned research pipeline (future work)

```text
Raw tracking data
        ↓
Data cleaning + coordinate normalization
        ↓
Player trajectory windows
        ↓
Relative movement representation
        ↓
 ┌─────────────────────┬──────────────────────┐
 │ Continuous baseline │ Movement tokenization│
 └─────────────────────┴──────────────────────┘
        ↓                         ↓
Trajectory prediction model   Token sequence model
        ↓                         ↓
        └──────────────┬──────────┘
                       ↓
              ADE / FDE / RMSE
```

A likely first tokenization baseline is fixed-length trajectory chunks clustered
with **K-means**, with each cluster ID becoming a movement token. Vector
quantization or a VQ-VAE could be explored later. Window lengths, chunk lengths,
vocabulary sizes and train/validation/test splits are **not yet decided**.

---

## Proposed team workstreams (3-person subteam)

These responsibilities are **proposed** and will be finalized at the first
meeting. Workstreams 2 and 3 have not been started. Code is organized by
component, not by person.

### Workstream 1: Data Pipeline & Preprocessing
**Current owner:** Ryan (subteam lead)

Done so far: repo setup, obtaining both datasets, loaders, dataset inspection and
comparison.

Responsibilities later:
- dataset loading
- preprocessing
- coordinate normalization (attacking direction, halves, substitutions)
- trajectory extraction
- standardized, model-ready data consumed by the other workstreams
- integration between components

### Workstream 2: Movement Tokenization
**Owner:** TBD at the first subteam meeting

Responsibilities later:
- movement representation (what a "chunk" of movement is)
- learned token vocabulary
- encoding trajectories into tokens and decoding them back
- token visualization and interpretation
- tokenization experiments (chunk duration, vocabulary size, reconstruction error)

### Workstream 3: Prediction Models & Evaluation
**Owner:** TBD at the first subteam meeting

Responsibilities later:
- continuous-coordinate baseline
- token prediction model (e.g. a small Transformer)
- ADE / FDE / RMSE and related metrics
- experiment comparisons (continuous vs. token; robustness to noise and missing data)
- result visualization

---

## First milestone (planned, after the meeting)

Before building a large model, get one small end-to-end pipeline working:

1. Load one match
2. Extract one or more player trajectories
3. Normalize the trajectories
4. Split them into short movement chunks
5. Cluster the chunks with K-means
6. Convert trajectories into movement tokens
7. Visualize the learned token prototypes
8. Decode the tokens back into trajectories
9. Measure reconstruction error

Then move to trajectory forecasting and compare token-based prediction with a
continuous baseline.

---

## Agenda for the first subteam meeting

- Confirm the primary dataset (and whether, and how, to use the second one).
- Assign owners for Workstreams 2 and 3.
- Agree on the standardized trajectory format that Workstream 1 will deliver.
- Agree on how to split matches for evaluation, given the small number of matches.
- Agree on working conventions: task branches, pull requests into `main`, and
  no raw data in Git.

---

## Repository

Repository: **`genai-sports-trajectory-prediction`**

Current structure:

```text
data/                 local datasets (git-ignored) + README
docs/                 DATASET_COMPARISON.md
notebooks/            dataset_exploration.ipynb
src/data_pipeline/    dataset loaders
src/data_analysis/    dataset inspection and comparison
```

Folders such as `src/tokenization/`, `src/models/` and `src/evaluation/` will be
added when that work starts. Use task-specific Git branches and merge completed
work into `main` through pull requests.
