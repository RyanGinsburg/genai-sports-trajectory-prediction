# GenAI Sports Trajectory Prediction — Project Plan

## Project Goal

We want to test whether soccer player movement can be represented as a sequence of **discrete movement tokens** and whether those tokens can be used to predict future player trajectories as effectively as, or better than, standard continuous-coordinate representations.

Instead of modeling a trajectory only as continuous coordinates such as:

`(x1, y1) -> (x2, y2) -> (x3, y3) -> ...`

we want to convert short movement segments into learned tokens:

`TOKEN_17 -> TOKEN_42 -> TOKEN_42 -> TOKEN_8 -> ...`

Each token should ideally represent a recurring movement pattern, such as accelerating forward, cutting left, slowing down, or making a curved run.

Our main research question is:

> **Can learned discrete movement tokens provide a useful representation for soccer trajectory prediction compared with continuous coordinate-based approaches?**

---

## Dataset

**Primary dataset:** PFF / Gradient 2022 FIFA World Cup tracking data.

The dataset contains player tracking and event data across World Cup matches. Our first step is to inspect the exact schema and determine:

- tracking frequency
- coordinate conventions
- player and team identifiers
- ball tracking availability
- missing or noisy observations
- event annotations
- how halves, substitutions, and direction changes are represented

If needed, IDSSE / Bundesliga tracking data can serve as a secondary or validation dataset.

---

## Initial Research Pipeline

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

The first tokenization baseline will likely use fixed-length trajectory chunks clustered with **K-means**. Each cluster ID becomes a movement token. More advanced approaches such as vector quantization or VQ-VAE can be explored later.

---

## Team Workstreams

### 1. Data Pipeline & Preprocessing — Ryan
- Inspect and document the World Cup dataset
- Build reusable data loaders
- Normalize coordinates and attacking direction
- Handle missing values, halves, substitutions, and player IDs
- Convert full-game tracking into standardized trajectory windows
- Define the data format consumed by the rest of the team

### 2. Movement Tokenization
- Decide how to represent short movement chunks
- Implement the first K-means tokenizer
- Encode trajectories into token sequences
- Decode tokens back into movement
- Visualize and interpret learned movement tokens
- Experiment with token duration and vocabulary size

### 3. Prediction Models / Architecture
- Build a continuous-coordinate prediction baseline
- Build a token-sequence prediction model
- Keep the two approaches comparable where possible
- Start simple before moving to more complex Transformer-based models

### 4. Evaluation & Research
- Implement ADE, FDE, RMSE, and token reconstruction error
- Review related work and relevant trajectory-prediction baselines
- Design experiment tables and visualizations
- Help test robustness to noise and missing tracking data

---

## First Milestone

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

Once this works, move to trajectory forecasting and compare token-based prediction with a continuous baseline.

---

## Repo / Collaboration Structure

Repository: **`genai-sports-trajectory-prediction`**

Suggested structure:

```text
src/
├── data_pipeline/
├── tokenization/
├── models/
└── evaluation/

notebooks/
experiments/
results/
docs/
```

Project code should be organized by **component**, not by person. Personal exploratory notebooks can be separated if useful, but shared code should move into the appropriate `src/` folder.

Use task-specific Git branches and merge completed work into `main` through pull requests.

---

## Near-Term Success Criteria

By the next few meetings, we should aim to have:

- the dataset schema fully understood
- a standardized trajectory format
- an initial tokenization method working
- token visualizations
- a simple continuous prediction baseline
- evaluation metrics implemented
- a clear experiment plan for comparing continuous vs. token representations

The goal is to first build a clean, reproducible research pipeline, then iterate on the tokenization method and model architecture.
