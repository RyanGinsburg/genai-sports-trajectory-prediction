"""Side-by-side comparison of the PFF World Cup and IDSSE Bundesliga datasets.

    python -m src.data_analysis.compare_datasets

Prints:
  1. Loading cost: raw size on disk, parse time, cache size.
  2. Volume: matches, minutes, player-hours, and how many of them are actually
     observed (PFF marks off-camera players ESTIMATED; IDSSE has no such flag).
  3. Motion at a common ~5 Hz by speed: acceleration and heading change.
  4. The interpolation test on ~8 s stretches: does the first second of
     movement already point at where the player ends up, and how straight is
     the path? Interpolated (gap-filled) data is straighter and "knows" its endpoint.

Run both loaders first (or let this script parse on first use).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_pipeline import idsse_loader as idsse
from ..data_pipeline import pff_loader as pff

PFF_FPS, IDSSE_FPS = 29.97, idsse.FPS
PFF_STRIDE_5HZ, IDSSE_STRIDE_5HZ = 6, 5  # 4.995 Hz and 5 Hz
SPEED_BINS = [0, 0.5, 1, 2, 3, 4, 5, 6, 9]
PROBE_MATCHES = ["3850", "3832", "10513", "3819", "3835", "10512"]


def dir_size_gb(path: Path, pattern: str = "*") -> float:
    return sum(p.stat().st_size for p in path.rglob(pattern) if p.is_file()) / 1e9


def motion(xy: np.ndarray, dt: float) -> pd.DataFrame:
    v = np.diff(xy, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    h = np.arctan2(v[:, 1], v[:, 0])
    dh = np.abs((h[1:] - h[:-1] + np.pi) % (2 * np.pi) - np.pi)
    return pd.DataFrame({"speed": np.linalg.norm(v[1:], axis=1),
                         "accel": np.linalg.norm(a, axis=1), "dheading_deg": np.degrees(dh)})


def stretch_stats(xy: np.ndarray, fps: float, seconds: float = 8.0) -> list[tuple[float, float]]:
    """Non-overlapping ~8 s stretches with >= 3 m net movement:
    (cosine between the first 1 s of movement and the direction to the end,
     path length / straight-line distance)."""
    w, out = int(seconds * fps), []
    for s in range(int(fps), len(xy) - w, w):
        seg = xy[s:s + w]
        to_end = seg[-1] - xy[s - 1]
        net = np.linalg.norm(to_end)
        if net < 3:
            continue
        first = seg[int(fps)] - xy[s - 1]
        out.append((float(first @ to_end / (np.linalg.norm(first) * net + 1e-9)),
                    float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum() / net)))
    return out


def label_runs(label: np.ndarray) -> list[tuple[int, int, int]]:
    """Maximal runs of equal labels: ``(start, end, label)``."""
    ch = np.flatnonzero(np.diff(label) != 0) + 1
    b = np.r_[0, ch, len(label)]
    return [(s, e, int(label[s])) for s, e in zip(b[:-1], b[1:])]


VISIBLE, ESTIMATED, MISSING = 1, 0, -1


def pff_tracks(match_id: str):
    """Yield ``(xy, label)`` per player-period; label is VISIBLE, ESTIMATED or
    MISSING (no smoothed position, ~31 player-frames per match)."""
    players, _ = pff.load_tracking(match_id)
    for _, g in players.groupby(["period", "side", "jersey"]):
        g = g.sort_values("frame")
        xy = g[["xs", "ys"]].values.astype(float)
        label = np.where(np.isnan(xy).any(axis=1), MISSING,
                         np.where(g["visibility"].values == 1, VISIBLE, ESTIMATED))
        yield xy, label


def idsse_tracks(match_id: str):
    players, _ = idsse.load_positions(match_id)
    for _, g in players.groupby(["period", "person_id"], observed=True):
        yield g.sort_values("frame")[["x", "y"]].values.astype(float)


def volume() -> pd.DataFrame:
    rows = {"PFF World Cup": {"matches": 0, "player_hours": 0.0, "observed_player_hours": 0.0,
                              "hours_in_fully_observed_6s_windows": 0.0},
            "IDSSE Bundesliga": {"matches": 0, "player_hours": 0.0, "observed_player_hours": 0.0,
                                 "hours_in_fully_observed_6s_windows": 0.0}}
    w = int(6 * PFF_FPS)
    for m in pff.tracking_match_ids():
        r = rows["PFF World Cup"]
        r["matches"] += 1
        for _, label in pff_tracks(m):
            r["player_hours"] += len(label) / PFF_FPS / 3600
            r["observed_player_hours"] += (label == VISIBLE).sum() / PFF_FPS / 3600
            # Disjoint 6 s windows that fit inside visible runs.
            r["hours_in_fully_observed_6s_windows"] += sum(
                (e - s) // w for s, e, v in label_runs(label) if v == VISIBLE) * 6 / 3600
    w = int(6 * IDSSE_FPS)
    for m in idsse.match_ids():
        r = rows["IDSSE Bundesliga"]
        r["matches"] += 1
        for xy in idsse_tracks(m):
            hours = len(xy) / IDSSE_FPS / 3600
            r["player_hours"] += hours
            r["observed_player_hours"] += hours
            r["hours_in_fully_observed_6s_windows"] += len(xy) // w * 6 / 3600
    return pd.DataFrame(rows).T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pff-matches", nargs="+", default=PROBE_MATCHES,
                    help="PFF matches used for the motion comparison")
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    print("== 1. Loading cost")
    pff_parse = [pff.load_parse_stats(m).get("parse_seconds") for m in pff.tracking_match_ids()]
    idsse_parse = [idsse.load_parse_stats(m)["parse_seconds"] for m in idsse.match_ids()]
    print(pd.DataFrame({
        "PFF World Cup": {"raw tracking on disk (GB)": dir_size_gb(Path(pff.DATA_ROOT) / pff.TRACKING_DIR),
                          "raw format": "bz2-compressed JSON lines (~800 MB per match uncompressed)",
                          "parse seconds per match (1 core)": f"{np.median(pff_parse):.1f} (median)",
                          "Parquet cache (GB)": dir_size_gb(Path(pff.CACHE_DIR))},
        "IDSSE Bundesliga": {"raw tracking on disk (GB)": dir_size_gb(Path(idsse.DATA_ROOT), "*positions*"),
                             "raw format": "XML, one line per frame per person (~370 MB per match)",
                             "parse seconds per match (1 core)": f"{np.median(idsse_parse):.1f} (median)",
                             "Parquet cache (GB)": dir_size_gb(Path(idsse.CACHE_DIR))},
    }).to_string())

    print("\n== 2. Volume (all matches)")
    print(volume().round(1).to_string())

    print(f"\n== 3. Motion at ~5 Hz (PFF matches: {', '.join(args.pff_matches)}; all IDSSE matches)")
    frames, stretches = [], []
    for m in args.pff_matches:
        for xy, label in pff_tracks(m):
            for s, e, v in label_runs(label):
                if v == MISSING:
                    continue
                kind = "PFF visible" if v == VISIBLE else "PFF estimated"
                seg = xy[s:e][::PFF_STRIDE_5HZ]
                if len(seg) >= 3:
                    frames.append(motion(seg, PFF_STRIDE_5HZ / PFF_FPS).assign(kind=kind))
                if e - s > 10 * PFF_FPS and s > 30 and not np.isnan(xy[s - 30:s]).any():
                    stretches += [(kind, *t) for t in stretch_stats(xy[s - 30:e], PFF_FPS)]
    for m in idsse.match_ids():
        for xy in idsse_tracks(m):
            frames.append(motion(xy[::IDSSE_STRIDE_5HZ], IDSSE_STRIDE_5HZ / IDSSE_FPS).assign(kind="IDSSE"))
            stretches += [("IDSSE", *t) for t in stretch_stats(xy, IDSSE_FPS)]
    d = pd.concat(frames)
    d["speed_bin"] = pd.cut(d["speed"], SPEED_BINS)
    t = d.groupby(["speed_bin", "kind"], observed=True).agg(
        accel_median=("accel", "median"), dheading_median_deg=("dheading_deg", "median")).unstack("kind")
    print(t.round(2).to_string())
    print("\nshare of time in each speed bin:")
    print(pd.crosstab(d["speed_bin"], d["kind"], normalize="columns").round(3).to_string())

    print("\n== 4. Interpolation test on ~8 s stretches (net movement >= 3 m)")
    s = pd.DataFrame(stretches, columns=["kind", "cos_first_1s_vs_end", "path_over_straight_line"])
    print(s.groupby("kind").agg(n=("cos_first_1s_vs_end", "size"),
                                cos_first_1s_vs_end=("cos_first_1s_vs_end", "mean"),
                                path_over_straight_line=("path_over_straight_line", "median"))
          .round(3).to_string())


if __name__ == "__main__":
    main()
