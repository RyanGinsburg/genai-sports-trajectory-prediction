"""Inspect the IDSSE / Sportec (DFL) Bundesliga dataset.

    python -m src.data_analysis.idsse_inspection [--out DIR]

Per match: competition, duration, frame gaps, players per frame, missing
values, coordinate ranges and precision, whether positions really update every
frame, speeds, orientation, ball status, and the event file's contents. The
findings are written up in docs/DATASET_COMPARISON.md.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_pipeline import idsse_loader as idsse

FPS = idsse.FPS
MAX_PLAUSIBLE_SPEED = 12.0


def player_tracks(players: pd.DataFrame):
    """Yield ``(period, person_id, track)`` with each track sorted by frame."""
    for (period, pid), g in players.groupby(["period", "person_id"], observed=True):
        yield period, pid, g.sort_values("frame")


def event_summary(match_id: str, data_root: str) -> dict:
    text = idsse._file(data_root, "03_02_events_raw", match_id).read_text(encoding="utf-8")
    types = Counter(re.findall(r"<Event [^>]*>\s*<(\w+)", text))
    return {"events": sum(types.values()), "event_types": len(types),
            "top_event_types": ", ".join(f"{k} {v}" for k, v in types.most_common(6)),
            "substitutions": text.count("<Substitution")}


def match_summary(match_id: str, data_root: str, cache_dir: str) -> dict:
    info = idsse.load_match_info(match_id, data_root)
    players, ball = idsse.load_positions(match_id, data_root, cache_dir)
    r = {"match_id": match_id, "competition": info["competition"], "matchday": info["matchday"],
         "title": info["title"], "result": info["result"],
         "pitch": f'{info["pitch_length"]:g}x{info["pitch_width"]:g}',
         "parse_seconds": idsse.load_parse_stats(match_id, cache_dir)["parse_seconds"],
         "periods": ",".join(map(str, sorted(ball["period"].unique())))}

    r["minutes"] = sum(len(g) for _, g in ball.groupby("period")) / FPS / 60
    r["ball_frame_gaps"] = int(sum((np.diff(np.sort(g["frame"].values)) != 1).sum()
                                   for _, g in ball.groupby("period")))

    # A FrameSet covers the time a player is on the pitch, so gaps inside it are dropouts.
    gaps = dup = 0
    steps, provided = [], []
    for _, _, g in player_tracks(players):
        fd = np.diff(g["frame"].values)
        gaps += int((fd > 1).sum())
        dup += int((fd == 0).sum())
        ok = fd == 1
        steps.append(np.hypot(np.diff(g["x"].values), np.diff(g["y"].values))[ok])
        provided.append(g["speed"].values[1:][ok] / 3.6)  # S is km/h
    steps, provided = np.concatenate(steps), np.concatenate(provided)
    r["player_track_gaps"] = gaps
    r["player_duplicate_frames"] = dup
    r["players_used"] = int(players["person_id"].nunique())

    counts = players.groupby(["frame", "side"]).size().unstack(fill_value=0).reindex(
        ball["frame"].values, fill_value=0)
    for side, name in ((0, "home"), (1, "away")):
        r[f"{name}_min_players"] = int(counts[side].min())
        r[f"{name}_max_players"] = int(counts[side].max())
    r["frac_frames_22"] = float(((counts[0] == 11) & (counts[1] == 11)).mean())

    r["xy_nan"] = int(players[["x", "y"]].isna().any(axis=1).sum())
    cm = players["x"].values.astype(np.float64) * 100
    r["x_on_1cm_grid_frac"] = float((np.abs(cm - np.round(cm)) < 1e-3).mean())  # source precision
    for c in ("x", "y"):
        r[f"{c}_p001"] = float(np.percentile(players[c], 0.1))
        r[f"{c}_p999"] = float(np.percentile(players[c], 99.9))
    r["frac_outside_pitch_2m"] = float(((players["x"].abs() > info["pitch_length"] / 2 + 2)
                                        | (players["y"].abs() > info["pitch_width"] / 2 + 2)).mean())

    speed = steps * FPS
    moving = provided > 2.0  # > 8 cm per frame, far above the 1 cm rounding
    r["speed_median"] = float(np.median(speed))
    r["speed_p99"] = float(np.percentile(speed, 99))
    r["speed_frac_over_12mps"] = float((speed > MAX_PLAUSIBLE_SPEED).mean())
    r["zero_step_frac"] = float((steps == 0).mean())
    r["zero_step_frac_when_moving_over_2mps"] = float((steps[moving] == 0).mean())
    r["provided_vs_computed_speed_median_abs_diff"] = float(np.median(np.abs(provided - speed)))

    implied = {}
    for period, g in players.groupby("period"):
        g = g[g["frame"] < g["frame"].min() + 5 * FPS]
        implied[int(period)] = bool(g.loc[g["side"] == 0, "x"].mean() < g.loc[g["side"] == 1, "x"].mean())
    r["home_attacks_right_by_period"] = json.dumps(implied)

    r["ball_xy_nan"] = int(ball[["x", "y"]].isna().any(axis=1).sum())
    r["ball_in_play_frac"] = float((ball["status"] == 1).mean())
    r.update(event_summary(match_id, data_root))
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=idsse.DATA_ROOT)
    ap.add_argument("--cache-dir", default=idsse.CACHE_DIR)
    ap.add_argument("--out", help="optional directory for CSV outputs")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 60)

    per_match = pd.DataFrame([match_summary(m, args.data_root, args.cache_dir)
                              for m in idsse.match_ids(args.data_root)])
    print(per_match.T.to_string())
    print(f"\nmatches: {len(per_match)}; minutes: {per_match['minutes'].sum():.0f}; "
          f"distinct teams: {len({t for s in per_match['title'] for t in s.split(':')})}")
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        per_match.to_csv(Path(args.out) / "idsse_per_match.csv", index=False)


if __name__ == "__main__":
    main()
