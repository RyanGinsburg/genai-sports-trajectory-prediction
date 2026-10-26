"""Inspect the PFF FC World Cup 2022 dataset.

    python -m src.data_analysis.pff_inspection [--out DIR]

Sections printed:
  1. File coverage: which matches have tracking / metadata / rosters / events.
  2. Per-match tracking summary over all matches: frames, frame rate, gaps,
     players per frame, visibility, coordinate ranges, speeds, orientation, ball.
  3. Raw update rate: do raw coordinates really change every frame?
  4. Smoothing: is the smoothed signal centred in time (does it use future frames)?
     How far apart are raw and smoothed positions?
  5. Estimated (off-camera) stretches: are they filled in using where the
     player re-appears (future information)?
  6. Coverage: how much fully observed (VISIBLE) data exists, and who is visible.

Sections 3-6 use a fixed subset of matches (``--probe-matches``) to stay fast.
The findings are written up in docs/DATASET_COMPARISON.md.
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_pipeline import pff_loader as pff

FPS = 29.97
MAX_PLAUSIBLE_SPEED = 12.0  # m/s; elite sprint peaks are ~10-11 m/s
PROBE_MATCHES = ["3850", "3832", "10513", "3819", "3835", "10512"]


# ---------------------------------------------------------------- helpers

def home_attacks_right(players: pd.DataFrame, window_s: float = 5.0) -> dict[int, bool]:
    """Per period: does the home team attack +x? Teams stand in their own half
    at kick-off, so the team with the smaller mean x attacks right."""
    out = {}
    for period, g in players.groupby("period"):
        g = g[g["t"] - g["t"].min() < window_s]
        out[int(period)] = bool(g.loc[g["side"] == 0, "xs"].mean() < g.loc[g["side"] == 1, "xs"].mean())
    return out


def player_tracks(players: pd.DataFrame):
    """Yield ``(period, side, jersey, track)`` with each track sorted by frame."""
    for key, g in players.groupby(["period", "side", "jersey"]):
        yield (*key, g.sort_values("frame"))


def runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """Maximal runs of equal values: ``(start, end, value)``."""
    ch = np.flatnonzero(np.diff(mask.astype(int)) != 0) + 1
    b = np.r_[0, ch, len(mask)]
    return [(s, e, bool(mask[s])) for s, e in zip(b[:-1], b[1:])]


# ------------------------------------------------------------- section 1

def file_coverage(data_root: Path) -> pd.DataFrame:
    def ids(d: Path, suffix: str) -> set[str]:
        return {f.removesuffix(suffix) for f in os.listdir(d) if f.endswith(suffix)} if d.exists() else set()

    tracking = set(pff.tracking_match_ids(data_root))
    sets = {"tracking": tracking, "metadata": ids(data_root / pff.METADATA_DIR, ".json"),
            "roster": ids(data_root / pff.ROSTERS_DIR, ".json")}
    ev_root = data_root / pff.EVENTS_DIR
    sets["events (root)"] = ids(ev_root, ".json")
    for d in sorted(ev_root.iterdir()):
        if d.is_dir():
            sets[f"events {d.name}"] = ids(d, ".json")
    all_ids = sorted(set().union(*sets.values()), key=int)
    return pd.DataFrame([{"match_id": m, **{k: m in v for k, v in sets.items()}} for m in all_ids])


# ------------------------------------------------------------- section 2

def _speed_stats(players: pd.DataFrame, xcol: str, ycol: str) -> dict:
    df = players.sort_values(["period", "side", "jersey", "frame"])
    same = ((df["side"].values[1:] == df["side"].values[:-1])
            & (df["jersey"].values[1:] == df["jersey"].values[:-1])
            & (np.diff(df["frame"].values) == 1))
    speed = np.hypot(np.diff(df[xcol].values), np.diff(df[ycol].values))[same] * FPS
    vis = (df["visibility"].values[1:] == 1)[same]
    ok = ~np.isnan(speed)
    out = {}
    for name, m in (("all", ok), ("visible", ok & vis), ("estimated", ok & ~vis)):
        s = speed[m]
        out[f"{name}_median"] = float(np.median(s))
        out[f"{name}_p99"] = float(np.percentile(s, 99))
        out[f"{name}_frac_over_12mps"] = float((s > MAX_PLAUSIBLE_SPEED).mean())
        out[f"{name}_frac_zero_step"] = float((s == 0).mean())
    return out


def match_summary(args: tuple[str, str, str]) -> dict:
    match_id, data_root, cache_dir = args
    players, ball = pff.load_tracking(match_id, data_root, cache_dir)
    parse = pff.load_parse_stats(match_id, cache_dir)
    info = pff.load_match_info(match_id, data_root)
    r = {"match_id": match_id, "home": info.home_team, "away": info.away_team,
         "aux_files": ",".join(info.sources), "parse_seconds": parse.get("parse_seconds"),
         "raw_lines": parse["raw_lines"], "duplicate_frames": parse["duplicate_frames"],
         "nonidentical_duplicates": parse["nonidentical_duplicates"], "frames": len(ball),
         "periods": ",".join(map(str, sorted(ball["period"].unique())))}

    dts, gaps = [], 0
    for _, g in ball.groupby("period"):
        dts.append(np.diff(g["t"].values))
        gaps += int((np.diff(g["frame"].values) > 1).sum())
    r["fps_est"] = 1.0 / float(np.median(np.concatenate(dts)))
    r["fps_metadata"] = info.fps
    r["frame_gaps"] = gaps
    r["minutes"] = float(sum(g["t"].max() - g["t"].min() for _, g in ball.groupby("period")) / 60)

    counts = players.groupby(["frame", "side"]).size().unstack(fill_value=0).reindex(
        ball["frame"].values, fill_value=0)
    for side, name in ((0, "home"), (1, "away")):
        r[f"{name}_frac_lt11"] = float((counts[side] < 11).mean())
        r[f"{name}_frac_gt11"] = float((counts[side] > 11).mean())
        r[f"{name}_jerseys"] = int(players.loc[players["side"] == side, "jersey"].nunique())

    # Frames missing for a player between their first and last appearance in a period.
    frames = ball["frame"].values
    missing = total = 0
    for _, _, _, g in player_tracks(players):
        f = g["frame"].values
        n = int(((frames >= f.min()) & (frames <= f.max())).sum())
        total += n
        missing += n - len(f)
    r["player_dropout_frac"] = missing / max(total, 1)

    r["visible_frac"] = float((players["visibility"] == 1).mean())
    r["confidence_low_frac"] = float((players["confidence"] == 0).mean())
    r["raw_xy_nan"] = int(players[["x", "y"]].isna().any(axis=1).sum())
    r["smoothed_xy_nan"] = int(players[["xs", "ys"]].isna().any(axis=1).sum())
    for c in ("xs", "ys"):
        r[f"{c}_p001"] = float(np.nanpercentile(players[c], 0.1))
        r[f"{c}_p999"] = float(np.nanpercentile(players[c], 99.9))
    r["pitch"] = f"{info.pitch_length}x{info.pitch_width}" if info.pitch_length else None
    for tag, (xc, yc) in (("raw", ("x", "y")), ("smoothed", ("xs", "ys"))):
        for k, v in _speed_stats(players, xc, yc).items():
            r[f"speed_{tag}_{k}"] = v

    implied = home_attacks_right(players)
    r["home_attacks_right_by_period"] = json.dumps(implied)
    r["home_start_left_metadata"] = info.home_team_start_left
    r["orientation_matches_metadata"] = (None if info.home_team_start_left is None
                                         else implied[1] == info.home_team_start_left)
    r["ball_raw_missing_frac"] = float(ball["x"].isna().mean())
    r["ball_smoothed_missing_frac"] = float(ball["xs"].isna().mean())
    known = {(int(s), int(j)) for s, j in players[["side", "jersey"]].drop_duplicates().values}
    r["jerseys_with_player_id"] = f"{len(known & set(info.players))}/{len(known)}"
    return r


# ------------------------------------------------------------- section 3

def raw_update_rate(players: pd.DataFrame) -> dict:
    """Raw x/y repeated on alternate frames means the true update rate is half the frame rate."""
    zero, alt = [], []
    for _, _, _, g in player_tracks(players):
        z = (np.hypot(np.diff(g["x"].values), np.diff(g["y"].values)) == 0).astype(int)
        zero.append(z.mean())
        alt.append((z[1:] != z[:-1]).mean())
    g = next(player_tracks(players))[3]
    pattern = (np.hypot(np.diff(g["x"].values), np.diff(g["y"].values))[:24] == 0).astype(int)
    return {"raw_zero_step_frac": float(np.mean(zero)), "raw_zero_step_alternation": float(np.mean(alt)),
            "first_24_steps_zero_pattern": "".join(map(str, pattern))}


# ------------------------------------------------------------- section 4

def smoothing_profile(players: pd.DataFrame, max_tracks: int = 60) -> dict:
    """Error between smoothed(t) and raw(t + lag) on long visible stretches.

    A filter that only uses the past lags behind the signal (minimum at a
    negative lag). A minimum at lag 0 with symmetric errors means a centred
    (non-causal) smoother that also uses future frames.
    """
    lag_err, resid = [], []
    for period, _, _, g in player_tracks(players):
        if period != 1 or len(lag_err) >= max_tracks:
            continue
        s, r = g[["xs", "ys"]].values.astype(float), g[["x", "y"]].values.astype(float)
        vis = g["visibility"].values == 1
        long_runs = [(a, b) for a, b, v in runs(vis)
                     if v and b - a > 600 and not np.isnan(s[a:b]).any()]
        if not long_runs:
            continue
        a, b = long_runs[0]
        s, r = s[a:b], r[a:b]
        lag_err.append({lag: float(np.linalg.norm((s[:len(s) - lag] - r[lag:]) if lag >= 0
                                                  else (s[-lag:] - r[:len(r) + lag]), axis=1).mean())
                        for lag in range(-15, 16, 3)})
        d = r - s
        d0 = d - d.mean(0)
        resid.append([np.linalg.norm(d, axis=1).mean(), np.linalg.norm(d.mean(0))]
                     + [float(np.mean(np.sum(d0[:-k] * d0[k:], 1)) / np.mean(np.sum(d0 * d0, 1)))
                        for k in (15, 30, 90)])
    lags = pd.DataFrame(lag_err).mean()
    res = pd.DataFrame(resid, columns=["mean_raw_minus_smoothed_m", "mean_offset_m",
                                       "autocorr_0.5s", "autocorr_1s", "autocorr_3s"]).median()
    return {"error_by_lag_frames": lags.round(3).to_dict(), "best_lag_frames": int(lags.idxmin()),
            **res.round(3).to_dict()}


# ------------------------------------------------------------- section 5

def estimated_gap_fill(players: pd.DataFrame) -> list[dict]:
    """For each ESTIMATED run (>= 1 s) bounded by VISIBLE frames, compare the
    estimated path with three references over the middle half of the run:
    a straight line to where the player re-appears (needs the future), the
    last visible velocity continued, and the last visible position held.
    """
    out = []
    for _, _, _, g in player_tracks(players):
        xy = g[["xs", "ys"]].values.astype(float)
        vis = g["visibility"].values == 1
        for a, b, v in runs(vis):
            L = b - a
            if v or a < 4 or b == len(vis) or L < 30 or np.isnan(xy[a - 4:b + 1]).any():
                continue
            p0, p1 = xy[a - 1], xy[b]
            k = np.arange(1, L + 1)[:, None]
            est, mid = xy[a:b], slice(L // 4, 3 * L // 4)
            refs = {"line_to_reappearance": p0 + k / (L + 1) * (p1 - p0),
                    "last_velocity": p0 + k * (xy[a - 1] - xy[a - 4]) / 3,
                    "hold_last_position": np.repeat(p0[None], L, 0)}
            out.append({"seconds": L / FPS,
                        **{n: float(np.linalg.norm(est - ref, axis=1)[mid].mean()) for n, ref in refs.items()}})
    return out


# ------------------------------------------------------------- section 6

def visible_coverage(players: pd.DataFrame, ball: pd.DataFrame, info) -> dict:
    """Fully-visible window counts, visible run lengths, and visibility by
    distance to the ball and by position group."""
    windows = {s: [0, 0] for s in (2.0, 6.0, 10.0)}
    run_lengths = []
    for _, _, _, g in player_tracks(players):
        vis = g["visibility"].values == 1
        cs = np.concatenate([[0], np.cumsum(~vis)])
        for s in windows:
            w = int(s * FPS)
            if len(vis) > w:
                full = (cs[w:] - cs[:-w]) == 0
                windows[s][0] += int(full.sum())
                windows[s][1] += len(full)
        run_lengths += [(b - a) / FPS for a, b, v in runs(vis) if v]

    b = ball[["frame", "xs", "ys"]].rename(columns={"xs": "bx", "ys": "by"})
    df = players.merge(b, on="frame").dropna(subset=["bx", "xs"])
    df["dist_to_ball"] = np.hypot(df["xs"] - df["bx"], df["ys"] - df["by"])
    df["position"] = [info.players.get((s, j), {}).get("position")
                      for s, j in zip(df["side"], df["jersey"])]
    return {"windows": windows, "run_lengths": run_lengths, "visibility_rows": df[
        ["dist_to_ball", "position", "visibility"]]}


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=pff.DATA_ROOT)
    ap.add_argument("--cache-dir", default=pff.CACHE_DIR)
    ap.add_argument("--probe-matches", nargs="+", default=PROBE_MATCHES)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", help="optional directory for CSV outputs")
    args = ap.parse_args()
    data_root = Path(args.data_root)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("== 1. File coverage")
    cov = file_coverage(data_root)
    tracked = cov[cov["tracking"]]
    ev_cols = [c for c in cov.columns if c.startswith("events")]
    print(f"matches with any file: {len(cov)}; with tracking: {len(tracked)}")
    print(f"tracking + metadata: {int(tracked['metadata'].sum())}, + roster: {int(tracked['roster'].sum())}, "
          f"+ any events: {int(tracked[ev_cols].any(axis=1).sum())}, "
          f"+ metadata and roster: {int((tracked['metadata'] & tracked['roster']).sum())}, "
          f"no auxiliary file: {int((~tracked[['metadata', 'roster'] + ev_cols].any(axis=1)).sum())}")

    print("\n== 2. Per-match tracking summary (all matches)")
    jobs = [(m, str(data_root), args.cache_dir) for m in pff.tracking_match_ids(data_root)]
    with Pool(args.workers) as pool:
        per_match = pd.DataFrame(pool.map(match_summary, jobs))
    print(per_match.T.to_string())
    num = per_match.select_dtypes("number")
    print("\nacross matches (mean / min / max):")
    print(pd.DataFrame({"mean": num.mean(), "min": num.min(), "max": num.max()}).round(4).to_string())

    sections = {"raw": [], "smooth": [], "gaps": [], "cov": []}
    for m in args.probe_matches:
        players, ball = pff.load_tracking(m, data_root, args.cache_dir)
        sections["raw"].append(raw_update_rate(players))
        sections["smooth"].append(smoothing_profile(players))
        sections["gaps"] += estimated_gap_fill(players)
        sections["cov"].append(visible_coverage(players, ball, pff.load_match_info(m, data_root)))
    print(f"\n(sections 3-6 use matches {', '.join(args.probe_matches)})")

    print("\n== 3. Raw update rate")
    print(pd.DataFrame(sections["raw"]).to_string(index=False))

    print("\n== 4. Smoothing: error of smoothed(t) vs raw(t + lag), metres")
    lags = pd.DataFrame([s["error_by_lag_frames"] for s in sections["smooth"]]).mean()
    print(lags.round(3).to_string())
    print(pd.DataFrame([{k: v for k, v in s.items() if k != "error_by_lag_frames"}
                        for s in sections["smooth"]]).to_string(index=False))

    print("\n== 5. Estimated stretches: mean distance (m) from three references, middle half of run")
    gaps = pd.DataFrame(sections["gaps"])
    gaps["length"] = pd.cut(gaps["seconds"], [1, 2, 4, 8, 16, 1000],
                            labels=["1-2 s", "2-4 s", "4-8 s", "8-16 s", ">16 s"])
    print(gaps.groupby("length", observed=True)[["line_to_reappearance", "last_velocity",
                                                   "hold_last_position"]].median().round(2).to_string())
    print(f"runs: {len(gaps)}")

    print("\n== 6. Fully observed data")
    for s in (2.0, 6.0, 10.0):
        full = sum(c["windows"][s][0] for c in sections["cov"])
        tot = sum(c["windows"][s][1] for c in sections["cov"])
        print(f"{s:>4} s windows fully VISIBLE: {100 * full / tot:.1f}%")
    rl = np.concatenate([c["run_lengths"] for c in sections["cov"]])
    print(f"visible run length: median {np.median(rl):.1f} s, p90 {np.percentile(rl, 90):.1f} s")
    vis = pd.concat([c["visibility_rows"] for c in sections["cov"]])
    vis["dist_bin"] = pd.cut(vis["dist_to_ball"], [0, 5, 10, 20, 30, 40, 60, 120])
    print("visible fraction by distance to ball (m), frames where the ball is tracked:")
    print(vis.groupby("dist_bin", observed=True)["visibility"].mean().round(2).to_string())
    print("visible fraction by position group:")
    print(vis.groupby("position")["visibility"].mean().round(2).sort_values().to_string())

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        cov.to_csv(out / "pff_file_coverage.csv", index=False)
        per_match.to_csv(out / "pff_per_match.csv", index=False)


if __name__ == "__main__":
    main()
