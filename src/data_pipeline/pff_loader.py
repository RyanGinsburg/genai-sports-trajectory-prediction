"""Loader for the PFF FC 2022 FIFA World Cup dataset (``data/pff_world_cup/``).

Raw layout:

    Tracking Data/<match>.jsonl.bz2      one JSON object per video frame (29.97 fps)
    Metadata/<match>.json                fps, period times, pitch size, homeTeamStartLeft
    Rosters/<match>.json                 jersey -> player id, name, position, team
    Event Data/[<release>/]<match>.json  PFF events (also embed jersey -> playerId)

Only some matches have each auxiliary file, so everything except tracking is
optional. Parsing one tracking file takes ~15 s, so each match is parsed once
into two Parquet tables (players, ball) under ``cache_dir``.

Raw-data artifacts handled while parsing:
  * Frames are repeated verbatim when several game events attach to one frame.
    The first copy of each ``frameNum`` is kept.
  * ``*PlayersSmoothed`` lists repeat each player on those same frames, so
    smoothed entries are matched to raw ones by jersey number.

Nothing else is changed: coordinates stay as delivered (metres, centre origin,
orientation as in the file).
"""

from __future__ import annotations

import bz2
import json
import os
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = "data/pff_world_cup"
CACHE_DIR = "data/cache/pff"

SIDES = {"home": 0, "away": 1}
VISIBILITY = {"ESTIMATED": 0, "VISIBLE": 1}
CONFIDENCE = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
MISSING_CODE = -1

TRACKING_DIR = "Tracking Data"
METADATA_DIR = "Metadata"
ROSTERS_DIR = "Rosters"
EVENTS_DIR = "Event Data"


@dataclass
class MatchInfo:
    """Per-match context assembled from whichever auxiliary files exist."""

    match_id: str
    fps: float | None = None
    pitch_length: float | None = None
    pitch_width: float | None = None
    home_team_start_left: bool | None = None
    home_team: str | None = None
    away_team: str | None = None
    # (side, jersey) -> {"player_id", "name", "position"}
    players: dict[tuple[int, int], dict] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


def tracking_match_ids(data_root: str | Path = DATA_ROOT) -> list[str]:
    files = os.listdir(Path(data_root) / TRACKING_DIR)
    return sorted(f.removesuffix(".jsonl.bz2") for f in files if f.endswith(".jsonl.bz2"))


def _code(table: dict, value) -> int:
    return table.get(value, MISSING_CODE)


def _num(v) -> float:
    return float("nan") if v is None else float(v)


def parse_tracking(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Parse a raw ``.jsonl.bz2`` tracking file.

    Returns ``(players, ball, parse_stats)``. ``players`` has one row per
    (frame, side, jersey) with raw (x, y) and smoothed (xs, ys) metres.
    """
    t0 = time.perf_counter()
    p_cols = {k: array(t) for k, t in [
        ("frame", "l"), ("period", "b"), ("t", "d"), ("side", "b"), ("jersey", "h"),
        ("x", "f"), ("y", "f"), ("xs", "f"), ("ys", "f"),
        ("visibility", "b"), ("confidence", "b"),
    ]}
    b_cols = {k: array(t) for k, t in [
        ("frame", "l"), ("period", "b"), ("t", "d"),
        ("x", "f"), ("y", "f"), ("z", "f"), ("xs", "f"), ("ys", "f"), ("zs", "f"),
        ("visibility", "b"),
    ]}
    stats = {"raw_lines": 0, "duplicate_frames": 0, "nonidentical_duplicates": 0,
             "smoothed_missing_for_raw": 0, "missing_raw_xy": 0}
    seen: set[int] = set()
    prev = None

    with bz2.open(path, "rt") as fh:
        for line in fh:
            stats["raw_lines"] += 1
            d = json.loads(line)
            frame = d["frameNum"]
            if frame in seen:
                stats["duplicate_frames"] += 1
                same = (prev is not None and prev["frameNum"] == frame
                        and all(prev[k] == d[k] for k in ("homePlayers", "awayPlayers", "balls")))
                stats["nonidentical_duplicates"] += not same
                continue
            seen.add(frame)
            prev = d
            period, t = d["period"], d["periodElapsedTime"]

            for side_name, side in SIDES.items():
                raw = d.get(f"{side_name}Players") or []
                smoothed = {p["jerseyNum"]: p for p in (d.get(f"{side_name}PlayersSmoothed") or [])}
                for p in raw:
                    s = smoothed.get(p["jerseyNum"])
                    if s is None:
                        stats["smoothed_missing_for_raw"] += 1
                    if p.get("x") is None or p.get("y") is None:
                        stats["missing_raw_xy"] += 1
                    p_cols["frame"].append(frame)
                    p_cols["period"].append(period)
                    p_cols["t"].append(t)
                    p_cols["side"].append(side)
                    p_cols["jersey"].append(int(p["jerseyNum"]))
                    p_cols["x"].append(_num(p.get("x")))
                    p_cols["y"].append(_num(p.get("y")))
                    p_cols["xs"].append(_num(s.get("x")) if s else float("nan"))
                    p_cols["ys"].append(_num(s.get("y")) if s else float("nan"))
                    p_cols["visibility"].append(_code(VISIBILITY, p.get("visibility")))
                    p_cols["confidence"].append(_code(CONFIDENCE, p.get("confidence")))

            balls = d.get("balls") or []
            b = balls[0] if balls else {}
            bs = d.get("ballsSmoothed") or {}
            b_cols["frame"].append(frame)
            b_cols["period"].append(period)
            b_cols["t"].append(t)
            for k in ("x", "y", "z"):
                b_cols[k].append(_num(b.get(k)))
                b_cols[k + "s"].append(_num(bs.get(k)))
            b_cols["visibility"].append(_code(VISIBILITY, b.get("visibility")))

    players = pd.DataFrame({k: np.frombuffer(v, dtype=v.typecode) for k, v in p_cols.items()})
    ball = pd.DataFrame({k: np.frombuffer(v, dtype=v.typecode) for k, v in b_cols.items()})
    stats["frames"] = len(ball)
    stats["parse_seconds"] = round(time.perf_counter() - t0, 2)
    return players, ball, stats


def build_cache(match_id: str, data_root: str | Path = DATA_ROOT,
                cache_dir: str | Path = CACHE_DIR) -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    players, ball, stats = parse_tracking(Path(data_root) / TRACKING_DIR / f"{match_id}.jsonl.bz2")
    players.to_parquet(cache_dir / f"{match_id}_players.parquet", index=False)
    ball.to_parquet(cache_dir / f"{match_id}_ball.parquet", index=False)
    (cache_dir / f"{match_id}_parse_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_tracking(match_id: str, data_root: str | Path = DATA_ROOT,
                  cache_dir: str | Path = CACHE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(players, ball)`` for a match, parsing and caching on first use."""
    cache_dir = Path(cache_dir)
    p_path = cache_dir / f"{match_id}_players.parquet"
    b_path = cache_dir / f"{match_id}_ball.parquet"
    if not (p_path.exists() and b_path.exists()):
        build_cache(match_id, data_root, cache_dir)
    return pd.read_parquet(p_path), pd.read_parquet(b_path)


def load_parse_stats(match_id: str, cache_dir: str | Path = CACHE_DIR) -> dict:
    return json.loads((Path(cache_dir) / f"{match_id}_parse_stats.json").read_text())


def _event_files(data_root: Path, match_id: str) -> list[Path]:
    """Event JSON for a match across all releases, newest release first."""
    root = data_root / EVENTS_DIR
    candidates = [d / f"{match_id}.json" for d in sorted(root.iterdir()) if d.is_dir()]
    candidates.append(root / f"{match_id}.json")
    return [c for c in candidates if c.exists()]


def load_match_info(match_id: str, data_root: str | Path = DATA_ROOT,
                    read_events: bool = True) -> MatchInfo:
    """Collect fps, pitch, orientation and jersey -> player info from optional files.

    Precedence: Metadata > Rosters > Event Data. Missing values stay ``None``.
    """
    data_root = Path(data_root)
    info = MatchInfo(match_id)

    meta_path = data_root / METADATA_DIR / f"{match_id}.json"
    home_id = None
    if meta_path.exists():
        m = json.loads(meta_path.read_text())[0]
        info.sources.append("metadata")
        info.fps = m.get("fps")
        info.home_team_start_left = m.get("homeTeamStartLeft")
        info.home_team = m["homeTeam"]["name"]
        info.away_team = m["awayTeam"]["name"]
        pitches = (m.get("stadium") or {}).get("pitches") or []
        if pitches:
            info.pitch_length, info.pitch_width = pitches[0]["length"], pitches[0]["width"]
        home_id = m["homeTeam"]["id"]

    roster_path = data_root / ROSTERS_DIR / f"{match_id}.json"
    if roster_path.exists():
        info.sources.append("roster")
        # Without metadata we cannot tell which roster team is home.
        for r in json.loads(roster_path.read_text()) if home_id is not None else []:
            side = 0 if r["team"]["id"] == home_id else 1
            info.players[(side, int(r["shirtNumber"]))] = {
                "player_id": r["player"]["id"], "name": r["player"]["nickname"],
                "position": r.get("positionGroupType")}

    if read_events:
        for ev_path in _event_files(data_root, match_id)[:1]:  # newest release only
            info.sources.append(f"events:{ev_path.parent.name}")
            for e in json.loads(ev_path.read_text()):
                sm = e.get("stadiumMetadata") or {}
                if info.home_team_start_left is None and sm.get("homeTeamStartLeft") is not None:
                    info.home_team_start_left = sm["homeTeamStartLeft"]
                if info.pitch_length is None and sm.get("pitchLength"):
                    info.pitch_length, info.pitch_width = sm["pitchLength"], sm["pitchWidth"]
                for side, key in ((0, "homePlayers"), (1, "awayPlayers")):
                    for p in e.get(key) or []:
                        k = (side, int(p["jerseyNum"]))
                        if k not in info.players and p.get("playerId") is not None:
                            info.players[k] = {"player_id": str(p["playerId"]), "name": None,
                                               "position": p.get("positionGroupType")}
    return info


def _build_cache_job(args: tuple[str, str, str]) -> tuple[str, dict]:
    return args[0], build_cache(*args)


def main() -> None:
    """Parse every tracking file into the Parquet cache (in parallel)."""
    import argparse
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="rebuild existing caches")
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    jobs = [(m, args.data_root, args.cache_dir) for m in tracking_match_ids(args.data_root)
            if args.force or not (cache / f"{m}_parse_stats.json").exists()]
    with Pool(args.workers) as pool:
        for match_id, stats in pool.imap_unordered(_build_cache_job, jobs):
            print(match_id, stats, flush=True)


if __name__ == "__main__":
    main()
