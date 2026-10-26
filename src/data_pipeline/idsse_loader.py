"""Loader for the IDSSE / Sportec (DFL) Bundesliga dataset (``data/idsse_bundesliga/``).

Raw layout, three XML files per match, keyed by a DFL match id such as DFL-MAT-J03WMX:

    DFL_02_01_matchinformation_<COMP>_<MATCH>.xml   teams, players, pitch size
    DFL_03_02_events_raw_<COMP>_<MATCH>.xml         events (corner-origin coordinates)
    DFL_04_03_positions_raw_observed_<COMP>_<MATCH>.xml
        <FrameSet GameSection TeamId PersonId> blocks, one per person per half,
        each holding <Frame N T X Y D S A M [Z BallPossession BallStatus]/> at 25 Hz.
        Positions are centre-origin metres; TeamId is a club id, "referee" or "BALL".

The 350-420 MB positions files are read line by line with regexes (a full XML
parse is needlessly slow) and cached as Parquet. Referees are skipped.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from array import array
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = "data/idsse_bundesliga"
CACHE_DIR = "data/cache/idsse"
FPS = 25.0
SECTIONS = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}

_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]* X="([-\d.]+)" Y="([-\d.]+)"(?: Z="([-\d.]+)")?'
                    r' D="([-\d.]+)" S="([-\d.]+)" A="([-\d.]+)" M="(-?\d+)"'
                    r'(?: BallPossession="(\d+)" BallStatus="(\d+)")?')


def match_ids(data_root: str | Path = DATA_ROOT) -> list[str]:
    return sorted(p.name.split("_")[-1].removesuffix(".xml")
                  for p in Path(data_root).glob("DFL_04_03_positions_raw_observed_*.xml"))


def _file(data_root: str | Path, kind: str, match_id: str) -> Path:
    (path,) = Path(data_root).glob(f"DFL_{kind}_*_{match_id}.xml")
    return path


def load_match_info(match_id: str, data_root: str | Path = DATA_ROOT) -> dict:
    root = ET.parse(_file(data_root, "02_01_matchinformation", match_id)).getroot()
    g = root.find(".//General").attrib
    env = root.find(".//Environment").attrib
    players = {}
    for team in root.iter("Team"):
        side = 0 if team.get("Role") == "home" else 1
        for p in team.iter("Player"):
            players[p.get("PersonId")] = {"side": side, "team_id": team.get("TeamId"),
                                          "team": team.get("TeamName"),
                                          "jersey": int(p.get("ShirtNumber")),
                                          "name": p.get("Shortname"),
                                          "position": p.get("PlayingPosition"),
                                          "starting": p.get("Starting") == "true"}
    return {"match_id": match_id, "competition": g["CompetitionName"], "season": g["Season"],
            "matchday": int(g["MatchDay"]), "title": g["MatchTitle"], "result": g["Result"],
            "home_team_id": g["HomeTeamId"], "guest_team_id": g["GuestTeamId"],
            "pitch_length": float(env["PitchX"]), "pitch_width": float(env["PitchY"]),
            "players": players}


def parse_positions(path: str | Path, info: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return ``(players, ball, parse_stats)``.

    ``players``: frame, period, side (0 home / 1 guest), jersey, person_id, x, y
    (metres), speed (provided ``S``, km/h), accel (provided ``A``).
    ``ball``: frame, period, x, y, z, possession (1/2), status (1 = in play).
    """
    t0 = time.perf_counter()
    p = {k: array(t) for k, t in [("frame", "l"), ("period", "b"), ("side", "b"),
                                  ("jersey", "h"), ("x", "f"), ("y", "f"),
                                  ("speed", "f"), ("accel", "f")]}
    person: list[str] = []
    b = {k: array(t) for k, t in [("frame", "l"), ("period", "b"), ("x", "f"), ("y", "f"),
                                  ("z", "f"), ("possession", "b"), ("status", "b")]}
    mode, period, side, jersey, pid = None, 0, 0, 0, ""
    lines = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            lines += 1
            if "<Frame " in line:
                if mode is None:
                    continue
                m = _FRAME.search(line)
                n, x, y = int(m[1]), float(m[2]), float(m[3])
                if mode == "ball":
                    for k, v in (("frame", n), ("period", period), ("x", x), ("y", y),
                                 ("z", float(m[4] or "nan")), ("possession", int(m[9] or -1)),
                                 ("status", int(m[10] or -1))):
                        b[k].append(v)
                else:
                    for k, v in (("frame", n), ("period", period), ("side", side),
                                 ("jersey", jersey), ("x", x), ("y", y),
                                 ("speed", float(m[6])), ("accel", float(m[7]))):
                        p[k].append(v)
                    person.append(pid)
            elif "<FrameSet " in line:
                section, team, pid = _FRAMESET.search(line).groups()
                period = SECTIONS[section]
                if team == "BALL":
                    mode = "ball"
                elif team == "referee":
                    mode = None
                else:
                    mode = "player"
                    meta = info["players"][pid]
                    side, jersey = meta["side"], meta["jersey"]
    players = pd.DataFrame({k: np.frombuffer(v, dtype=v.typecode) for k, v in p.items()})
    players["person_id"] = pd.Categorical(person)
    ball = pd.DataFrame({k: np.frombuffer(v, dtype=v.typecode) for k, v in b.items()})
    stats = {"raw_lines": lines, "player_rows": len(players), "ball_frames": len(ball),
             "parse_seconds": round(time.perf_counter() - t0, 2)}
    return players, ball, stats


def build_cache(match_id: str, data_root: str | Path = DATA_ROOT,
                cache_dir: str | Path = CACHE_DIR) -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    info = load_match_info(match_id, data_root)
    players, ball, stats = parse_positions(
        _file(data_root, "04_03_positions_raw_observed", match_id), info)
    players.to_parquet(cache_dir / f"{match_id}_players.parquet", index=False)
    ball.to_parquet(cache_dir / f"{match_id}_ball.parquet", index=False)
    (cache_dir / f"{match_id}_parse_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_positions(match_id: str, data_root: str | Path = DATA_ROOT,
                   cache_dir: str | Path = CACHE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(players, ball)`` for a match, parsing and caching on first use."""
    cache_dir = Path(cache_dir)
    pp, bp = cache_dir / f"{match_id}_players.parquet", cache_dir / f"{match_id}_ball.parquet"
    if not (pp.exists() and bp.exists()):
        build_cache(match_id, data_root, cache_dir)
    return pd.read_parquet(pp), pd.read_parquet(bp)


def load_parse_stats(match_id: str, cache_dir: str | Path = CACHE_DIR) -> dict:
    return json.loads((Path(cache_dir) / f"{match_id}_parse_stats.json").read_text())


def _build_cache_job(args: tuple[str, str, str]) -> tuple[str, dict]:
    return args[0], build_cache(*args)


def main() -> None:
    """Parse every positions file into the Parquet cache (in parallel)."""
    import argparse
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--force", action="store_true", help="rebuild existing caches")
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    jobs = [(m, args.data_root, args.cache_dir) for m in match_ids(args.data_root)
            if args.force or not (cache / f"{m}_parse_stats.json").exists()]
    with Pool(args.workers) as pool:
        for match_id, stats in pool.imap_unordered(_build_cache_job, jobs):
            print(match_id, stats, flush=True)


if __name__ == "__main__":
    main()
