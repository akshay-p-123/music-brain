"""DEAM ingestion (ROADMAP.md Phase 1, PROJECT_SPEC.md Section 4).

DEAM: MediaEval Database for Emotional Analysis in Music
(cvml.unige.ch/databases/DEAM). 1802 clips (1744 45s excerpts + 58
full-length songs), Valence/arousal rated continuously
(roughly [-1, 1]) every 0.5s by >=5 crowdworkers and averaged; the first
15s of each clip is dropped from the label grid because raters need a
few seconds to settle before their rating is trusted -- so every clip's
label trajectory starts at t=15s, not t=0s.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://cvml.unige.ch/databases/DEAM"
DEFAULT_DEAM_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "deam"

# archive name -> a path that only exists once that archive is extracted
_ARCHIVES = {
    "DEAM_audio.zip": "MEMD_audio",
    "DEAM_Annotations.zip": "annotations",
    "metadata.zip": "metadata",
}

_DYNAMIC_ANNOT_DIR = "annotations/annotations averaged per song/dynamic (per second annotations)"
_SAMPLE_COL_RE = re.compile(r"sample_(\d+)ms")


def fetch_deam(dest_dir: Path = DEFAULT_DEAM_DIR, force: bool = False) -> Path:
    #Download and extract DEAM's audio + annotations + metadata archives. ~1.3GB total, audio dominates. 
    
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for archive_name, marker_dir in _ARCHIVES.items():
        if (dest_dir / marker_dir).exists() and not force:
            continue
        archive_path = dest_dir / archive_name
        with requests.get(f"{BASE_URL}/{archive_name}", timeout=300, stream=True) as resp:
            resp.raise_for_status()
            with open(archive_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
        archive_path.unlink()
    return dest_dir


@dataclass
class DeamClipVA:
    song_id: int
    times_s: np.ndarray  # (T,) seconds from clip start, ascending
    valence: np.ndarray  # (T,)
    arousal: np.ndarray  # (T,)


def _parse_dynamic_csv(path: Path) -> dict[int, dict[float, float]]:
    """One row per song -> {time_s: value}, dropping NaN (unrated) columns.
    out is a nested dict: {song_id: {time_s: value}}"""
    df = pd.read_csv(path)
    sample_cols = [c for c in df.columns if c != "song_id"]
    col_times_s = {c: int(_SAMPLE_COL_RE.match(c).group(1)) / 1000.0 for c in sample_cols}
    out: dict[int, dict[float, float]] = {}
    for row in df.itertuples(index=False):
        row_map = row._asdict()
        song_id = int(row_map["song_id"])
        out[song_id] = {
            col_times_s[c]: float(row_map[c])
            for c in sample_cols
            if pd.notna(row_map[c])
        }
    return out


def load_dynamic_va(dest_dir: Path = DEFAULT_DEAM_DIR) -> dict[int, DeamClipVA]:
    """Load the per-song averaged 0.5s-resolution VA trajectories.

    Valence and arousal are stored in separate wide CSVs whose sample-time
    columns don't perfectly match (arousal.csv has one extra column in the
    released files) -- aligned here by intersecting each song's actual
    {time_s: value} keys rather than assuming identical columns/positions.
    """
    dest_dir = Path(dest_dir)
    valence = _parse_dynamic_csv(dest_dir / _DYNAMIC_ANNOT_DIR / "valence.csv")
    arousal = _parse_dynamic_csv(dest_dir / _DYNAMIC_ANNOT_DIR / "arousal.csv")

    clips: dict[int, DeamClipVA] = {}
    for song_id in sorted(set(valence) & set(arousal)):
        v_map, a_map = valence[song_id], arousal[song_id]
        times = sorted(set(v_map) & set(a_map))
        if not times:
            continue
        clips[song_id] = DeamClipVA(
            song_id=song_id,
            times_s=np.array(times, dtype=np.float64),
            valence=np.array([v_map[t] for t in times], dtype=np.float64),
            arousal=np.array([a_map[t] for t in times], dtype=np.float64),
        )
    return clips


def deam_audio_path(song_id: int, dest_dir: Path = DEFAULT_DEAM_DIR) -> Path:
    return Path(dest_dir) / "MEMD_audio" / f"{song_id}.mp3"


def list_song_ids(dest_dir: Path = DEFAULT_DEAM_DIR) -> list[int]:
    """Song ids with both a dynamic VA trajectory and an audio file on disk."""
    va_ids = set(load_dynamic_va(dest_dir)) #song ids w/ valid VA's
    audio_ids = {
        int(p.stem) for p in (Path(dest_dir) / "MEMD_audio").glob("*.mp3") if p.stem.isdigit() #songs w/ audio file
    }
    return sorted(va_ids & audio_ids) #set intersection
