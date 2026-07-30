"""Manufacture the windowed, parcellated fMRI-trace + VA cache

For each clip: audio -> TRIBEv2 (frozen, audio-only) -> raw fsaverage5
vertex trace -> Schaefer-400 parcellation -> resampled onto a fixed
``window_s`` time grid, alongside the VA labels resampled onto the same
grid. Both signals are linearly interpolated onto one shared grid rather
than assuming they already share a sample rate, because TRIBEv2's native
output resolution (~1 prediction/second) does not match
DEAM's native 0.5s label resolution -- this is the "upsample the trace"
side of the still-open window-size decision (ROADMAP.md Phase 1 item 6).

Caches to local disk, one ``.npz`` per clip, keyed by song id -- skips
clips already cached so a run can be resumed after an interruption
(PROJECT_SPEC.md's Colab-ephemeral-disk practicalities apply here too).
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from musicbrain.datasets.deam import DeamClipVA
from musicbrain.parcellation import Schaefer400Parcellator
from musicbrain.tribev2_utils import predict_fmri_trace

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "deam"


@dataclass
class CacheResult:
    song_id: int
    n_windows: int
    raw_trace_hz: float  # observed TRIBEv2 output rate for this clip, for the window-size decision
    wall_time_s: float
    cache_path: Path


def _resample_columns(values: np.ndarray, src_times_s: np.ndarray, dst_times_s: np.ndarray) -> np.ndarray:
    """Linearly interpolate each column of values from src to dst times."""
    if values.ndim == 1:
        return np.interp(dst_times_s, src_times_s, values)
    return np.stack(
        [np.interp(dst_times_s, src_times_s, values[:, j]) for j in range(values.shape[1])],
        axis=1,
    )


def cache_clip_path(song_id: int, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{song_id}.npz"


def cache_clip(
    song_id: int,
    audio_path: Path,
    va: DeamClipVA,
    model,
    parcellator: Schaefer400Parcellator,
    window_s: float = 0.5,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> CacheResult:
    """Run one clip through TRIBEv2 -> Schaefer-400 -> shared window grid, cache it."""
    t0 = time.time()
    vertex_preds, raw_times_s = predict_fmri_trace(model, audio_path)
    parcel_preds = parcellator.aggregate(vertex_preds)  # (n_raw, 400)

    raw_trace_hz = (
        (len(raw_times_s) - 1) / (raw_times_s[-1] - raw_times_s[0])
        if len(raw_times_s) > 1
        else float("nan")
    )

    t_lo = max(va.times_s.min(), raw_times_s.min())
    t_hi = min(va.times_s.max(), raw_times_s.max())
    if t_hi <= t_lo:
        raise ValueError(
            f"song {song_id}: VA label range [{va.times_s.min()}, {va.times_s.max()}] "
            f"does not overlap TRIBEv2's predicted range [{raw_times_s.min()}, {raw_times_s.max()}]"
        )
    target_times_s = np.arange(t_lo, t_hi, window_s)

    trace = _resample_columns(parcel_preds, raw_times_s, target_times_s)
    valence = _resample_columns(va.valence, va.times_s, target_times_s)
    arousal = _resample_columns(va.arousal, va.times_s, target_times_s)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_clip_path(song_id, cache_dir)
    np.savez(
        out_path,
        song_id=song_id,
        window_s=window_s,
        times_s=target_times_s,
        trace=trace.astype(np.float32),
        valence=valence.astype(np.float32),
        arousal=arousal.astype(np.float32),
        raw_trace_hz=raw_trace_hz,
    )
    return CacheResult(
        song_id=song_id,
        n_windows=len(target_times_s),
        raw_trace_hz=raw_trace_hz,
        wall_time_s=time.time() - t0,
        cache_path=out_path,
    )


def build_cache(
    song_ids: list[int],
    audio_paths: dict[int, Path],
    va_by_song: dict[int, DeamClipVA],
    window_s: float = 0.5,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    device: str | None = None,
    num_workers: int | None = None,
    skip_existing: bool = True,
    verbose: bool = True,
    push_every: int | None = None,
    hub_repo_id: str | None = None,
    hub_token: str | None = None,
) -> list[CacheResult]:
    """Build the fMRI-trace + VA cache for a list of DEAM song ids, loading TRIBEv2 once.

    Resumable: song ids whose cache file already exists are skipped
    (loaded from disk to still report their stats) unless
    ``skip_existing=False`` -- pair with ``pull_cache_from_hub`` at the start
    of a Colab session to resume clips a previous, disconnected session
    already cached (this function only sees local disk; it doesn't know
    about the Hub on its own).

    If *hub_repo_id* is set, pushes the cache to that HF Hub dataset repo
    every *push_every* newly-processed clips (not counting skipped ones),
    plus once more at the end -- Colab sessions are ephemeral, so this is
    what makes a long run's progress survive a disconnect (ROADMAP.md's
    Colab-specific practicalities).
    """
    from musicbrain.tribev2_utils import load_tribev2_model

    parcellator = Schaefer400Parcellator()
    model = None
    results: list[CacheResult] = []
    n_since_push = 0

    for song_id in song_ids:
        out_path = cache_clip_path(song_id, cache_dir)
        if skip_existing and out_path.exists():
            with np.load(out_path) as npz:
                results.append(
                    CacheResult(
                        song_id=song_id,
                        n_windows=len(npz["times_s"]),
                        raw_trace_hz=float(npz["raw_trace_hz"]),
                        wall_time_s=0.0,
                        cache_path=out_path,
                    )
                )
            if verbose:
                print(f"[cache] song {song_id}: already cached, skipping")
            continue

        if model is None:
            if verbose:
                print("[cache] loading TRIBEv2 (audio-only)...")
            model = load_tribev2_model(device=device, num_workers=num_workers)

        result = cache_clip(
            song_id=song_id,
            audio_path=audio_paths[song_id],
            va=va_by_song[song_id],
            model=model,
            parcellator=parcellator,
            window_s=window_s,
            cache_dir=cache_dir,
        )
        results.append(result)
        n_since_push += 1
        if verbose:
            print(
                f"[cache] song {song_id}: {result.n_windows} windows, "
                f"raw trace ~{result.raw_trace_hz:.2f}Hz, {result.wall_time_s:.1f}s"
            )

        if push_every and hub_repo_id and n_since_push >= push_every:
            if verbose:
                print(f"[cache] pushing {n_since_push} newly-cached clip(s) to {hub_repo_id}...")
            push_cache_to_hub(cache_dir, hub_repo_id, token=hub_token)
            n_since_push = 0

    if push_every and hub_repo_id and n_since_push:
        if verbose:
            print(f"[cache] pushing final {n_since_push} clip(s) to {hub_repo_id}...")
        push_cache_to_hub(cache_dir, hub_repo_id, token=hub_token)

    return results


def pull_cache_from_hub(repo_id: str, cache_dir: Path = DEFAULT_CACHE_DIR, token: str | None = None) -> int:
    """Download any already-cached clips from *repo_id* into *cache_dir*.

    Resume support for Colab's ephemeral disk: call this before
    ``build_cache`` at the start of a (possibly new) session so clips a
    previous, disconnected session already pushed aren't reprocessed.
    Returns the number of files pulled; returns 0 (not an error) if
    *repo_id* doesn't exist yet -- that just means this is the first run.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_dir = snapshot_download(
            repo_id, repo_type="dataset", token=token, allow_patterns=["*.npz"]
        )
    except (RepositoryNotFoundError, EntryNotFoundError):
        return 0

    n_pulled = 0
    for src in Path(snapshot_dir).glob("*.npz"):
        dest = cache_dir / src.name
        if not dest.exists():
            shutil.copy(src, dest)
            n_pulled += 1
    return n_pulled


def push_cache_to_hub(cache_dir: Path, repo_id: str, token: str | None = None) -> str:
    """Push the local ``.npz`` cache to a Hugging Face Hub dataset repo.

    Colab sessions are ephemeral: this lets a long run checkpoint incrementally
    to somewhere that survives a disconnect, by calling this after every
    chunk of clips rather than only once at the end. Creates the repo if
    it doesn't exist yet.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    return api.upload_folder(
        folder_path=str(cache_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["*.npz"],
    )
