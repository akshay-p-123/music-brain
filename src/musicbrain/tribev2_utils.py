"""Load TRIBEv2 audio-only and run raw vertex-level inference on a clip.

Wraps the three non-obvious findings from Phase 0
(ROADMAP.md "Phase 0 -- De-risk the two unverified assumptions"):

1. ``TribeModel.get_events_dataframe()`` defaults to ``audio_only=False``,
   which shells out to whisperx/Whisper large-v3 to transcribe speech on
   every clip. Bypassed here by building the one-row events DataFrame by
   hand and calling ``get_audio_and_text_events(..., audio_only=True)``
   directly.
2. The released ``config.yaml`` hardcodes ``device: cuda`` for the audio
   feature extractor. Overridden via ``config_update`` when no GPU is
   available.
3. ``TribeModel.from_pretrained`` round-trips the HF repo id through
   ``pathlib.Path`` on Windows, mangling "facebook/tribev2" into
   "facebook\\tribev2". Worked around by pre-downloading via
   ``huggingface_hub`` (forward slashes preserved) and pointing
   ``from_pretrained`` at the local snapshot dir instead -- a no-op on
   Colab's Linux runtime, so this path is safe to leave in unconditionally.
"""

from __future__ import annotations

import pathlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd

if platform.system() == "Windows":
    # The released config.yaml was pickled on Linux and contains PosixPath
    # objects reconstructed via yaml.UnsafeLoader; PosixPath can't be
    # instantiated on Windows at all. No-op on Colab/Linux.
    pathlib.PosixPath = pathlib.WindowsPath

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_CACHE = REPO_ROOT / "external" / "hf_cache"
DEFAULT_CACHE_FOLDER = REPO_ROOT / "external" / "cache"
TRIBEV2_REPO_ID = "facebook/tribev2"


def load_tribev2_model(
    device: str | None = None,
    hf_cache_dir: Path = DEFAULT_HF_CACHE,
    cache_folder: Path = DEFAULT_CACHE_FOLDER,
    num_workers: int | None = None,
):
    """Load TRIBEv2's audio branch only (Wav2Vec-BERT 2.0), frozen, eval mode."""
    import torch
    from huggingface_hub import hf_hub_download
    from tribev2.demo_utils import TribeModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    hf_cache_dir = Path(hf_cache_dir)
    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        # Default num_workers is N_CPUS, which spawns that many
        # multiprocessing workers per predict() call (each re-importing
        # torch/sklearn/scipy). Windows' spawn-based multiprocessing reliably
        # triggered "paging file too small" DLL-load crashes doing this on
        # this machine, so it's forced serial there; Colab/Linux uses fork,
        # not spawn, so it isn't affected and can use real parallelism for
        # the full-scale (1802-clip) run instead of paying for it serially.
        num_workers = 0 if platform.system() == "Windows" else 4

    config_path = Path(
        hf_hub_download(TRIBEV2_REPO_ID, "config.yaml", cache_dir=str(hf_cache_dir))
    )
    hf_hub_download(TRIBEV2_REPO_ID, "best.ckpt", cache_dir=str(hf_cache_dir))

    model = TribeModel.from_pretrained(
        config_path.parent,
        cache_folder=str(cache_folder),
        config_update={
            "data.audio_feature.device": device,
            "data.num_workers": num_workers,
        },
    )
    return model


def predict_fmri_trace(model, audio_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Run audio-only inference on one clip.

    Returns
    -------
    vertex_preds : (n_segments, n_vertices) raw fsaverage5 vertex predictions.
    times_s : (n_segments,) start time in seconds of each predicted segment,
        relative to the start of the audio file (from TRIBEv2's own
        ``Segment.start``, not assumed from the nominal 1/TR spacing --
        Phase 0 measured ~1 prediction/second on a synthetic clip and this
        should be re-checked against a real clip's actual segment times).
    """
    from tribev2.demo_utils import get_audio_and_text_events

    event = {
        "type": "Audio",
        "filepath": str(audio_path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
    preds, segments = model.predict(events=events, verbose=False)
    times_s = np.array([s.start for s in segments], dtype=np.float64)
    return preds, times_s
