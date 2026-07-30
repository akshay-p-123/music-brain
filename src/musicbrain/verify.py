"""verification checkpoint, not a training step

Two checks:

- Held-out correlation, pooled over all windows.
- Within-clip dynamic tracking: correlation between predicted and true
  VA *trajectories* computed separately per held-out clip, then averaged.
  This is the check that distinguishes "the model learned the windowed
  trajectory" from "the model learned the clip average and repeats it" --
  the latter can still score well on the pooled metric above if clip
  averages vary more than within-clip dynamics do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from musicbrain.train import ClipSequenceDataset, WindowDataset


@dataclass
class HeldOutCorrelation:
    valence_r: float
    arousal_r: float
    n_windows: int


def _predict(model, ds: WindowDataset, device=None) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(ds.traces).to(device)
        return model(x).cpu().numpy()


def held_out_correlation(model, val_ds: WindowDataset, device=None) -> HeldOutCorrelation:
    pred = _predict(model, val_ds, device)
    true = val_ds.va
    return HeldOutCorrelation(
        valence_r=float(np.corrcoef(pred[:, 0], true[:, 0])[0, 1]),
        arousal_r=float(np.corrcoef(pred[:, 1], true[:, 1])[0, 1]),
        n_windows=len(true),
    )


@dataclass
class ClipTrackingResult:
    song_id: int
    n_windows: int
    valence_r: float | None
    arousal_r: float | None


def within_clip_tracking(
    model, val_ds: WindowDataset, device=None, min_windows: int = 4
) -> list[ClipTrackingResult]:
    """Correlation is computed per clip, on that clip's own windows only.
    Clips with fewer than ``min_windows`` are reported with ``None``
    rather than a numerically unstable correlation from a handful of points.
    """
    pred = _predict(model, val_ds, device)
    results = []
    for song_id in sorted(set(val_ds.song_id_per_window.tolist())):
        mask = val_ds.song_id_per_window == song_id
        n = int(mask.sum())
        if n < min_windows:
            results.append(ClipTrackingResult(song_id, n, None, None))
            continue
        true_v, true_a = val_ds.va[mask, 0], val_ds.va[mask, 1]
        pred_v, pred_a = pred[mask, 0], pred[mask, 1]
        # A constant true or predicted trajectory makes corrcoef return
        # NaN by construction (zero variance) -- not a bug, filtered out
        # in summarize_tracking rather than here so callers can still see it.
        v_r = float(np.corrcoef(pred_v, true_v)[0, 1])
        a_r = float(np.corrcoef(pred_a, true_a)[0, 1])
        results.append(ClipTrackingResult(song_id, n, v_r, a_r))
    return results


def summarize_tracking(results: list[ClipTrackingResult]) -> dict[str, float]:
    v = [r.valence_r for r in results if r.valence_r is not None and not np.isnan(r.valence_r)]
    a = [r.arousal_r for r in results if r.arousal_r is not None and not np.isnan(r.arousal_r)]
    return {
        "mean_valence_r": float(np.mean(v)) if v else float("nan"),
        "mean_arousal_r": float(np.mean(a)) if a else float("nan"),
        "n_clips_scored": len(v),
        "n_clips_total": len(results),
    }


def _predict_temporal(model, ds: ClipSequenceDataset, device=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``TemporalFmriTrunkVA`` clip-by-clip -- its forward pass needs a
    whole clip's ordered window sequence at once (see model.py), unlike
    ``WindowDataset``'s flat, shuffled pool. Returns pooled
    ``(preds, trues, song_id_per_window)`` in the same flattened shape
    ``held_out_correlation``/``within_clip_tracking`` already use, so the
    *_temporal variants below reuse identical correlation logic and the
    same ``HeldOutCorrelation``/``ClipTrackingResult`` dataclasses --
    numbers from both models are directly comparable.
    """
    device = device or next(model.parameters()).device
    model.eval()
    preds, trues, song_ids = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            song_id = ds.song_ids[i]
            x, y = ds[i]
            pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            preds.append(pred)
            trues.append(y.numpy())
            song_ids.extend([song_id] * len(pred))
    return np.concatenate(preds), np.concatenate(trues), np.array(song_ids, dtype=np.int64)


def held_out_correlation_temporal(model, val_ds: ClipSequenceDataset, device=None) -> HeldOutCorrelation:
    pred, true, _ = _predict_temporal(model, val_ds, device)
    return HeldOutCorrelation(
        valence_r=float(np.corrcoef(pred[:, 0], true[:, 0])[0, 1]),
        arousal_r=float(np.corrcoef(pred[:, 1], true[:, 1])[0, 1]),
        n_windows=len(true),
    )


def within_clip_tracking_temporal(
    model, val_ds: ClipSequenceDataset, device=None, min_windows: int = 4
) -> list[ClipTrackingResult]:
    pred, true, song_id_per_window = _predict_temporal(model, val_ds, device)
    results = []
    for song_id in sorted(set(song_id_per_window.tolist())):
        mask = song_id_per_window == song_id
        n = int(mask.sum())
        if n < min_windows:
            results.append(ClipTrackingResult(song_id, n, None, None))
            continue
        v_r = float(np.corrcoef(pred[mask, 0], true[mask, 0])[0, 1])
        a_r = float(np.corrcoef(pred[mask, 1], true[mask, 1])[0, 1])
        results.append(ClipTrackingResult(song_id, n, v_r, a_r))
    return results
