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


def predict_pooled_temporal(model, ds: ClipSequenceDataset, device=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    pred, true, _ = predict_pooled_temporal(model, val_ds, device)
    return HeldOutCorrelation(
        valence_r=float(np.corrcoef(pred[:, 0], true[:, 0])[0, 1]),
        arousal_r=float(np.corrcoef(pred[:, 1], true[:, 1])[0, 1]),
        n_windows=len(true),
    )


def within_clip_tracking_temporal(
    model, val_ds: ClipSequenceDataset, device=None, min_windows: int = 4
) -> list[ClipTrackingResult]:
    pred, true, song_id_per_window = predict_pooled_temporal(model, val_ds, device)
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


def predict_pooled(model, val_ds: WindowDataset, device=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same flattened ``(pred, true, song_id_per_window)`` shape as
    ``predict_pooled_temporal``, for the baseline (i.i.d.) model -- lets
    diagnostics like ``compare_predicted_variance`` work identically
    regardless of which model produced the predictions."""
    pred = _predict(model, val_ds, device)
    return pred, val_ds.va, val_ds.song_id_per_window


def lag1_autocorrelation(values: np.ndarray) -> float:
    """Pearson correlation between ``values[:-1]`` and ``values[1:]`` --
    lag-1 autocorrelation, a simple, model-free measure of how smoothly/
    predictably a sequence evolves window-to-window. Needs at least 3
    points to be numerically meaningful."""
    if len(values) < 3:
        return float("nan")
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


@dataclass
class LabelAutocorrelation:
    mean_valence_autocorr: float
    mean_arousal_autocorr: float
    n_clips: int


def true_label_autocorrelation(ds: ClipSequenceDataset, min_windows: int = 4) -> LabelAutocorrelation:
    """Mean lag-1 autocorrelation of the *true* valence/arousal labels,
    per clip, averaged -- tests whether one axis genuinely has more
    exploitable local temporal structure than the other in the real DEAM
    labels themselves, entirely independent of any trained model. If
    arousal's true labels autocorrelate more strongly than valence's,
    that alone would explain a temporal-mixing model helping arousal more
    than valence, regardless of architecture/training details.
    """
    valence_acs, arousal_acs = [], []
    for _, _, va in ds.clips:
        if len(va) < min_windows:
            continue
        v_ac, a_ac = lag1_autocorrelation(va[:, 0]), lag1_autocorrelation(va[:, 1])
        if not np.isnan(v_ac):
            valence_acs.append(v_ac)
        if not np.isnan(a_ac):
            arousal_acs.append(a_ac)
    return LabelAutocorrelation(
        mean_valence_autocorr=float(np.mean(valence_acs)) if valence_acs else float("nan"),
        mean_arousal_autocorr=float(np.mean(arousal_acs)) if arousal_acs else float("nan"),
        n_clips=len(valence_acs),
    )


@dataclass
class VarianceComparison:
    mean_true_valence_std: float
    mean_pred_valence_std: float
    mean_true_arousal_std: float
    mean_pred_arousal_std: float
    n_clips: int

    @property
    def valence_std_ratio(self) -> float:
        """pred/true valence std, averaged per clip -- well below 1 means
        the model is predicting a flatter (over-smoothed) trajectory than
        the true signal actually is on this axis."""
        return self.mean_pred_valence_std / self.mean_true_valence_std

    @property
    def arousal_std_ratio(self) -> float:
        return self.mean_pred_arousal_std / self.mean_true_arousal_std


def compare_predicted_variance(
    pred: np.ndarray, true: np.ndarray, song_id_per_window: np.ndarray, min_windows: int = 4
) -> VarianceComparison:
    """Per-clip predicted-vs-true standard deviation, averaged across held-
    out clips -- diagnoses *model* over-smoothing (as opposed to
    ``true_label_autocorrelation``'s *data*-side check): a model producing
    a too-flat prediction on a given axis (low variance relative to how
    much the true signal actually moves) will show a low pred/true std
    ratio there, which directly explains weak within-clip correlation on
    that axis independent of whether the true labels had structure to
    exploit at all. Takes the same flattened arrays
    ``predict_pooled``/``predict_pooled_temporal`` return, so it works
    identically for either model.
    """
    true_v, pred_v, true_a, pred_a = [], [], [], []
    for song_id in sorted(set(song_id_per_window.tolist())):
        mask = song_id_per_window == song_id
        if mask.sum() < min_windows:
            continue
        true_v.append(np.std(true[mask, 0]))
        pred_v.append(np.std(pred[mask, 0]))
        true_a.append(np.std(true[mask, 1]))
        pred_a.append(np.std(pred[mask, 1]))
    return VarianceComparison(
        mean_true_valence_std=float(np.mean(true_v)),
        mean_pred_valence_std=float(np.mean(pred_v)),
        mean_true_arousal_std=float(np.mean(true_a)),
        mean_pred_arousal_std=float(np.mean(pred_a)),
        n_clips=len(true_v),
    )
