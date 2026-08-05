"""Step 1 -- train the trunk + VA head (PROJECT_SPEC.md Section 5).

One loss: Huber regression on (valence, arousal), applied per window.
Windows are independent training examples (model.py has no cross-window
mixing), so this is a plain per-window regression problem -- clip
identity only matters for the train/val *split* (so held-out correlation
in verify.py isn't leaked by seeing other windows from the same song)
and for grouping windows back into per-clip trajectories at evaluation
time, not for anything in the training loop itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from musicbrain.model import FmriTrunkVA, TemporalFmriTrunkVA
from musicbrain.fmri_cache import DEFAULT_CACHE_DIR, cache_clip_path


def split_song_ids(
    song_ids: list[int], val_frac: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Split by song id (clip), not by window, so held-out correlation
    (Step 2) is measured on clips the model never saw any window of."""
    rng = random.Random(seed)
    ids = list(song_ids)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    return ids[n_val:], ids[:n_val]


class WindowDataset(Dataset):
    """Flattened (trace_window[P], va[2]) pairs across a set of cached clips."""

    def __init__(self, song_ids: list[int], cache_dir: Path = DEFAULT_CACHE_DIR):
        self.song_ids = list(song_ids)
        traces, va, song_id_per_window, time_per_window = [], [], [], []
        for song_id in self.song_ids:
            with np.load(cache_clip_path(song_id, cache_dir)) as npz:
                trace = npz["trace"]
                valence = npz["valence"]
                arousal = npz["arousal"]
                times = npz["times_s"]
            traces.append(trace)
            va.append(np.stack([valence, arousal], axis=1))
            song_id_per_window.extend([song_id] * len(trace))
            time_per_window.extend(times.tolist())

        self.traces = np.concatenate(traces, axis=0).astype(np.float32)
        self.va = np.concatenate(va, axis=0).astype(np.float32)
        self.song_id_per_window = np.array(song_id_per_window, dtype=np.int64)
        self.time_per_window = np.array(time_per_window, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.traces[idx]), torch.from_numpy(self.va[idx])


class ClipSequenceDataset(Dataset):
    """One item per *clip* (not per window): the whole clip's
    ``(trace[T, P], va[T, 2])`` sequence, in time order, ``T`` varying by
    clip. Feeds ``TemporalFmriTrunkVA``/``train_step1_temporal`` -- the
    flattened, shuffled-window ``WindowDataset`` above throws away
    ordering entirely, which is fine for the i.i.d. baseline but would
    defeat the point of a model meant to use temporal context.
    """

    def __init__(self, song_ids: list[int], cache_dir: Path = DEFAULT_CACHE_DIR):
        self.song_ids = list(song_ids)
        self.clips: list[tuple[int, np.ndarray, np.ndarray]] = []
        for song_id in self.song_ids:
            with np.load(cache_clip_path(song_id, cache_dir)) as npz:
                trace = npz["trace"].astype(np.float32)
                va = np.stack([npz["valence"], npz["arousal"]], axis=1).astype(np.float32)
            self.clips.append((song_id, trace, va))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        _, trace, va = self.clips[idx]
        return torch.from_numpy(trace), torch.from_numpy(va)


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_valence_r: list[float] = field(default_factory=list)
    val_arousal_r: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    best_val_corr: float = float("-inf")  # mean(valence_r, arousal_r) at best_epoch


def train_step1(
    train_ds: WindowDataset,
    val_ds: WindowDataset,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    seed: int = 0,
    patience: int | None = None,
    verbose: bool = True,
) -> tuple[FmriTrunkVA, TrainHistory]:
    """Train the trunk + VA head, restoring the best-held-out-correlation
    epoch's weights before returning.

    Selection was originally by val_loss, but a full-scale run picked
    epoch 2 of 30 as "best" -- suspicious, because Huber/MSE-style loss has
    a specific failure mode here: a model can minimize it early just by
    predicting something close to the per-window mean (safe, low-variance,
    low-error) without capturing any real dynamic signal at all. Loss
    doesn't distinguish "tracks the true trajectory" from "conservatively
    hugs the average" -- correlation does, and correlation (not loss) is
    what ROADMAP.md's exit criteria and verify.py's held_out_correlation
    actually care about. So selection/early-stopping now use
    mean(valence_r, arousal_r) on the held-out set instead, computed once
    per epoch from the same validation forward pass already being done for
    val_loss (no extra compute). val_loss is still tracked/logged for
    comparison -- if loss-best and correlation-best land on very different
    epochs, that itself is informative about whether loss was ever a good
    proxy here.

    On the full-scale DEAM run (101,823 train windows), val_loss reliably
    bottomed out within the first ~5 epochs and then plateaued/got noisier
    for the remaining 25 while train_loss kept dropping -- ordinary
    overfitting once real data volume is large enough to no longer be the
    bottleneck. Pass *patience* (epochs without a new best correlation
    before stopping) to also cut training short instead of always running
    all *epochs* -- worth setting on Colab given the above.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    n_parcels = train_ds.traces.shape[1]
    model = FmriTrunkVA(n_parcels=n_parcels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.HuberLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    history = TrainHistory()

    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_best = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(x)
        train_loss = running / len(train_ds)

        model.eval()
        running = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                running += loss_fn(pred, y).item() * len(x)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(y.cpu().numpy())
        val_loss = running / len(val_ds)
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        valence_r = float(np.corrcoef(val_preds[:, 0], val_trues[:, 0])[0, 1])
        arousal_r = float(np.corrcoef(val_preds[:, 1], val_trues[:, 1])[0, 1])
        mean_corr = float(np.nanmean([valence_r, arousal_r]))

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_valence_r.append(valence_r)
        history.val_arousal_r.append(arousal_r)
        if verbose:
            print(
                f"[train] epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_valence_r={valence_r:+.3f}  val_arousal_r={arousal_r:+.3f}"
            )

        improved = not np.isnan(mean_corr) and mean_corr > history.best_val_corr
        if improved:
            history.best_val_corr = mean_corr
            history.best_val_loss = val_loss
            history.best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if patience is not None and epochs_since_best >= patience:
                if verbose:
                    print(f"[train] no val correlation improvement for {patience} epochs, stopping early at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(
                f"[train] restored best checkpoint: epoch {history.best_epoch}  "
                f"val_loss={history.best_val_loss:.4f}  mean_val_corr={history.best_val_corr:+.3f}"
            )

    return model, history


def collate_clips(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length ``(trace[T_i, P], va[T_i, 2])`` clips
    to the batch's own max ``T``, for ``DataLoader(..., collate_fn=...)``.

    Returns ``(x, y, lengths, mask)``: ``lengths`` (each clip's real,
    unpadded window count) is what ``TemporalFmriTrunkVA.forward`` needs
    for ``pack_padded_sequence`` so the GRU isn't corrupted by padding;
    ``mask`` (bool, ``(B, T_max)``) marks real vs. padded positions so
    loss/correlation computation can exclude the padding.
    """
    traces, vas = zip(*batch)
    lengths = torch.tensor([t.shape[0] for t in traces], dtype=torch.long)
    T_max = int(lengths.max())
    B, P = len(traces), traces[0].shape[1]

    x = torch.zeros(B, T_max, P, dtype=traces[0].dtype)
    y = torch.zeros(B, T_max, 2, dtype=vas[0].dtype)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, (trace, va) in enumerate(zip(traces, vas)):
        t = trace.shape[0]
        x[i, :t] = trace
        y[i, :t] = va
        mask[i, :t] = True
    return x, y, lengths, mask


def train_step1_temporal(
    train_ds: ClipSequenceDataset,
    val_ds: ClipSequenceDataset,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gru_hidden: int = 128,
    grad_clip_norm: float | None = 5.0,
    device: str | None = None,
    seed: int = 0,
    patience: int | None = None,
    verbose: bool = True,
) -> tuple[TemporalFmriTrunkVA, TrainHistory]:
    """Train ``TemporalFmriTrunkVA`` with proper cross-clip batching.

    Mirrors ``train_step1``'s overall shape (correlation-based checkpoint
    selection, optional early stopping, same ``TrainHistory``). A first
    prototype trained one clip per gradient step (no batching) and
    underperformed the i.i.d. baseline on both pooled and within-clip
    correlation, with a training curve that stopped after only 11 epochs
    and at least one clip collapsing to a constant (zero-variance)
    prediction -- signs of an under-optimized, noisy training regime
    (batch size effectively 1) rather than proof the architecture itself
    doesn't help. This version batches multiple variable-length clips
    together via padding + masking (``collate_clips``,
    ``pack_padded_sequence`` inside the model) for a fairer optimization
    comparison, plus gradient clipping (*grad_clip_norm*) since GRUs over
    long sequences are prone to exactly the instability observed before.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    n_parcels = train_ds[0][0].shape[-1]
    model = TemporalFmriTrunkVA(n_parcels=n_parcels, gru_hidden=gru_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.HuberLoss(reduction="none")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_clips)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_clips)
    history = TrainHistory()
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_best = 0

    for epoch in range(epochs):
        model.train()
        running, n_windows = 0.0, 0
        for x, y, lengths, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(x, lengths)
            masked_loss = loss_fn(pred, y)[mask]  # (n_real_timesteps, 2), padding excluded
            loss = masked_loss.mean()
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            running += loss.item() * masked_loss.shape[0]
            n_windows += masked_loss.shape[0]
        train_loss = running / n_windows

        model.eval()
        running, n_windows = 0.0, 0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x, y, lengths, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                pred = model(x, lengths)
                masked_loss = loss_fn(pred, y)[mask]
                running += masked_loss.sum().item()
                n_windows += masked_loss.shape[0]
                val_preds.append(pred[mask].cpu().numpy())
                val_trues.append(y[mask].cpu().numpy())
        val_loss = running / n_windows
        val_preds_cat = np.concatenate(val_preds)
        val_trues_cat = np.concatenate(val_trues)
        valence_r = float(np.corrcoef(val_preds_cat[:, 0], val_trues_cat[:, 0])[0, 1])
        arousal_r = float(np.corrcoef(val_preds_cat[:, 1], val_trues_cat[:, 1])[0, 1])
        mean_corr = float(np.nanmean([valence_r, arousal_r]))

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_valence_r.append(valence_r)
        history.val_arousal_r.append(arousal_r)
        if verbose:
            print(
                f"[train-temporal] epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_valence_r={valence_r:+.3f}  val_arousal_r={arousal_r:+.3f}"
            )

        improved = not np.isnan(mean_corr) and mean_corr > history.best_val_corr
        if improved:
            history.best_val_corr = mean_corr
            history.best_val_loss = val_loss
            history.best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if patience is not None and epochs_since_best >= patience:
                if verbose:
                    print(f"[train-temporal] no val correlation improvement for {patience} epochs, stopping early at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(
                f"[train-temporal] restored best checkpoint: epoch {history.best_epoch}  "
                f"val_loss={history.best_val_loss:.4f}  mean_val_corr={history.best_val_corr:+.3f}"
            )

    return model, history


def save_checkpoint(model: FmriTrunkVA | TemporalFmriTrunkVA, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(path: Path, n_parcels: int = 400, device: str | None = None) -> FmriTrunkVA:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FmriTrunkVA(n_parcels=n_parcels).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_checkpoint_temporal(
    path: Path, n_parcels: int = 400, gru_hidden: int = 128, device: str | None = None
) -> TemporalFmriTrunkVA:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalFmriTrunkVA(n_parcels=n_parcels, gru_hidden=gru_hidden).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
