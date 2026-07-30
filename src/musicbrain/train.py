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

from musicbrain.model import FmriTrunkVA
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


def save_checkpoint(model: FmriTrunkVA, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(path: Path, n_parcels: int = 400, device: str | None = None) -> FmriTrunkVA:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FmriTrunkVA(n_parcels=n_parcels).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
