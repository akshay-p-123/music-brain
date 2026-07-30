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
    best_epoch: int = -1
    best_val_loss: float = float("inf")


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
    """Train the trunk + VA head, restoring the best-val-loss epoch's
    weights before returning.

    On the full-scale DEAM run (101,823 train windows), val_loss reliably
    bottomed out within the first ~5 epochs and then plateaued/got noisier
    for the remaining 25 while train_loss kept dropping -- ordinary
    overfitting once real data volume is large enough to no longer be the
    bottleneck. Training to a fixed epoch count and keeping only the final
    weights would silently return a measurably worse checkpoint than one
    from partway through training, so this tracks the best val_loss seen and
    reloads those weights at the end. Pass *patience* (epochs without a new
    best before stopping) to also cut training short instead of always
    running all *epochs* -- worth setting on Colab given the above.
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
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                running += loss_fn(pred, y).item() * len(x)
        val_loss = running / len(val_ds)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        if verbose:
            print(f"[train] epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if patience is not None and epochs_since_best >= patience:
                if verbose:
                    print(f"[train] no val_loss improvement for {patience} epochs, stopping early at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"[train] restored best checkpoint: epoch {history.best_epoch}  val_loss={history.best_val_loss:.4f}")

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
