"""Deterministic anchor-word interpolation (PROJECT_SPEC.md Section 3.3).

No trainable weights: anchor (valence, arousal) coordinates come from the
NRC-VAD lexicon, anchor embeddings come from the frozen generation LLM's
own vocabulary, and a window's predicted (v, a) is mapped to a soft token
via a softmax kernel over distance in VA space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from musicbrain.lexicon import NRCVAD

# The 8-word circumplex set used by EmotionCaps' construction pipeline
# (PROJECT_SPEC.md Section 3.3, step 1).
ANCHOR_WORDS = [
    "eventful",
    "uneventful",
    "pleasant",
    "unpleasant",
    "exciting",
    "boring",
    "quiet",
    "chaotic",
]


@dataclass
class AnchorSet:
    words: list[str]
    va: np.ndarray  # (n_anchors, 2), columns = [valence, arousal]

    @classmethod
    def from_lexicon(cls, lexicon: NRCVAD, words: list[str] | None = None) -> "AnchorSet":
        words = list(words) if words is not None else list(ANCHOR_WORDS)
        va = np.array([lexicon.va(w) for w in words], dtype=np.float64)
        return cls(words=words, va=va)


def interpolation_weights(
    query_va: np.ndarray, anchor_va: np.ndarray, temperature: float = 0.3
) -> np.ndarray:
    """Softmax-over-negative-distance kernel from each query point to the anchors.

    Parameters
    ----------
    query_va:
        (n_windows, 2) array of predicted (valence, arousal) points.
    anchor_va:
        (n_anchors, 2) array of anchor (valence, arousal) coordinates.
    temperature:
        Softmax temperature. Smaller -> more concentrated on the nearest
        anchor(s); larger -> smoother blend. This is the one tunable
        parameter the spec allows (Section 3.3), and it is not fit against
        any text target.

    Returns
    -------
    weights: (n_windows, n_anchors) array, each row sums to 1.
    """
    query_va = np.atleast_2d(query_va)
    dists = np.linalg.norm(query_va[:, None, :] - anchor_va[None, :, :], axis=-1)
    logits = -dists / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights


def describe_trajectory(
    query_va: np.ndarray, anchor_set: AnchorSet, temperature: float = 0.3
) -> list[str]:
    """Plain-text fallback (spec Section 3.3): the single nearest anchor word
    per window, for a human-readable trajectory description like
    "calm -> tense -> release" without relying on the soft-token mechanism.
    """
    weights = interpolation_weights(query_va, anchor_set.va, temperature=temperature)
    nearest = weights.argmax(axis=-1)
    return [anchor_set.words[i] for i in nearest]
