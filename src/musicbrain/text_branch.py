"""Wire the trained trunk/VA head to the frozen-LLM text branch on real
DEAM clips (ROADMAP.md Phase 1 item 5, PROJECT_SPEC.md Section 6 sanity
check).

Runs a cached clip's fMRI trace through the trained model to get a
predicted VA trajectory, then feeds it to ``FrozenGenerationLLM`` via the
anchor-word soft-token interpolation (anchors.py/soft_prompt.py, both
already spot-checked in Phase 0 on hand-built trajectories -- this module
is what runs that same mechanism on the model's own real predictions for
the first time).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from musicbrain.anchors import AnchorSet, describe_trajectory
from musicbrain.soft_prompt import FrozenGenerationLLM
from musicbrain.fmri_cache import DEFAULT_CACHE_DIR, cache_clip_path

# "Show don't tell": ask for a vivid, figurative gestalt read of the arc,
# not a literal timestep-by-timestep narration of valence/arousal moving up
# and down. Few-shot exemplars (below) are what actually shifts a frozen,
# never-fine-tuned model's style -- the instruction alone was found (local
# validation run) to still default to generic "rollercoaster ride" phrasing.
INSTRUCTION = (
    "Describe the overall feeling of a piece of music in one vivid sentence, "
    "the way a person would describe a song's vibe to a friend -- using "
    "concrete imagery, metaphor, or scene-setting language where it feels "
    "natural. Do not use clinical language like \"emotional arc\", "
    "\"trajectory\", \"valence\", \"arousal\", or \"increases/decreases\". "
    "Base your description on the feeling trajectory given (each token is a "
    "moment in time, moving from the start of the clip to the end). Here are "
    "some examples:\n\n"
)
QUERY_PREFIX = "Trajectory:"
QUERY_SUFFIX = "\nDescription:"

# Hand-built VA trajectories + hand-written target-style sentences, in the
# same style as notebooks/phase0_spike3_anchor_interpolation.py's
# TRAJECTORIES dict. These are embedded as real soft-token blocks (using
# whatever anchor_set the caller passes) ahead of the actual query, not
# hard-coded text -- the standard lever for shifting style without training.
FEW_SHOT_EXEMPLARS: list[tuple[np.ndarray, str]] = [
    (
        np.array(
            [
                [0.6, -0.7],
                [0.5, -0.3],
                [-0.3, 0.6],
                [-0.7, 0.8],
                [0.4, -0.5],
            ]
        ),
        "It drifts in like a quiet fog, then the floor drops out into a "
        "jagged, frantic storm before settling back into an uneasy hush.",
    ),
    (
        np.array([[0.62, -0.2], [0.6, -0.18], [0.58, -0.22], [0.61, -0.2], [0.6, -0.19]]),
        "A warm, unhurried afternoon glow that never breaks stride, like "
        "sunlight pooling on a porch.",
    ),
    (
        np.array([[-0.6, 0.7], [-0.65, 0.75], [-0.6, 0.72], [-0.62, 0.7], [-0.58, 0.68]]),
        "Restless and on edge the whole way through, like pacing a hallway "
        "that never ends.",
    ),
]

_SENTENCE_END = re.compile(r"[.!?]")


def _trim_to_first_sentence(text: str) -> str:
    """Cut generated text at the first sentence-ending punctuation.

    A fixed ``max_new_tokens`` budget occasionally cuts a sentence off
    mid-thought; this keeps output to one clean sentence when the model
    finished one before hitting the token budget.
    """
    match = _SENTENCE_END.search(text)
    return text[: match.end()] if match else text


def _build_few_shot_prefix(llm: FrozenGenerationLLM, anchor_set: AnchorSet, temperature: float) -> torch.Tensor:
    """Assemble the few-shot block of embeddings: for each hand-built
    exemplar, ``QUERY_PREFIX`` + its interpolated soft tokens + its
    hand-written target-style sentence. Lives here (not in soft_prompt.py)
    so ``FrozenGenerationLLM`` stays generic and unaware of this project's
    prompt design -- it only supplies the generic embedding building blocks
    (``embed_text``/``interpolate_soft_tokens``).
    """
    parts = []
    for traj, sentence in FEW_SHOT_EXEMPLARS:
        soft = llm.interpolate_soft_tokens(traj, anchor_set, temperature).unsqueeze(0).to(llm.model.dtype)
        parts.append(llm.embed_text(QUERY_PREFIX))
        parts.append(soft)
        parts.append(llm.embed_text(f"{QUERY_SUFFIX} {sentence}\n\n"))
    return torch.cat(parts, dim=1)


def _generate_vibe_sentence(
    llm: FrozenGenerationLLM,
    anchor_set: AnchorSet,
    query_va: np.ndarray,
    temperature: float,
    max_new_tokens: int,
) -> str:
    query_soft = llm.interpolate_soft_tokens(query_va, anchor_set, temperature).unsqueeze(0).to(llm.model.dtype)
    inputs_embeds = torch.cat(
        [
            llm.embed_text(INSTRUCTION),
            _build_few_shot_prefix(llm, anchor_set, temperature),
            llm.embed_text(QUERY_PREFIX),
            query_soft,
            llm.embed_text(QUERY_SUFFIX),
        ],
        dim=1,
    )
    raw = llm.generate_from_embeds(inputs_embeds, max_new_tokens=max_new_tokens)
    return _trim_to_first_sentence(raw.strip())


def predict_va_trajectory(model, song_id: int, cache_dir: Path = DEFAULT_CACHE_DIR, device=None) -> np.ndarray:
    """Run the trained trunk+VA head over one cached clip's fMRI trace."""
    device = device or next(model.parameters()).device
    with np.load(cache_clip_path(song_id, cache_dir)) as npz:
        trace = npz["trace"].astype(np.float32)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(trace).to(device)
        return model(x).cpu().numpy()


def rank_by_dynamics(song_ids: list[int], cache_dir: Path = DEFAULT_CACHE_DIR) -> list[tuple[int, float]]:
    """Rank clips by how much their *true* VA trajectory moves around
    (mean per-axis std), most dynamic first -- used to pick a genuinely
    sharp-swing clip and a genuinely flat clip for the spec's sanity
    check, independent of what the (possibly still-undertrained) model
    predicts.
    """
    scored = []
    for song_id in song_ids:
        with np.load(cache_clip_path(song_id, cache_dir)) as npz:
            valence, arousal = npz["valence"], npz["arousal"]
        spread = float(np.std(valence) + np.std(arousal))
        scored.append((song_id, spread))
    return sorted(scored, key=lambda t: t[1], reverse=True)


@dataclass
class SanityCheckResult:
    dynamic_song_id: int
    flat_song_id: int
    dynamic_sentence: str
    flat_sentence: str
    dynamic_fallback: list[str]
    flat_fallback: list[str]


def generate_sentence(
    model,
    llm: FrozenGenerationLLM,
    anchor_set: AnchorSet,
    song_id: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    temperature: float = 0.3,
    max_new_tokens: int = 80,
) -> str:
    query_va = predict_va_trajectory(model, song_id, cache_dir)
    return _generate_vibe_sentence(llm, anchor_set, query_va, temperature, max_new_tokens)


def dynamic_vs_flat_sanity_check(
    model,
    llm: FrozenGenerationLLM,
    anchor_set: AnchorSet,
    held_out_song_ids: list[int],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    temperature: float = 0.3,
) -> SanityCheckResult:
    """PROJECT_SPEC.md Section 6: compare a sharp-VA-swing held-out clip
    against a flat one; the generated sentences should differ in a way
    that reflects the trajectory difference, not just overall tone.
    """
    ranked = rank_by_dynamics(held_out_song_ids, cache_dir)
    dynamic_song_id = ranked[0][0]
    flat_song_id = ranked[-1][0]

    dynamic_va = predict_va_trajectory(model, dynamic_song_id, cache_dir)
    flat_va = predict_va_trajectory(model, flat_song_id, cache_dir)

    dynamic_sentence = _generate_vibe_sentence(llm, anchor_set, dynamic_va, temperature, max_new_tokens=80)
    flat_sentence = _generate_vibe_sentence(llm, anchor_set, flat_va, temperature, max_new_tokens=80)

    return SanityCheckResult(
        dynamic_song_id=dynamic_song_id,
        flat_song_id=flat_song_id,
        dynamic_sentence=dynamic_sentence,
        flat_sentence=flat_sentence,
        dynamic_fallback=describe_trajectory(dynamic_va, anchor_set, temperature=temperature),
        flat_fallback=describe_trajectory(flat_va, anchor_set, temperature=temperature),
    )
