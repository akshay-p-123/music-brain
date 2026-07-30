"""Vibe-descriptor anchor vocabulary (PROJECT_SPEC.md Section 3.3).

Replaces the abstract 8-word EmotionCaps circumplex set (``anchors.ANCHOR_WORDS``)
with MTG-Jamendo's mood/theme tag vocabulary -- real single-concept mood/scene
words a listener would actually reach for ("melancholic", "psychedelic-adjacent
"dreamy", ...) -- while staying VA-groundable: coordinates come from the NRC-VAD
lexicon where covered, and a one-time, cached LLM placement for the handful of
tags NRC-VAD doesn't have an entry for.

This only pulls the small mood/theme tag-name file from MTG-Jamendo's GitHub
repo, not any part of the ~18.5k-track audio dataset itself -- that's a
separate, still-open Phase 2 training-data decision (see ROADMAP.md), not
what this module is for.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import requests

from musicbrain.anchors import AnchorSet
from musicbrain.lexicon import NRCVAD

MOODTHEME_URL = (
    "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/tags/moodtheme.txt"
)
DEFAULT_LEXICON_DIR = Path(__file__).resolve().parents[2] / "data" / "lexicons"
MOODTHEME_FILENAME = "mtg_jamendo_moodtheme_tags.txt"
LLM_ESTIMATE_FILENAME = "mtg_jamendo_va_llm_estimated.json"

# Real NRC-VAD entries used as few-shot in-context calibration for the LLM
# fallback, so it places uncovered words *relative to* real lexicon values
# rather than cold-guessing a scale.
_FEW_SHOT_CALIBRATION_WORDS = ["happy", "sad", "calm", "chaotic", "boring", "exciting"]


def fetch_moodtheme_tags(dest_dir: Path = DEFAULT_LEXICON_DIR, force: bool = False) -> list[str]:
    """Download and cache MTG-Jamendo's mood/theme tag vocabulary (59 tags)."""
    dest_dir = Path(dest_dir)
    dest_path = dest_dir / MOODTHEME_FILENAME
    if dest_path.exists() and not force:
        return _read_tags(dest_path)

    dest_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(MOODTHEME_URL, timeout=30)
    response.raise_for_status()
    dest_path.write_text(response.text, encoding="utf-8")
    return _read_tags(dest_path)


_TAG_NAMESPACE_PREFIX = "mood/theme---"


def _read_tags(path: Path) -> list[str]:
    """Read tag names, stripping MTG-Jamendo's ``mood/theme---`` namespace
    prefix -- the raw file lists full tag IDs (``mood/theme---calm``), not
    bare words, and NRC-VAD/the LLM fallback both key on the bare word.
    """
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [line.removeprefix(_TAG_NAMESPACE_PREFIX) for line in lines]


# Cheap, dependency-free morphological fallbacks: NRC-VAD annotates a fixed
# set of word forms/lemmas, not every inflection, so a tag can be missing
# while its lemma is already scored (e.g. "inspiring" is absent but "inspire"
# is present, 0.836/0.254). Trying these first is strictly cheaper and more
# grounded than asking the LLM to re-estimate a word whose lemma the lexicon
# basically already answers.
_LEMMA_SUFFIX_RULES: list[tuple[str, str]] = [
    ("ing", "e"),  # inspiring -> inspire
    ("ing", ""),   # boring -> bor (rarely hits; harmless if absent too)
    ("ed", "e"),
    ("ed", ""),
    ("s", ""),
]


def _resolve_lemma(word: str, lexicon: NRCVAD) -> str | None:
    """Return the lexicon key to use for *word*: itself if directly covered,
    else a simple morphological lemma NRC-VAD covers instead, else None."""
    if word in lexicon:
        return word
    for suffix, replacement in _LEMMA_SUFFIX_RULES:
        if word.endswith(suffix):
            candidate = word[: -len(suffix)] + replacement
            if candidate in lexicon:
                return candidate
    return None


def split_lexicon_coverage(tags: list[str], lexicon: NRCVAD) -> tuple[dict[str, str], list[str]]:
    """Partition *tags* into (tag -> lexicon key covering it, directly or via
    a simple lemma) and the remaining list needing the LLM fallback."""
    resolved: dict[str, str] = {}
    uncovered: list[str] = []
    for tag in tags:
        key = _resolve_lemma(tag, lexicon)
        if key is not None:
            resolved[tag] = key
        else:
            uncovered.append(tag)
    return resolved, uncovered


def _parse_json_va(raw: str, expected_words: list[str]) -> dict[str, tuple[float, float]]:
    """Extract the JSON object mapping word -> [valence, arousal].

    A greedy-decoded 1.5B instruct model sometimes emits a scratch/placeholder
    object before its real (code-fenced) answer, so this can't just span the
    first ``{`` to the last ``}`` -- instead it scans all flat (non-nested)
    ``{...}`` candidates and takes the *last* one that both parses and has
    every expected word, which is reliably the model's final answer.
    """
    candidates = re.findall(r"\{[^{}]*\}", raw)
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if all(w in parsed for w in expected_words):
            return {w: (float(parsed[w][0]), float(parsed[w][1])) for w in expected_words}
    raise ValueError(f"No parseable JSON object covering {expected_words} found in LLM output: {raw!r}")


def estimate_va_via_llm(
    words: list[str],
    llm,
    lexicon: NRCVAD,
    cache_path: Path | None = None,
) -> dict[str, tuple[float, float]]:
    """One-time, deterministic LLM placement of *words* onto the
    valence-arousal circumplex, for tags NRC-VAD doesn't cover. Cached to
    *cache_path* so a given word is only ever placed once.
    """
    cache_path = Path(cache_path) if cache_path is not None else DEFAULT_LEXICON_DIR / LLM_ESTIMATE_FILENAME
    cached: dict[str, tuple[float, float]] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = {k: (v[0], v[1]) for k, v in json.load(f).items()}

    missing = [w for w in words if w not in cached]
    if missing:
        examples = "\n".join(
            f'- "{w}": valence={lexicon.va(w)[0]:.2f}, arousal={lexicon.va(w)[1]:.2f}'
            for w in _FEW_SHOT_CALIBRATION_WORDS
        )
        prompt = (
            "You are placing words onto a valence-arousal emotion circumplex.\n"
            "Valence ranges from -1 (very negative) to 1 (very positive).\n"
            "Arousal ranges from -1 (very calm/sleepy) to 1 (very excited/energetic).\n\n"
            f"Examples of real placements:\n{examples}\n\n"
            "Estimate valence and arousal for each of the following words, on "
            "the same scale, and respond with ONLY a JSON object mapping each "
            f"word to a [valence, arousal] pair:\n{json.dumps(missing)}\n\nJSON:"
        )
        raw = llm.generate_text(prompt, temperature=0.0, max_new_tokens=200)
        cached.update(_parse_json_va(raw, missing))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cached, f, indent=2)

    return {w: cached[w] for w in words}


def build_vibe_anchor_set(lexicon: NRCVAD, llm, dest_dir: Path = DEFAULT_LEXICON_DIR) -> AnchorSet:
    """Build the MTG-Jamendo-mood/theme-tag anchor set: NRC-VAD lookup where
    covered, one-time cached LLM placement for the rest. Returns the same
    ``AnchorSet`` dataclass ``anchors.py`` already defines -- its
    interpolation mechanics are vocabulary-agnostic, no changes needed there.
    """
    tags = fetch_moodtheme_tags(dest_dir)
    resolved, uncovered = split_lexicon_coverage(tags, lexicon)
    va = {tag: lexicon.va(key) for tag, key in resolved.items()}
    if uncovered:
        va.update(estimate_va_via_llm(uncovered, llm, lexicon, cache_path=dest_dir / LLM_ESTIMATE_FILENAME))
    words = list(va.keys())
    va_array = np.array([va[w] for w in words], dtype=np.float64)
    return AnchorSet(words=words, va=va_array)
