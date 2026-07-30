"""Cross-modal fMRI retrieval spike (ROADMAP.md "Still open" section) --
exploratory research code, not part of the shipped pipeline in Part A.

The idea being tested: TRIBEv2 is trimodal and predicts onto the *same*
fsaverage5 vertex space (-> Schaefer-400 parcels, via ``parcellation.py``)
regardless of whether the stimulus was audio, text, or video. That means a
song's predicted fMRI vector and a mood word's predicted fMRI vector are
literally comparable points in one shared space -- structurally sound, not
a hack. The open empirical question this spike is meant to answer is
whether proximity in that space actually tracks shared affective content,
or is dominated by modality-specific nuisance variance (auditory cortex
firing for any sound, language network firing for any text, regardless of
content). ``validate_against_deam`` below is the cheap, concrete way to
check that: for each DEAM window, retrieve the nearest mood word and
correlate *that word's* NRC-VAD valence/arousal against the window's *true*
DEAM label. If it holds up, this idea could replace or cross-check the
NRC-VAD-anchor mechanism in ``vibe_lexicon.py``/``text_branch.py``.

**Real blocker, not yet resolved:** turning a word into an fMRI vector
requires TRIBEv2's text branch, whose model
(``text_feature.model_name: meta-llama/Llama-3.2-3B`` in the released
config) is gated on Hugging Face. This environment has no ``HF_TOKEN`` set
and no cached Llama weights (checked: no ``llama*`` directory under
``external/hf_cache``). ``embed_word_events``/``load_tribev2_text_model``
below raise a clear ``RuntimeError`` up front rather than failing deep
inside ``TribeModel`` -- **running them for real requires the user to
accept Meta's Llama 3.2 license on their own HF account and provide an
``HF_TOKEN`` with access**, plus ~6-12GB of disk for the weights. Nothing
else in this module is blocked: event construction and the ANN index
itself are plain code, testable now against synthetic vectors.

**Gradient-based inversion (differentiable text -> fMRI), investigated but
not implemented here -- feasibility notes only:**
``neuralset.extractors.text.HuggingFaceText._get_data`` wraps its entire
embedding computation in ``torch.no_grad()`` and returns cached, detached
numpy arrays (see ``_get_timed_arrays``/``_get_data``, both decorated with
``@infra.apply(..., cache_type="MemmapArrayFile")``); and
``tribev2.model.FmriEncoderModel.forward``/``aggregate_features`` only ever
consume pre-extracted per-modality feature tensors already sitting in a
``SegmentData`` batch (``batch.data[modality]``, shape ``B, T, H`` after
each modality's own projector). There is no existing differentiable path
from raw text through to an fMRI prediction anywhere in this codebase.
Making one work would mean: (1) hand-building a parallel, differentiable
version of ``HuggingFaceText``'s per-token/per-layer aggregation
(``_aggregate_tokens``/``_aggregate_layers`` in
``neuralset/extractors/base.py``) that consumes ``inputs_embeds`` instead
of tokenized text and replicates the exact
``layers``/``layer_aggregation``/``token_aggregation`` config TRIBEv2's
release uses; and (2) manually constructing a ``SegmentData``-shaped batch
to call ``FmriEncoderModel.forward`` directly, bypassing the
extractor-cache/DataLoader pipeline (``TribeModel.predict`` /
``self.data.get_loaders``) entirely. That's a substantially bigger lift
than nearest-neighbor retrieval (B1 below) for an unproven idea -- treated
as a feasibility read-through this pass, not attempted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import hnswlib
import numpy as np
import pandas as pd

from musicbrain.lexicon import NRCVAD
from musicbrain.parcellation import Schaefer400Parcellator
from musicbrain.tribev2_utils import DEFAULT_CACHE_FOLDER, DEFAULT_HF_CACHE, TRIBEV2_REPO_ID


def build_word_events(words: list[str]) -> pd.DataFrame:
    """Hand-construct a one-row-per-word events DataFrame for TRIBEv2's text
    extractor, bypassing the demo's default gTTS -> whisperx round trip
    (``tribev2.demo_utils.TextToEvents``) -- wasteful here since the text is
    already known, and the same heavy-dependency concern ROADMAP.md Phase 0
    already ruled out for the audio pipeline.

    Fields match what ``neuralset.extractors.text.HuggingFaceText`` actually
    reads (``_get_timed_arrays``/``_get_data``): ``context`` is required
    because ``contextualized=True`` is that extractor's default -- for a
    standalone word, using the word itself as its own context is the
    natural (if minimal) choice, since there's no surrounding sentence to
    give it one. Each word gets its own ``timeline`` so ``TribeModel.predict``
    treats it as an independent single-event trial rather than merging
    same-timeline words into one segment (mirrors how
    ``tribev2_utils.predict_fmri_trace`` processes one audio file per call).

    **Not yet validated against the real text extractor** (see module
    docstring) -- this construction is grounded in the extractor's actual
    field reads, not guessed, but should be spot-checked on the first real
    run once ``HF_TOKEN``/license access is available.
    """
    rows = []
    for i, word in enumerate(words):
        rows.append(
            {
                "type": "Word",
                "text": word,
                "sentence": word,
                "context": word,
                "language": "english",
                "start": 0.0,
                "duration": 1.0,
                "timeline": f"word_{i:05d}",
                "subject": "default",
            }
        )
    return pd.DataFrame(rows)


def _require_hf_token() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "This requires a Hugging Face token with access to the gated "
            "meta-llama/Llama-3.2-3B model (TRIBEv2's text branch, "
            "text_feature.model_name in its released config.yaml). Accept "
            "the license at https://huggingface.co/meta-llama/Llama-3.2-3B "
            "on your own HF account, then set the HF_TOKEN environment "
            "variable before calling this."
        )


def load_tribev2_text_model(
    device: str | None = None,
    hf_cache_dir: Path = DEFAULT_HF_CACHE,
    cache_folder: Path = DEFAULT_CACHE_FOLDER,
):
    """Load TRIBEv2 with its text branch active (Llama-3.2-3B), frozen, eval
    mode -- mirrors ``tribev2_utils.load_tribev2_model``'s audio-only loader,
    but overrides ``data.text_feature.device`` instead of
    ``data.audio_feature.device``. Raises immediately if ``HF_TOKEN`` isn't
    set (see module docstring); this is the one call in this module that
    actually needs it.
    """
    _require_hf_token()
    import torch
    from huggingface_hub import hf_hub_download
    from tribev2.demo_utils import TribeModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    hf_cache_dir = Path(hf_cache_dir)
    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)

    config_path = Path(
        hf_hub_download(TRIBEV2_REPO_ID, "config.yaml", cache_dir=str(hf_cache_dir))
    )
    hf_hub_download(TRIBEV2_REPO_ID, "best.ckpt", cache_dir=str(hf_cache_dir))

    return TribeModel.from_pretrained(
        config_path.parent,
        cache_folder=str(cache_folder),
        config_update={
            "data.text_feature.device": device,
            "data.num_workers": 0,  # see tribev2_utils.load_tribev2_model's Windows-crash note
        },
    )


def embed_word_events(words: list[str], model, parcellator: Schaefer400Parcellator | None = None) -> np.ndarray:
    """Run *words* through TRIBEv2's text branch -> Schaefer-400
    parcellation, one vector per word. Requires *model* loaded via
    ``load_tribev2_text_model`` (which already enforces the HF_TOKEN
    requirement above).
    """
    from neuralset.events.utils import standardize_events

    parcellator = parcellator or Schaefer400Parcellator()
    events = standardize_events(build_word_events(words))
    preds, segments = model.predict(events=events, verbose=False)
    return parcellator.aggregate(preds)  # (n_words, 400)


@dataclass
class TextFmriIndex:
    """hnswlib-backed nearest-neighbor index over word -> Schaefer-400 fMRI
    vectors. Chosen over FAISS/Annoy for this project (see ROADMAP.md "Still
    open"): pip-installable (built cleanly from source here, no prebuilt
    Windows wheel for this Python version but no toolchain friction either),
    supports incremental add/update (relevant since this vocabulary is
    expected to grow, and Phase 3's CLIP-based photo retrieval wants the
    same pattern), and a good recall/speed fit at this project's realistic
    scale (tens to low-thousands of vectors) -- at today's n=59 scale,
    brute-force numpy would work identically; this is about having the
    right interface for where this is headed, not today's raw vector count.
    """

    words: list[str]
    index: hnswlib.Index

    @classmethod
    def build(cls, words: list[str], vectors: np.ndarray, space: str = "cosine") -> "TextFmriIndex":
        dim = vectors.shape[1]
        index = hnswlib.Index(space=space, dim=dim)
        index.init_index(max_elements=len(words), ef_construction=200, M=16)
        index.add_items(vectors.astype(np.float32), np.arange(len(words)))
        index.set_ef(min(50, len(words)))
        return cls(words=list(words), index=index)

    def knn_query(self, query_vectors: np.ndarray, k: int = 1) -> tuple[list[list[str]], np.ndarray]:
        labels, distances = self.index.knn_query(query_vectors.astype(np.float32), k=k)
        retrieved = [[self.words[i] for i in row] for row in labels]
        return retrieved, distances


@dataclass
class RetrievalValidation:
    n_windows: int
    valence_r: float
    arousal_r: float


def validate_against_deam(
    index: TextFmriIndex, lexicon: NRCVAD, query_vectors: np.ndarray, true_valence: np.ndarray, true_arousal: np.ndarray
) -> RetrievalValidation:
    """The concrete validity check this spike hinges on: for each queried
    DEAM window, retrieve its single nearest word and correlate *that
    word's* NRC-VAD (valence, arousal) against the window's own *true* DEAM
    label. High correlation is evidence retrieval tracks real affective
    content rather than modality-specific nuisance variance; near-zero is
    evidence it doesn't, at least not without further normalization.
    """
    retrieved, _ = index.knn_query(query_vectors, k=1)
    retrieved_words = [row[0] for row in retrieved]
    retrieved_va = np.array([lexicon.va(w) if w in lexicon else (np.nan, np.nan) for w in retrieved_words])

    valid = ~np.isnan(retrieved_va).any(axis=1)
    valence_r = float(np.corrcoef(retrieved_va[valid, 0], true_valence[valid])[0, 1])
    arousal_r = float(np.corrcoef(retrieved_va[valid, 1], true_arousal[valid])[0, 1])
    return RetrievalValidation(n_windows=int(valid.sum()), valence_r=valence_r, arousal_r=arousal_r)
