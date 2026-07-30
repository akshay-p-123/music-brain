# Audio → Brain State → Emotion Trajectory → Sentence and Photo

**Scope:** general-purpose. Soundscape and music are not distinguished —
one pipeline, one pooled training set, any audio in.

## 1. Project goal

Take an audio clip and produce, from one shared brain-derived signal:

1. A sentence that reflects the *arc* of the clip's feeling over time, not
   a single static impression.
2. A valence/arousal (VA) trajectory — an interpretable, time-resolved
   affect readout.
3. A retrieved photo matching the clip (mechanism still open, see §7).

The audio is routed through TRIBEv2, a frozen model that predicts the fMRI
response a human brain would have to that stimulus. This is a learned
transform of the audio, not an added sensor — it cannot contain information
the audio didn't carry — but it gives a shared, biologically-grounded
origin for all outputs, and a natural place to extract a continuous affect
signal.

**Core design principle: freeze everything expensive, train only a small
bridge.** TRIBEv2 and the generation LLM are never fine-tuned. The only
trained parameters are a compact windowed trunk and a small VA head.

---

## 2. Why the design changed from an earlier caption-supervised version

Two things constrain this spec that are worth stating explicitly, because
they ruled out designs that would otherwise be natural:

- **No captions in any approved dataset.** EmoSoundscapes, DEAM, PMEmo,
  EmoMusic, and MERP are valence/arousal labels only, no text. MTG-Jamendo
  has mood/theme tags, not sentences. This rules out training an LLM
  soft-prompt projector via caption cross-entropy, and rules out training a
  caption-contrastive alignment head — there is no text target to train
  against.
- **A single clip-level label collapses the thing this project is actually
  about.** Reducing a whole song or soundscape to one emotion word or one
  VA point throws away exactly the temporal structure the fMRI trunk was
  built to capture — tension building, release, mood shifting mid-clip.
  DEAM and PMEmo's labels are natively time-resolved (every 0.5s); the
  architecture should use that resolution, not discard it.

The resolution: train only what can be trained without text (a windowed
trunk + VA regression head), and get language out of the frozen LLM
**without training a projector**, via a deterministic, geometry-based
construction described in §3.3. No caption data is required anywhere in
this pipeline.

---

## 3. Architecture

```
audio ──► [TRIBEv2, frozen] ──► fMRI trace [T, P]
                                    │
                                    ▼
                    [Trunk, trained, windowed — NOT pooled to one vector]
                    fingerprint h_t for each window t
                                    │
                                    ▼
                          [VA head, trained]
                    h_t → (valence_t, arousal_t)  — a trajectory, one point per window
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                                 ▼
        [anchor-word interpolation]          [anchor-word interpolation]
        (v_t, a_t) → soft token per window    (v_t, a_t) → query construction
                    │                                 │
        sequence of soft tokens,             image search query
        one per window, in order                     │
                    │                                 ▼
        [frozen LLM reads sequence]              retrieved photo
                    │
                sentence reflecting the arc
```

### 3.1 Trunk (`FmriTrunk`, windowed)

Architectural precedent for this module comes from two published fMRI
decoders — full citations in §9:

- **MindEye1** (Scotti et al., 2023) contributes the *shape*: a single
  large linear squeeze from raw fMRI down to a few hundred/thousand
  channels, carrying the bulk of the parameters, followed by a couple of
  small residual blocks with aggressive dropout. It was built for a single
  time-collapsed fMRI snapshot ("beta values"), so it has no real temporal
  handling of its own.
- **Dynadiff** (Careil, Benchetrit & King, 2025) contributes the *temporal*
  half this spec needs. Its brain module applies **timestep-specific
  linear layers** (distinct weights per time sample) before any temporal
  aggregation, and its own ablation shows this matters: removing the
  timestep-specific layers in favor of one shared layer across all time
  samples cost 2.95 CLIP-similarity points and 1.33 AlexNet(2) points.
  Dynadiff still aggregates over time at the end to produce one embedding
  per clip; this spec deliberately **omits that final aggregation** to
  keep a full per-window trajectory, since collapsing it is exactly the
  failure mode §2 was written to avoid.

- **Input:** TRIBEv2 output parcellated to `P ≈ 400–1000` (Schaefer atlas).
- **Windowing:** do **not** pool the full clip into one vector. Split the
  fMRI trace into short windows (e.g. matching DEAM/PMEmo's 0.5s label
  resolution, or a coarser fixed window if that proves too fine-grained in
  practice) and produce one fingerprint per window.
- **Per-window squeeze:** one linear layer `P → 512`, applied independently
  to each window with shared weights — MindEye1's squeeze shape, applied
  per-timestep in the spirit of Dynadiff's timestep-specific layer rather
  than once on a single pooled input.
- **Body:** 2 residual blocks at width 512, applied per window
  (linear → norm → GELU → linear, residual add).
- **Regularization:** dropout 0.5 after the squeeze, 0.15 in the residual
  blocks — fMRI data is small and noisy regardless of windowing.
- **Output:** a sequence of fingerprints `h_1 ... h_T`, one per window, not
  a single pooled `h`.

### 3.2 VA head (replaces the earlier 3-axis PAD head)

`512 → 128 → 2`, one hidden layer, GELU, dropout 0.1. Applied per window,
producing `(valence_t, arousal_t)` for each `h_t`. MSE/Huber loss.
Dominance is dropped — none of the approved datasets label it.

### 3.3 Anchor-word interpolation (deterministic, not trained)

This is what makes the VA trajectory "vectors the LLM understands
directly" without any caption-supervised training:

1. Take a small set of anchor emotion words. **Superseded from the original
   8-word EmotionCaps circumplex set** (eventful, uneventful, pleasant,
   unpleasant, exciting, boring, quiet, chaotic) **to MTG-Jamendo's 59-word
   mood/theme tag vocabulary** (calm, melancholic, uplifting, dreamy, epic,
   ...) — real single-concept mood/scene words a listener would actually
   reach for, not an abstract circumplex, while staying just as
   VA-groundable (`src/musicbrain/vibe_lexicon.py`). This is a
   *tag-vocabulary-only* use of MTG-Jamendo (one small metadata file with
   the 59 tag names) — unrelated to §4/ROADMAP.md's separate, still-open
   decision about ingesting MTG-Jamendo's ~18.5k-track *audio* corpus as
   training data.
2. Look up each anchor word's (valence, arousal) coordinate from a
   published affective word-norm lexicon (NRC-VAD). **56 of the 59
   MTG-Jamendo tags have a direct NRC-VAD entry**, and one more
   (`inspiring`) resolves via a cheap suffix-stripping lemma fallback to a
   form NRC-VAD does cover (`inspire`) — NRC-VAD scores a fixed set of
   word forms, not every inflection, so a missing tag's lemma is sometimes
   already scored (`vibe_lexicon._resolve_lemma`). Only the remaining 2
   (`ambiental`, `soundscape`) — a non-English word and a compound/technical
   term, neither reducible to a covered lemma — fall back to a one-time,
   deterministic (temperature=0) LLM placement onto the circumplex,
   few-shot-calibrated against real NRC-VAD (word, valence, arousal)
   examples and cached to `data/lexicons/mtg_jamendo_va_llm_estimated.json`
   so each word is only ever placed once
   (`vibe_lexicon.estimate_va_via_llm`).
3. Look up each anchor word's real embedding in the frozen LLM's own
   vocabulary — genuine, in-distribution vectors.
4. For each window's predicted `(valence_t, arousal_t)`, compute a weighted
   interpolation across the anchor embeddings (inverse-distance or softmax
   kernel over distance in VA space). The result is a blended, continuous
   vector positioned between real word-embeddings — no word is ever
   discretely chosen.
5. Feed the resulting sequence of interpolated vectors to the frozen LLM as
   soft tokens, in temporal order, so it can write about the shape of the
   change, not just an average mood.

**No trainable weights here** (at most a single tunable temperature
parameter for the interpolation kernel, not trained against text).

**Prompting, revised:** the generation prompt (`text_branch.py`) asks for a
vivid, figurative, "show don't tell" one-sentence read of the trajectory —
concrete imagery/metaphor/scene-setting, explicitly avoiding clinical
language ("emotional arc", "trajectory", "valence/arousal",
"increases/decreases"). The output stays fully trajectory-aware (per-window
soft tokens preserved, not collapsed to a clip-level label — see §2's
rationale). A frozen, never-fine-tuned model was found (local validation)
to default to generic "rollercoaster ride" phrasing even under an
instruction-only version of this prompt, so 2-3 hand-authored VA
trajectory -> hand-written target-style sentence pairs are embedded ahead
of the real query as few-shot exemplars — genuine soft-token blocks plus
their example text, not hard-coded strings — which is the standard lever
for shifting a frozen model's output style without any training.

**Known risk, stated plainly:** feeding a frozen LLM a sequence of
interpolated vectors it never saw during its own training is not the same
as trained soft-prompting (which typically fits those vectors end-to-end
against the LLM). This should produce coherent output because the vectors
are convex combinations of things the LLM already understands, but it is
unverified for this setup and should be spot-checked on real examples
before being trusted. **Fallback if it produces garbage:** alongside the
soft tokens, also construct a plain-text trajectory description from the
same interpolation weights ("moves from calm toward tense, then to a brief
release") as ordinary text tokens — guaranteed in-distribution, at the cost
of being explicit rather than implicit.

### 3.4 Image branch

Not yet finalized — see open decision in §7. The same anchor-interpolation
idea can extend here (query a CLIP text encoder using the anchor words or
their interpolated blend, zero-shot, no training required), but this has
not been worked through in the same detail as the text branch.

---

## 4. Data sources (final — six datasets, no substitutions)

| Dataset | Content | Temporal resolution | Domain |
|---|---|---|---|
| EmoSoundscapes | 1213 6-sec Freesound clips, human VA ratings | one label per clip | environmental sound |
| DEAM | 1802 clips, VA every 0.5s, multi-rater | dynamic, native | music |
| PMEmo (PMEmo2019) | 794 pop-song choruses, VA every 0.5s + EDA physiological data | dynamic, native | music |
| EmoMusic | 744 clips, 45s, VA | static per clip (or check for dynamic subseries) | music |
| MERP | 54 full songs, VA chosen to cover all 4 VA quadrants | dynamic | music |
| MTG-Jamendo (mood/theme) | large corpus, categorical mood/theme tags, no native VA | n/a — tags only | music |

**No caption data is used anywhere in this pipeline.** WavCaps, AudioCaps,
Clotho, EmotionCaps, MusicCaps, Song Describer, and JamendoMaxCaps were
considered in earlier design iterations and are explicitly **not** part of
this spec, since the architecture no longer has a caption-supervised
component to feed them to.

**Soundscape/music pooling.** Per the "general purpose, not distinguished"
requirement, all six sources are pooled into one training set for the
trunk and VA head — no separate soundscape-model / music-model split.
Because EmoSoundscapes (environmental, listening ratings) and the four
music datasets may not share a calibration scale, per-source held-out
correlation (§5, Step 2) must be checked before fully trusting the pooled
model, even though the shipped model stays unified.

**MTG-Jamendo integration.** Has no continuous VA, so it cannot directly
feed the VA regression loss. Two options, both viable, not yet decided
between:
- Map its mood/theme tags to approximate VA coordinates via a
  word-norm lexicon (same lexicon as §3.3's anchors), folding its much
  larger clip count into training.
- Use it purely as a large, free, categorical generalization check: does
  the trunk's VA-then-anchor-word output land near the tag MTG-Jamendo
  already assigned, on data the model never trained on.

---

## 5. Training pipeline

### Step 0 — Manufacture the data (once)

Run every clip from all six sources through TRIBEv2, cache the
**windowed, parcellated** fMRI trace (not pooled to one vector) alongside
its VA label(s) at native resolution. One-time cost; every later stage
reads this cache.

### Step 1 — Train the trunk + VA head, single objective

One loss: MSE/Huber regression on VA, applied per window, trained jointly
across the pooled six-dataset set. No caption-contrastive loss — nothing to
align to.

### Step 2 — Verification (checkpoint, not a training step)

Two checks before trusting the pooled model:
- **Per-source held-out correlation.** Break out DEAM vs. PMEmo vs.
  EmoMusic vs. MERP vs. EmoSoundscapes. If one source drags down pooled
  performance or shows a visibly different scale, treat that as a
  calibration issue to resolve (per-source normalization, reweighting)
  rather than ignoring it.
- **Within-clip dynamic tracking.** For DEAM/PMEmo specifically, check that
  the trunk's per-window output actually tracks the real 0.5s-resolution
  trajectory, not just the clip-level average. This is the test that
  validates the entire premise of windowing rather than pooling.

### No Step 3/4 in the old (caption-supervised) sense

The former "train an LLM projector" and "train an image-alignment head"
stages are gone — replaced by the deterministic anchor-interpolation
construction in §3.3, which has no training loop of its own.

---

## 6. Evaluation

| Component | Metric |
|---|---|
| VA head | Held-out correlation per axis and per source dataset (expect arousal to correlate better than valence, consistent with prior literature) |
| Within-clip dynamics | Correlation between predicted and true VA *trajectories* on DEAM/PMEmo held-out clips, not just clip-level averages |
| LLM branch | No reference sentences exist to score against automatically — evaluation is qualitative: read generated sentences for held-out clips, check whether the trajectory (not just an average mood) is reflected |
| LLM branch — sanity check | Compare output when fed a genuinely dynamic clip (e.g. a DEAM track with a sharp VA swing) against a flat/static clip; the sentences should differ in a way that reflects the difference in trajectory, not just in overall tone |
| Image branch | TBD pending §7 resolution |
| Whole pipeline | Cross-branch coherence and, ultimately, human judgment — does a listener agree the sentence/VA trajectory/photo match what they hear |

---

## 7. Open design decisions

1. **Image branch mechanism.** Not worked through to the same depth as the
   text branch. Candidate: zero-shot CLIP text-encoder query from anchor
   words or their interpolated blend, mirroring §3.3. Needs its own design
   pass before implementation.
2. **Anchor-interpolation validity.** Flagged as an unverified assumption
   in §3.3 — spot-check on real examples early, before committing further
   engineering on top of it.
3. **MTG-Jamendo integration method.** Lexicon-mapped training signal vs.
   held-out categorical check only — decide based on how much the lexicon
   mapping is trusted vs. how valuable a large free generalization test is.
4. **Window size.** Not yet fixed — start near DEAM/PMEmo's native 0.5s
   label resolution and adjust based on Step 2's within-clip tracking
   results; too fine may be mostly noise, too coarse re-collapses the
   trajectory this design exists to preserve.
5. **Per-source VA calibration.** Whether soundscape (EmoSoundscapes) and
   music (DEAM/PMEmo/EmoMusic/MERP) ratings need separate normalization
   before pooling — resolve using Step 2's per-source correlation results.

---

## 8. Infrastructure — free hosting only, no paid tier

### 8.1 The Docker-vs-ZeroGPU conflict

Two free paths exist on Hugging Face Spaces, and they are **not currently
compatible with each other**:

- **CPU Basic** (2 vCPU, 16GB RAM, no GPU): the standing free tier, works
  with a Docker Space, no daily quota, sleeps after inactivity.
- **ZeroGPU** (shared H200/RTX-6000-class GPU, ~5 minutes/day free quota):
  free GPU access, but as of the most recent check, ZeroGPU Spaces are
  **Gradio-SDK-only**, not compatible with a raw Docker Space. There is
  also a recent report that new free accounts may default to ZeroGPU-only
  at Space creation with Docker marked paid — this should be verified
  directly at Space-creation time on the actual account being used, since
  it appears to have changed recently.

**Decision for this project: build on CPU Basic + Docker.** This matches
the stated Docker preference, has no quota clock to manage, and is
unambiguously free. Revisit Gradio + ZeroGPU only if measured CPU-only
latency proves unworkable — don't pre-optimize for GPU before confirming
CPU is actually too slow.

### 8.2 Sizing the pipeline to 16GB RAM, CPU-only

Resident models at inference time, and what to do about each:

- **TRIBEv2** — trimodal (text/video/audio) at full capacity, but this
  pipeline is audio-only: only the audio branch (Wav2Vec-BERT 2.0,
  ~600M params) needs to be resident. The 3B LLaMA text branch and
  V-JEPA2-Giant video branch are not required and should not be loaded.
  Expect real latency on CPU regardless; budget for it explicitly rather
  than being surprised.
- **Generation LLM** — go as small as tolerable for quality: Llama-3.2-1B-
  Instruct or Qwen2.5-1.5B-Instruct, 4-bit quantized. This is the one
  component with real flexibility, and quantization here is what makes CPU-
  only inference time tractable.
- **Trunk + VA head** — tiny, negligible memory footprint.
- **No CLAP/CLIP hosting required for the text branch** (dropped along with
  the caption-supervised stimulus channel in the earlier design). If the
  image branch (§7, open) ends up needing CLIP, that's an additional
  frozen model sharing the same 16GB budget — factor this in when that
  decision is made.

### 8.3 Other free-tier constraints to plan around

- **Gated model access.** Llama checkpoints (used only if the generation
  LLM is a Llama variant) are gated on Hugging Face — set `HF_TOKEN` as a
  Space secret, not baked into the Dockerfile. TRIBEv2's own weights are
  ungated, but licensed CC-BY-NC-4.0 (non-commercial use only).
- **Cold start.** CPU Basic Spaces sleep after inactivity; loading TRIBEv2 +
  an LLM from scratch on every wake is slow. Cache weights in the Space's
  filesystem where possible rather than re-downloading on every restart.
- **No persistent storage on the free tier** — anything written at runtime
  does not survive a Space restart unless baked into the image or re-
  fetched from an external free host (e.g. the HF Hub itself, which is free
  for model/dataset storage).
- **Training happens off-Space, always.** Train on a separate free or
  low-cost environment (local GPU, Colab), never inside the Space itself.
  Push only the trained trunk/VA-head weights to a HF model repo (free);
  the Space pulls those plus the frozen backbones at startup.
- **Request latency.** With no GPU and two frozen models resident, expect
  slow per-request inference. If this makes the Space feel unresponsive,
  the first free-tier lever to pull is a smaller/more aggressively
  quantized generation LLM, not paid hardware.

---

## 9. References

- Scotti, P. S. et al. (2023). *Reconstructing the mind's eye: fMRI-to-image
  with contrastive learning and diffusion priors.* arXiv:2305.18274.
  ("MindEye1" / "MindEye" throughout this spec.) Source of the trunk's
  squeeze-plus-residual-blocks shape (§3.1) — a decoder mapping
  time-collapsed fMRI beta-values to an image embedding via one large
  linear projection (~90% of parameters) followed by small residual
  blocks with heavy dropout (0.5 early, 0.15 in-block), trained with a
  CLIP-style contrastive retrieval loss on ~16GB/single-GPU/few-hour
  budgets. Has no meaningful temporal handling — designed for a single
  snapshot per trial, not a timeseries.

- Careil, M., Benchetrit, Y., & King, J.-R. (2025). *Dynadiff: Single-stage
  Decoding of Images from Continuously Evolving fMRI.* arXiv:2505.14556.
  FAIR at Meta. Source of the per-window temporal handling adapted in
  §3.1 — a brain module using timestep-specific linear layers (distinct
  weights per fMRI time sample) before temporal aggregation, jointly
  fine-tuned end-to-end with a frozen latent diffusion model via LoRA
  adapters on cross-attention layers. Its own ablation (Table 2)
  confirms timestep-specific processing outperforms a single
  time-shared layer. This spec adapts the *timestep-specific-then-later*
  idea but, unlike Dynadiff, deliberately skips the final temporal
  aggregation step to preserve the full per-window trajectory (§2, §3.1).

- d'Ascoli, S., Rapin, J., Benchetrit, Y., Brooks, J., et al. (2026).
  *A Foundation Model of Vision, Audition, and Language for In-Silico
  Neuroscience.* arXiv:2605.04326. Meta FAIR. Introduces **TRIBE v2**,
  the trimodal (text/video/audio) fMRI-response predictor this spec routes
  audio through (§1, §3.1). Successor to the original **TRIBE**
  (Algonauts 2025 winner, arXiv:2507.22229). Weights: `facebook/tribev2`
  on Hugging Face, ungated, CC-BY-NC-4.0. Text branch is LLaMA-3.2-3B,
  video branch is V-JEPA2-Giant, audio branch is Wav2Vec-BERT 2.0 — the
  three modalities can be queried independently, so audio-only inference
  does not require loading the text or video branches. Output is
  per-average-subject predictions on the **fsaverage5 cortical surface
  mesh (~20,484 vertices)**, not pre-parcellated — §3.1's `P ≈ 400–1000`
  Schaefer input requires an explicit vertex→parcel aggregation step on
  top of the raw TRIBEv2 output (see `ROADMAP.md` Phase 0).
