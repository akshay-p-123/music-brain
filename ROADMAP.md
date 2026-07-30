# Roadmap

Companion to `PROJECT_SPEC.md`. Sequenced as: verify the riskiest
assumptions cheaply → prove the whole pipeline on one dataset → scale to
the full pooled set → build the still-open branches → ship.

Decisions locked in for this roadmap (see `PROJECT_SPEC.md` §9 for the
research behind them):

- **Compute:** Google Colab (free/Pro tier — not a local GPU or paid
  cloud). Every step below is written assuming ephemeral disk and capped
  session length.
- **Rollout:** proof-of-concept on one dataset (DEAM) before pooling all
  six.
- **License:** TRIBEv2's CC-BY-NC-4.0 is acceptable — this project is
  non-commercial.
- **Parcellation:** Schaefer-400 on fsaverage5, to turn TRIBEv2's raw
  ~20,484-vertex output into the trunk's `P≈400` input. Implemented via
  the canonical CBIG fsaverage5 `.annot` files + `nibabel` directly (no
  `neuromaps` dependency needed in practice) — see Phase 0 below.

---

## Colab-specific practicalities (apply to every phase below)

- **Sessions are ephemeral and time-limited.** Nothing written to local
  Colab disk survives a disconnect. Any cache built during Step 0 (audio
  → TRIBEv2 → parcellated windowed traces) must be pushed incrementally to
  a free **Hugging Face Hub dataset repo** (or Google Drive, mounted) as
  it's produced — not held in local disk until "done." Chunk long runs
  (e.g. one dataset per session, or checkpoint every N clips) so a
  disconnect loses minutes, not hours.
- **MTG-Jamendo is the outlier by volume:** ~18,486 full-length tracks vs.
  hundreds for everything else. Its Step 0 pass will dominate total
  compute time and should be budgeted/chunked separately from the other
  five datasets, not treated as "one more dataset."
- TRIBEv2 audio-only inference only needs the Wav2Vec-BERT 2.0 branch
  loaded — don't pull the LLaMA-3.2-3B or V-JEPA2-Giant weights in Colab;
  they're dead weight for this pipeline and will eat Colab's RAM/disk
  quota for nothing. **Confirmed in Phase 0** — but only when events are
  built via `get_audio_and_text_events(..., audio_only=True)` directly,
  not `TribeModel.get_events_dataframe()`, which silently pulls in
  Whisper large-v3 via `whisperx` otherwise (see Phase 0 findings).

---

## Phase 0 — De-risk the two unverified assumptions — DONE

Both spikes were run for real (locally, CPU-only, not just planned) and
both passed. Code lives in `notebooks/phase0_spike1_tribev2_audio.py` and
`notebooks/phase0_spike3_anchor_interpolation.py`; reusable pieces moved
into `src/musicbrain/` (`parcellation.py`, `lexicon.py`, `anchors.py`,
`soft_prompt.py`).

1. **TRIBEv2 audio-only inference spike — confirmed working.**
   `facebook/tribev2` loads, and calling the lower-level
   `get_audio_and_text_events(..., audio_only=True)` (bypassing
   `TribeModel.get_events_dataframe`, see finding below) correctly drops
   both the text and video extractors ("Removing extractor text/video as
   there are no corresponding events") and loads only the audio branch
   (`facebook/w2v-bert-2.0`). Output shape was exactly `(10, 20484)` for a
   10s clip — 20,484 = the expected fsaverage5 vertex count, confirming
   the footprint assumption in PROJECT_SPEC.md §8.2 is correct *once this
   bypass is used*.
2. **Vertex→Schaefer-400 aggregation — confirmed working, after a bug fix.**
   Used the canonical CBIG fsaverage5 `.annot` files (not nilearn's
   volumetric/MNI Schaefer fetcher, which is a different atlas
   representation of the same name) directly via `nibabel`, no `neuromaps`
   dependency needed. First pass produced 402 "parcels," not 400, because
   each hemisphere's `.annot` carries an extra non-cortical
   "Background+FreeSurfer_Defined_Medial_Wall" label alongside its 200 real
   parcels — fixed in `Schaefer400Parcellator` by filtering those out by
   name. Output shape is now exactly `(T, 400)` as the trunk expects.
3. **Anchor-word interpolation spot-check — confirmed coherent.** Built the
   8-word NRC-VAD anchor set, pulled real embeddings from
   Qwen2.5-1.5B-Instruct's vocabulary, fed a hand-built "calm → tense →
   release" VA trajectory in as interpolated soft tokens via
   `inputs_embeds`. Output: *"starts with a quiet and peaceful feeling,
   then becomes more chaotic and unpredictable as it progresses... a return
   to a calm and peaceful state"* — genuinely trajectory-aware, with zero
   projector training. A flat/near-constant trajectory produced a
   less-flat-than-ideal description ("gradual progression from positive to
   neutral") — not a failure, but a calibration nuance worth tracking
   during Phase 1's dynamic-vs-flat sanity check (§6).

**Three findings that change Phase 1/2 implementation, beyond the two
questions Phase 0 set out to answer:**

- **`TribeModel.get_events_dataframe()` is not safe to call directly for
  this project.** Even with only `audio_path` set, it calls
  `get_audio_and_text_events(..., audio_only=False)` by default, which
  shells out to `uvx whisperx` (Whisper **large-v3**) to transcribe
  speech — a multi-GB model and an external-binary dependency, run on
  *every clip*, even pure music/soundscape audio with no speech to find.
  Step 0's dataset-manufacturing code must call the lower-level
  `tribev2.demo_utils.get_audio_and_text_events(events_df, audio_only=True)`
  directly instead, building the same one-row events DataFrame
  (`type="Audio", filepath=..., start=0, timeline="default",
  subject="default"`) by hand. Skipping this is not just a convenience —
  running whisperx large-v3 across ~18.5k MTG-Jamendo tracks for no reason
  would dominate the Colab compute budget for zero benefit.
- **The released config hardcodes `device: cuda` per-modality, not
  `"auto"`.** `facebook/tribev2`'s `config.yaml` sets
  `data.audio_feature.device: cuda` explicitly (same for text/video/image).
  On CPU-only runs this crashes with `Torch not compiled with CUDA
  enabled` even though the brain model itself loads fine on CPU. Fix is
  `TribeModel.from_pretrained(..., config_update={"data.audio_feature.device": "cpu"})`
  when a GPU isn't available; on Colab, simpler to just pick a GPU runtime
  since TRIBEv2's own paper trained on A100/V100 and CPU inference is slow
  regardless (297s for one 10-second clip in this spike, single-threaded
  CPU, no batching).
- **TRIBEv2's effective output resolution was ~1 prediction/second in this
  test (10 predictions for a 10s clip), not 0.5s.** This directly informs
  the still-open "window size" decision (PROJECT_SPEC.md §7 item 4):
  DEAM/PMEmo's native 0.5s label resolution is *finer* than what TRIBEv2
  itself appears to natively produce, so matching it 1:1 isn't free —
  either upsample TRIBEv2's predictions to 0.5s (repeating/interpolating)
  or accept ~1s trunk windows and downsample the labels instead. Revisit
  this with a real (non-synthetic) DEAM clip early in Phase 1 before
  committing to one direction, since this was measured on a 10s synthetic
  tone, not real music.
- **Windows-only bugs, not relevant on Colab:** `from_pretrained` breaks
  on Windows twice (an HF repo-id string gets mangled through
  `pathlib.Path`, and the released `config.yaml` contains pickled
  `PosixPath` objects that Windows can't instantiate at all). Both were
  worked around locally to get this far, but neither should occur on
  Colab's Linux runtime — noted here only so it isn't mistaken for a
  TRIBEv2-wide bug if it resurfaces.

**Exit criteria — met.**

---

## Phase 1 — Single-dataset proof of concept (DEAM)

Why DEAM first: native 0.5s dynamic VA labels, full-length audio
available with low access friction, moderate size (1802 clips) — the
cleanest source to validate the *architecture*, isolated from the
cross-dataset calibration questions Phase 2 introduces.

**Status: infrastructure built and dry-run validated end-to-end on real
DEAM audio (this session, local CPU) -- exit criteria below not yet
met, that requires the full-scale run.** All of items 1-5 below have a
working implementation in `src/musicbrain/` (`datasets/deam.py`,
`fmri_cache.py`, `model.py`, `train.py`, `verify.py`, `text_branch.py`) plus
`notebooks/phase1_deam_pipeline.py` as the runnable end-to-end
orchestration script -- this is the code meant to scale up on Colab per
the practicalities above, not a separate throwaway spike. It was smoke
tested locally on a real (not synthetic) 15-clip DEAM subset (10
train / 5 held-out, full-length audio, not truncated) and ran cleanly:
Step 0 cached all 15 clips, Step 1 trained without erroring, Step 2
computed both metrics, and the text branch produced coherent sentences
from the *model's own* predicted trajectories (not hand-built ones, unlike
the Phase 0 spike). Findings that matter for the real run:

- **TRIBEv2's ~1 prediction/second rate is confirmed on real audio, not
  just Phase 0's synthetic tone.** All 15 real clips measured exactly
  1.00Hz. The window-size decision (item 6 below) still needs to be made
  from real Step 2 results, but the "TRIBEv2 native rate vs. DEAM's 0.5s
  labels" mismatch this decision hinges on is now doubly confirmed.
- **A Windows-only performance trap, not a TRIBEv2 problem.** The first
  timing pass measured ~2-15s of wall time per second of audio and
  intermittent worker-process crashes ("paging file too small" DLL
  errors). Root cause: TRIBEv2's data loader defaults `num_workers` to
  `os.cpu_count()`, and Windows' spawn-based multiprocessing re-imports
  torch/sklearn/scipy in every worker process, which this machine's page
  file couldn't sustain. Fixed by setting `config_update={"data.num_workers": 0}`
  in `tribev2_utils.load_tribev2_model` (see its docstring) -- after the
  fix, real per-clip TRIBEv2 inference is ~1-2s per 20s of audio, not
  minutes. Not expected to affect Colab (fork, not spawn) but harmless to
  leave in either way.
- **Exit criteria not met on this smoke-test subset, as expected.** With
  only 10 training clips (590 windows) the trunk/VA head overfits
  immediately (train loss keeps dropping, val loss plateaus/worsens after
  ~epoch 15) and held-out within-clip correlation is poor
  (mean valence r=-0.40, arousal r=+0.13 across the 5 held-out clips) --
  not the "actually correlated" bar the exit criteria require. This is
  the expected result of a 15-clip plumbing check, not a finding about
  the architecture; the real correlation number can only come from
  training on the full 1802-clip set.
- **The dynamic-vs-flat sanity check (Section 6) needs a properly trained
  model to be meaningful.** Run against this undertrained checkpoint, the
  two sentences came out similar ("rollercoaster ride... ups and downs"
  for both the high-variance and low-variance held-out clip) -- consistent
  with the model not yet having learned a real valence/arousal signal,
  not a failure of the anchor-interpolation mechanism itself (which
  Phase 0 already validated in isolation on hand-built trajectories).
  Re-run `text_branch.dynamic_vs_flat_sanity_check` after the full-scale
  Step 1 training to get a real read on this.

Remaining before Phase 1's exit criteria are actually met: run Step 0
over all 1802 clips (Colab, per the practicalities above -- local CPU
Step 0 is fast enough post-fix that even a full local run is plausible,
but Step 1 training quality is the real gate), then Step 1/Step 2/text
branch for real.

1. **Ingest DEAM.** Download audio + per-0.5s VA annotations.
2. **Step 0 for DEAM only.** Run every clip through the Phase-0-verified
   TRIBEv2 → Schaefer-400 pipeline, window to (initially) 0.5s, cache
   `[T, 400]` traces + VA labels to the HF Hub dataset repo.
3. **Train the trunk + VA head on DEAM alone** (§3.1/§3.2 architecture,
   §5 Step 1 objective).
4. **Step 2 verification, DEAM-only slice:** held-out VA correlation, and
   — the more important check — within-clip dynamic tracking (does the
   predicted trajectory actually track the real 0.5s trajectory, not just
   the clip average). This is the test that validates windowing over
   pooling; if it fails here, it's a one-dataset debugging problem, not a
   six-dataset one.
5. **Wire up the text branch end-to-end** on DEAM held-out clips: VA
   trajectory → anchor interpolation → frozen generation LLM → sentence.
   Run the spec's own sanity check (§6): compare a sharp-VA-swing DEAM
   track against a flat one, confirm the sentences differ in a way that
   reflects the trajectory difference, not just overall tone.

   **Text branch redesigned for vibe-descriptor output (done, this
   session).** The anchor vocabulary moved from the abstract 8-word
   EmotionCaps circumplex to MTG-Jamendo's 59-word mood/theme tag list, and
   the prompt now asks for a vivid, figurative "show don't tell" sentence
   (few-shot-exemplar-calibrated) instead of literal trajectory narration —
   see PROJECT_SPEC.md §3.3 for the full design and
   `src/musicbrain/vibe_lexicon.py`/`text_branch.py` for the implementation.
   **This is a tag-vocabulary-only use of MTG-Jamendo** (one small metadata
   file, 59 tag names) — entirely separate from item 3 below and from Phase
   2's still-open decision about ingesting MTG-Jamendo's ~18.5k-track
   *audio* corpus as training data; don't conflate the two. Verified
   locally: 56/59 tags hit NRC-VAD directly, the 3 LLM-estimated fallback
   placements (`ambiental`, `inspiring`, `soundscape`) landed sanely, and
   hand-built spike-3-style trajectories through the new prompt produced
   noticeably more figurative output ("...like a breath after holding your
   breath") than the old "rollercoaster ride" phrasing. Re-running the
   spec's dynamic-vs-flat sanity check against the existing (still
   undertrained, 15-clip) local checkpoint again produced near-identical
   sentences for both clips — expected per the finding below, not a
   regression; needs the full-scale Step 1 run to actually judge this.
6. **Window size decision (spec §7 item 4).** Use this phase's within-clip
   tracking results to pick a window size (start at 0.5s, widen if it's
   mostly noise) — this choice carries into Phase 2, so settle it here
   while there's only one dataset's noise characteristics to reason about.
   Reconcile against Phase 0's finding that TRIBEv2 itself only produced
   ~1s-resolution output on a synthetic clip — re-check this on a real
   DEAM clip before finalizing.

**Exit criteria:** a working audio → VA trajectory → sentence pipeline on
DEAM, with within-clip tracking that's actually correlated (not just
clip-average correlated), and a settled window size.

---

## Phase 2 — Scale to the remaining five datasets

What Phase 1 already built (TRIBEv2→Schaefer-400 cache pipeline, trunk/VA
head architecture, training loop, window size) carries over unchanged.
What's actually new per dataset is narrower than it looks — mostly a
label-format adapter, plus two datasets need a real decision:

- **PMEmo:** same shape as DEAM (0.5s dynamic VA) — near-identical
  adapter. Ignore the bundled EDA physiological data (spec doesn't call
  for it).
- **EmoMusic:** check whether the 45s clips have a genuine dynamic
  subseries or are effectively static-per-clip (spec flags this as
  unresolved); adapter branches on the answer.
- **MERP:** dynamic VA over full songs, small (54 tracks) — straightforward
  adapter, low weight in the pooled set by clip count.
- **EmoSoundscapes:** only a **single clip-level VA label**, not a
  trajectory. Decide how the windowed loss handles this: broadcast the one
  label to every window in the clip (simplest, but injects label noise
  into every window equally), or treat it as a separate lower-weight
  clip-level loss term. Pick one and note it — this is exactly the kind of
  per-source calibration issue §5 Step 2 asks you to check for.
- **MTG-Jamendo:** the real decision (spec §7 item 3, still open). No
  continuous VA at all. Two paths:
  - *Lexicon-mapped training signal:* map mood/theme tags to approximate
    VA via the same NRC-VAD lexicon as the anchor words, fold its ~18.5k
    tracks into training. More data, but only as trustworthy as the tag→VA
    mapping.
  - *Held-out generalization check only:* don't train on it; after Step 1,
    check whether the trunk's output lands near Jamendo's existing tags on
    data it never trained on. Simpler, no risk of a shaky lexicon mapping
    polluting the regression target, but throws away the largest available
    corpus as a training signal.
  Recommendation: start with the held-out-check-only path in this phase
  (lower risk, faster to ship), and revisit lexicon-mapped training later
  as an explicit follow-up if the pooled model needs more data.

Sequencing within Phase 2:

1. Build each dataset's adapter → common windowed-VA cache format (same
   format Phase 1 established for DEAM).
2. Run Step 0 for each (budget MTG-Jamendo separately per the Colab note
   above).
3. Re-run Step 1 training on the full pooled set.
4. Step 2 verification, now for real: per-source held-out correlation
   across all five scored sources (DEAM, PMEmo, EmoMusic, MERP,
   EmoSoundscapes), plus the MTG-Jamendo categorical check if that's the
   path chosen. If EmoSoundscapes or another source visibly drags down
   pooled performance or sits on a different scale, resolve via
   per-source normalization (spec §7 item 5) before shipping the pooled
   model — don't ignore it.

**Exit criteria:** pooled trunk/VA head trained across all six sources,
per-source correlations checked and any calibration issues resolved,
MTG-Jamendo integration decision made and implemented.

---

## Phase 3 — Image branch (open design, spec §7 item 1)

Not before Phase 1/2 because it's a smaller, separable piece and the spec
itself flags it as not worked through yet. Needs its own short design
pass, not just implementation:

1. Decide the retrieval corpus — the spec doesn't specify one. A fixed
   indexed photo set (e.g. a public Unsplash-derived dataset) with
   precomputed CLIP embeddings is the natural zero-shot-friendly choice.
2. Implement zero-shot CLIP text-encoder query from the anchor-word
   interpolation blend (mirrors §3.3's mechanism).
3. Factor CLIP into the 16GB inference budget (§8.2 already flags this as
   deferred to this decision point).
4. Define an evaluation approach for this branch (spec leaves it TBD in
   §6) — likely qualitative/human-judgment, same as the text branch.

---

## Phase 4 — Package and ship (HF Space, CPU Basic + Docker, §8)

1. Push trained trunk/VA-head weights to a HF Hub model repo.
2. Build the Docker Space per §8.1's decision (CPU Basic, not ZeroGPU) —
   re-verify at Space-creation time that Docker is still available on
   the free tier for this account, since the spec notes this may have
   changed recently.
3. Wire up `HF_TOKEN` as a Space secret (only needed if the chosen
   generation LLM is a gated Llama variant); TRIBEv2 itself is ungated.
4. Cache weights in the Space filesystem to soften cold starts; confirm
   nothing depends on runtime-written state surviving a restart.
5. Measure end-to-end request latency. If unacceptable, the first lever
   is a smaller/more quantized generation LLM (spec explicitly rules out
   reaching for paid hardware as the first fix).

---

## Phase 5 — Evaluation and polish

Run the full §6 evaluation table end to end: per-axis/per-source held-out
correlation, within-clip trajectory correlation, qualitative sentence
review on held-out clips, the dynamic-vs-flat sanity check, image branch
eval (once Phase 3 defines it), and a cross-branch human-judgment pass —
does a listener agree the sentence, VA trajectory, and photo actually match
what they hear.

---

## Still open after this roadmap (deliberately deferred, not forgotten)

- **MTG-Jamendo lexicon-mapped training** as a follow-up if the
  held-out-only path (Phase 2) leaves the pooled model wanting more data.
- **EmoSoundscapes / MERP exact CC sub-license** — confirm before any
  public redistribution of processed data, even though non-commercial use
  itself is already fine.
- Anything Phase 2's per-source calibration check surfaces that isn't a
  simple normalization fix.
- **RL / self-improvement loop for the text branch** — no feedback signal
  exists anywhere in this project yet (no human ratings, no automatic
  sentence-quality metric), so this stays an explicit idea for later, not
  something to build now.
- **Cross-modal fMRI retrieval spike (exploratory, started this session).**
  Idea: TRIBEv2 predicts onto the same Schaefer-400 space regardless of
  whether the stimulus was audio, text, or video, so a song's predicted
  fMRI vector and a mood-word's predicted fMRI vector are directly
  comparable — potentially replacing (or cross-checking) the NRC-VAD-anchor
  mechanism in §3.3 with something more end-to-end. Real blocker found:
  the text extractor's model (`meta-llama/Llama-3.2-3B`) is gated on
  Hugging Face and this environment has neither `HF_TOKEN` nor cached
  weights — running the retrieval side of this for real needs the user to
  accept Meta's license and provide a token. `hnswlib` added as this
  project's ANN library (pip-installable with Windows wheels, incremental
  index updates, good fit at this project's small-to-medium vocabulary
  scale — also the natural fit for Phase 3's CLIP-based photo retrieval
  later). Gradient-based inversion (differentiable text->fMRI) was
  investigated and looks like a substantially bigger lift than
  nearest-neighbor retrieval (see `src/musicbrain/vibe_retrieval.py`
  module docstring for the detailed feasibility notes) — not attempted
  this pass.
