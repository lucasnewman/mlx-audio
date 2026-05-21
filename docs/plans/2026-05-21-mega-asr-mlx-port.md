# Mega-ASR MLX Port Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Each task uses superpowers:test-driven-development (write failing test → see it fail → minimal code → pass → commit).

**Goal:** Port Mega-ASR (robustness layer over Qwen3-ASR-1.7B) to mlx-audio with full fidelity — the audio-quality router and in-place LoRA switching — reusing the existing `qwen3_asr` MLX backbone.

**Architecture:** Mega-ASR = vanilla Qwen3-ASR-1.7B + a 23.1M-param PEFT LoRA adapter + a 1.17M-param raw-audio quality router. At inference, the router classifies each utterance as clean/degraded; degraded utterances run with LoRA deltas added to the base weights, clean ones run the base path. We add a new `mlx_audio/stt/models/mega_asr/` module that wraps the existing `Qwen3ASRModel`, ports the router, and applies/removes LoRA deltas in place — no second model in memory.

**Tech Stack:** Python ≥3.10, MLX, `mlx_lm` (KVCache, generate_step), `transformers` (tokenizer + WhisperFeatureExtractor), `huggingface_hub`, `safetensors`. Reference (PyTorch, for fixtures only): `torch`, PEFT, `soundfile`, `scipy`.

**Context docs:** `docs/mega-asr-porting-research.md`. Reference repo `github.com/xzf-thu/Mega-ASR@29b14321`, weights `hf.co/zhifeixie/Mega-ASR`. All sources Apache-2.0.

**Reuse, do not reimplement:** `mlx_audio/stt/models/qwen3_asr/qwen3_asr.py` (`Qwen3ASRModel`, `AudioEncoder`, `TextModel`, `sanitize`, `post_load_hook`, `generate`, `stream_transcribe`) and `config.py` (`ModelConfig`, `AudioEncoderConfig`, `TextConfig`).

> **SOURCE-OF-TRUTH RULE:** For the router, the **checkpoint state-dict tensor names and shapes are authoritative**. Checkpoint *metadata* (and the research doc's metadata-derived numbers) are only hints and are known to be inaccurate (metadata says `num_layers=4` + single linear head; the released tensors show only `transformer.layers.0.*` + `transformer.norm.*` and a 2-layer MLP head `classifier.0`/`classifier.3`). Never hardcode router dims/layer counts from metadata — read them from the state dict (Task 0.3).

---

## Phase 0 — Setup & parity fixtures

### Task 0.1: Create isolated worktree

**Files:** none (git operation)

REQUIRED SUB-SKILL: superpowers:using-git-worktrees.

**Step 1:** From `/Volumes/DATA/mlx-audio` create a worktree off updated `main`:
```bash
git fetch origin
git worktree add -b feature/mega-asr ../mlx-audio-mega-asr origin/main
```
**Step 2:** Verify: `cd ../mlx-audio-mega-asr && git status -sb` → on `feature/mega-asr`, clean.
**Step 3:** Install dev + reference deps in this worktree:
```bash
uv sync --extra dev
uv pip install torch peft soundfile scipy huggingface_hub safetensors qwen-asr
```
Expected: import works: `uv run python -c "import torch, peft, soundfile"`.

### Task 0.2: Acquire weights + reference fixtures

**Files:**
- Create: `tests/stt/mega_asr/fixtures/gen_reference.py`
- Create: `tests/stt/mega_asr/fixtures/clean.wav`, `degraded.wav` (≤5 s, 16 kHz mono)
- Create (generated): `tests/stt/mega_asr/fixtures/reference.json`

**Step 1:** Download weights:
```bash
uv run python -c "from huggingface_hub import snapshot_download; \
print(snapshot_download('zhifeixie/Mega-ASR'))"
```
**Step 2:** Write `gen_reference.py` that, using the reference PyTorch `MegaASR`, emits for both clips:
```json
{
  "clean":   {"degraded_prob": 0.0xx, "use_lora": false, "text": "..."},
  "degraded":{"degraded_prob": 0.9xx, "use_lora": true,  "text": "..."},
  "router_logmel_clean_first10": [...],   // first 10 frames of 80-bin log-mel
  "router_logits_clean": [l0, l1],
  "router_logits_degraded": [l0, l1]
}
```
Also dump the **router's intermediate log-mel** and **raw logits** — these anchor parity tests downstream.
**Step 3:** Run it: `uv run python tests/stt/mega_asr/fixtures/gen_reference.py`. Expected: `reference.json` written; `degraded_prob` for `degraded.wav` ≥ 0.5 and for `clean.wav` < 0.5.
**Step 4:** Commit.
```bash
git add tests/stt/mega_asr/fixtures && git commit -m "test(mega_asr): add parity fixtures from PyTorch reference"
```

### Task 0.3: Capture exact reference router structure (checkpoint state dict = source of truth)

**Files:**
- Create: `tests/stt/mega_asr/fixtures/dump_router_keys.py`
- Create: `docs/plans/mega-asr-router-reference.md`

> The checkpoint metadata is unreliable (claims `num_layers=4`, single linear head). The released `audio_quality_router/best_acc_model.safetensors` actually contains only `transformer.layers.0.*` + `transformer.norm.*` and a 2-layer MLP head `classifier.0.*`/`classifier.3.*`. **The state-dict tensor names and shapes are the source of truth; metadata is only a hint.**

**Step 1:** Dump the real state dict. `dump_router_keys.py` opens the router safetensors and prints every tensor key with its shape, then infers and prints: `d_model` (from `transformer.norm.weight`), the number of distinct `transformer.layers.N` groups, the conv frontend channels/kernels, and the `classifier.0`/`classifier.3` in/out dims.
Run: `uv run python tests/stt/mega_asr/fixtures/dump_router_keys.py`
Expected: a printed table of `key → shape`; record the exact `num_layers`, `d_model`, conv layout, and classifier head dims.

**Step 2:** Read the reference source for details NOT encoded in tensor names: `src/MegaASR/model/utils/audio_quality.py` (LogMel args; Conv1d kernel/stride/padding; positional-encoding type sinusoidal vs learned; `nn.TransformerEncoderLayer` `activation` and `norm_first`; attention-pooling exact form; the activation+dropout sitting between `classifier.0` and `classifier.3`), `src/MegaASR/model/router.py` (how `degraded_prob`/threshold are computed), `src/MegaASR/model/utils/lora_switch.py` (`_set_lora` add/subtract math + scaling source).

**Step 3:** Write `docs/plans/mega-asr-router-reference.md` recording the FINAL reconciled structure: every tensor name → shape, with file:line references for source-only details. Where source defaults and the checkpoint disagree, **the checkpoint wins** — document the discrepancy explicitly.

**Step 4 (QA — executable):** Verify the reference doc is implementable and complete.
Run: `uv run python tests/stt/mega_asr/fixtures/dump_router_keys.py` and cross-check that every printed tensor appears in `mega-asr-router-reference.md` with a matching shape.
Expected result: every router checkpoint tensor is accounted for in the doc (frontend conv, `transformer.layers.0..N-1`, `transformer.norm`, attention-pooling params, `classifier.0`/`classifier.3`), and the doc reconciles source-vs-checkpoint differences so Phase 2 can be implemented with zero ambiguity (no leftover "TBD"/"4 layers?" notes).

**Step 5:** Commit.

---

## Phase 1 — Weight conversion

### Task 1.1: Router weight converter

**Files:**
- Create: `mlx_audio/stt/models/mega_asr/convert_router.py`
- Test: `tests/stt/mega_asr/test_convert_router.py`

**Step 1 (failing test):**
```python
def test_router_weights_remap():
    from mlx_audio.stt.models.mega_asr.convert_router import convert_router_weights
    out = convert_router_weights(ROUTER_SAFETENSORS)  # dict[str, mx.array]
    # 35 source tensors expected
    assert len(out) == 35
    # every Conv1d weight must be MLX layout [out, kernel, in] (ndim 3);
    # use the actual key names recorded in Task 0.3 — do NOT guess names
    conv_weights = [v for k, v in out.items()
                    if "conv" in k and k.endswith("weight") and v.ndim == 3]
    assert conv_weights, "expected at least one 3-D conv weight"
```
**Step 2:** Run → FAIL (module missing).
**Step 3:** Load with `safetensors`, map PyTorch keys → MLX module paths using the EXACT names from Task 0.3, transpose each `Conv1d` weight `[out,in,k] → [out,k,in]` and leave `Linear` as `[out,in]`. Return `dict[str, mx.array]`.
**Step 4:** Run → PASS.
**Step 5:** Commit.

### Task 1.2: LoRA adapter converter (respect `rank_pattern`)

**Files:**
- Create: `mlx_audio/stt/models/mega_asr/convert_lora.py`
- Test: `tests/stt/mega_asr/test_convert_lora.py`

**Step 1 (failing test):**
```python
def test_lora_pairs_and_scaling():
    from mlx_audio.stt.models.mega_asr.convert_lora import load_lora_adapter
    adapter = load_lora_adapter(MEGA_MERGED_DIR)   # -> {module_path: {"A":mx, "B":mx, "scaling":float}}
    # 1078 tensors => 539 modules
    assert len(adapter) == 539
    m = adapter["audio_tower.layers.0.self_attn.q_proj"]
    assert m["A"].shape[0] == m["B"].shape[1]      # rank matches
    # scaling = alpha_module / r_module (from rank_pattern/alpha_pattern, fallback global r/alpha=24/24)
    assert m["scaling"] > 0
```
**Step 2:** Run → FAIL.
**Step 3:** Parse `adapter_config.json` (`r`, `lora_alpha`, `rank_pattern`, `alpha_pattern`). For each `*.lora_A.weight`/`*.lora_B.weight` pair: strip prefix `base_model.model.thinker.`, derive module path, look up per-module `r`/`alpha` (fallback to globals), `scaling = alpha/r`. Keep `A=[r,in]`, `B=[out,r]` as `mx.array` (float32). Assert pair counts and rank consistency.
**Step 4:** Run → PASS.
**Step 5:** Commit.

---

## Phase 2 — Router port

### Task 2.1: Slaney log-mel frontend (80-bin)

**Files:**
- Create: `mlx_audio/stt/models/mega_asr/router.py` (start `LogMel80`)
- Test: `tests/stt/mega_asr/test_router_frontend.py`

**Step 1 (failing test):** Parity vs fixture (`router_logmel_clean_first10`):
```python
def test_logmel_matches_reference():
    import numpy as np, mlx.core as mx, json, soundfile as sf
    from mlx_audio.stt.models.mega_asr.router import LogMel80
    wav, sr = sf.read(FIX/"clean.wav"); ref = json.load(open(FIX/"reference.json"))
    mel = LogMel80()(mx.array(wav, mx.float32))     # [n_frames, 80]
    got = np.array(mel[:10]); exp = np.array(ref["router_logmel_clean_first10"])
    assert np.allclose(got, exp, atol=1e-3)
```
**Step 2:** Run → FAIL.
**Step 3:** STFT (`n_fft=400`, `hop=160`, `win=400`, Hann), power spectrum, Slaney mel filterbank (80, Slaney norm), `(log10(mel)+4)/4`. Reuse mel helpers in `mlx_audio` if a Slaney filterbank exists; otherwise build the filterbank with `librosa.filters.mel(..., norm='slaney', htk=False)` baked to a constant `mx.array`.
**Step 4:** Run → PASS.
**Step 5:** Commit.

### Task 2.2: Conv subsample + positional encoding

**Files:** Modify `router.py`; Test: `tests/stt/mega_asr/test_router_frontend.py`

**Step 1 (failing test):** shape: `[T,80] → conv → [T//4, d_model]` (d_model from Task 0.3) then `+ pos_enc`.
**Step 2:** FAIL.
**Step 3:** Two `nn.Conv1d(stride=2)` (kernel/padding per Task 0.3) projecting `80 → d_model` with the reference activation; add positional encoding (type per Task 0.3). Note MLX `Conv1d` expects `[B, T, C]`.
**Step 4:** PASS.
**Step 5:** Commit.

### Task 2.3: transformer_mini encoder (layer count from checkpoint)

**Files:** Modify `router.py`; Test: `tests/stt/mega_asr/test_router_encoder.py`

> Use the layer count and `d_model`/`nhead`/`dim_feedforward` recorded in Task 0.3 — **do NOT hardcode 4 layers from metadata.** The released checkpoint contains only `transformer.layers.0.*` (+ `transformer.norm.*`); implement exactly `N = number of distinct transformer.layers.* groups`.

**Step 1 (failing test):** single-layer numeric parity vs a torch `TransformerEncoderLayer` with identical (deterministically-seeded) weights, using the exact `activation`/`norm_first` from Task 0.3, within `atol=1e-3`.
**Step 2:** FAIL.
**Step 3:** Implement the `transformer_mini` layer matching the reference config (norm placement + activation per Task 0.3): MHA(`d_model`, `nhead`) + FFN(`dim_feedforward`). Stack `N` layers (from checkpoint) and apply the final `transformer.norm`.
**Step 4:** PASS.
**Step 5:** Commit.

### Task 2.4: Attention pooling + MLP classifier head (shapes from checkpoint)

**Files:** Modify `router.py`; Test: `tests/stt/mega_asr/test_router_pool.py`

> The head is a **2-layer MLP**, not a single linear: checkpoint keys `classifier.0.*` (Linear) and `classifier.3.*` (Linear→2) with activation+dropout at indices 1,2 between them. Build it from the recorded `classifier.0`/`classifier.3` in/out dims.

**Step 1 (failing test):** output `[2]` logits for one clip; head shapes match checkpoint (`classifier.0`: `[hidden, d_model]`, `classifier.3`: `[2, hidden]`); finite.
**Step 2:** FAIL.
**Step 3:** Implement attention pooling exactly as Task 0.3 (learned-query attention over time → context vector), then the MLP head `Linear(d_model→hidden)` → activation → dropout → `Linear(hidden→2)` (activation/dropout per Task 0.3).
**Step 4:** PASS.
**Step 5:** Commit.

### Task 2.5: AudioQualityRouter end-to-end parity (ACCEPTANCE)

**Files:** Modify `router.py` (`AudioQualityRouter.degraded_prob`, `.route`); Test: `tests/stt/mega_asr/test_router_e2e.py`

**Step 1 (failing test):** load converted router weights (Task 1.1) and assert against `reference.json`:
```python
def test_router_logits_and_decision():
    r = AudioQualityRouter.from_converted(ROUTER_MLX_WEIGHTS)
    for clip in ["clean", "degraded"]:
        wav = load(clip)
        logits = r.logits(wav)            # [2]
        assert np.allclose(np.array(logits), ref[f"router_logits_{clip}"], atol=2e-3)
        assert r.route(wav)["use_lora"] == ref[clip]["use_lora"]   # threshold 0.5
```
**Step 2:** FAIL.
**Step 3:** Wire frontend→conv→encoder→pool→head; `degraded_prob = softmax(logits)[1]`; `route()` returns `{"degraded_prob", "use_lora": prob >= 0.5}`.
**Step 4:** PASS (parity green = router done).
**Step 5:** Commit.

---

## Phase 3 — LoRA apply / switch in MLX

### Task 3.1: Build per-module deltas

**Files:**
- Create: `mlx_audio/stt/models/mega_asr/lora.py`
- Test: `tests/stt/mega_asr/test_lora_math.py`

**Step 1 (failing test):**
```python
def test_delta_equals_scaled_BA():
    import mlx.core as mx, numpy as np
    from mlx_audio.stt.models.mega_asr.lora import build_deltas
    A = mx.random.normal((8, 16)); B = mx.random.normal((32, 8)); s = 1.5
    deltas = build_deltas({"m.proj": {"A": A, "B": B, "scaling": s}})
    exp = s * (np.array(B) @ np.array(A))          # [32,16]
    assert np.allclose(np.array(deltas["m.proj"]), exp, atol=1e-5)
```
**Step 2:** FAIL.
**Step 3:** `build_deltas(adapter) -> {path: mx.array}` with `delta = scaling * (B @ A)` in float32; assert `delta.shape == (B.shape[0], A.shape[1])`.
**Step 4:** PASS.
**Step 5:** Commit.

### Task 3.2: In-place add/remove with idempotent toggle

**Files:** Modify `lora.py` (`apply_deltas`, `remove_deltas`); Test: `tests/stt/mega_asr/test_lora_switch.py`

**Step 1 (failing test):** apply then remove restores base weights (`atol=1e-4`); applying twice is rejected/guarded.
```python
def test_apply_remove_roundtrip(tiny_qwen3asr_model):
    base = snapshot_weights(tiny_qwen3asr_model)
    apply_deltas(tiny_qwen3asr_model, deltas)
    remove_deltas(tiny_qwen3asr_model, deltas)
    assert weights_close(tiny_qwen3asr_model, base, atol=1e-4)
```
**Step 2:** FAIL.
**Step 3:** Walk the model module tree; for each `path` add/subtract `delta` to the target `nn.Linear.weight` (cast delta to weight dtype at the add site, accumulate in fp32 if needed). Track an `_lora_active` flag to guard double-apply. Use `model.update(tree_unflatten(...))` or direct attribute set on the resolved leaf module.
**Step 4:** PASS.
**Step 5:** Commit.

---

## Phase 4 — MegaASR model & registration

### Task 4.1: Config, skeleton, registry

**Files:**
- Create: `mlx_audio/stt/models/mega_asr/__init__.py`, `config.py`, `mega_asr.py`
- Modify: `mlx_audio/stt/utils.py:12-26` (add `"mega_asr": "mega_asr"` to `MODEL_REMAPPING`)
- Test: `tests/stt/mega_asr/test_registration.py`

**Step 1 (failing test):** `get_model_class("mega_asr", ...)` imports the module and exposes `Model` + `ModelConfig`.
**Step 2:** FAIL.
**Step 3:** `MegaASRConfig.from_dict` parses base `audio_config`/`text_config` (delegating to `qwen3_asr` configs) plus `router_config` and `lora_config`. `mega_asr.Model` skeleton with `ModelConfig = MegaASRConfig`. Add registry entry.
**Step 4:** PASS.
**Step 5:** Commit.

### Task 4.2: Weight loading (base + adapter + router)

**Files:** Modify `mega_asr.py`; Test: `tests/stt/mega_asr/test_load.py`

**Step 1 (failing test):** with a prepared MLX model dir, `mlx_audio.stt.load(dir)` returns a `Model` whose `_asr` is a `Qwen3ASRModel`, `_router` is `AudioQualityRouter`, and `_deltas` has 539 entries.
**Step 2:** FAIL.
**Step 3 (impl):**
- Decide MLX repo layout: base weights as standard `model*.safetensors` (so `base_load_model`'s `load_weights` feeds `Qwen3ASRModel`), with `config.json` `model_type="mega_asr"` carrying base + router + lora config. Ship `lora.safetensors` and `router.safetensors` as **separate, non-shard** files loaded explicitly in `post_load_hook`.
- **Verify** how `mlx_audio.utils.load_weights` globs files (sub-step: read it). If it would slurp `lora.safetensors`/`router.safetensors`, place them in a subdir (e.g. `extras/`) or have `Model.sanitize` drop non-base keys so only `Qwen3ASRModel` weights remain.
- `Model.__init__` builds inner `Qwen3ASRModel(config)` + `AudioQualityRouter(router_config)`.
- `Model.sanitize` = `Qwen3ASRModel.sanitize` (base only).
- `Model.model_quant_predicate` delegates to inner (keeps `audio_tower` unquantized).
- `Model.post_load_hook`: call `Qwen3ASRModel.post_load_hook` (tokenizer + WhisperFeatureExtractor), then load router weights into `_router`, load adapter via `convert_lora` + `build_deltas` into `_deltas`.
**Step 4:** PASS.
**Step 5:** Commit.

### Task 4.3: Routed generate (ACCEPTANCE)

**Files:** Modify `mega_asr.py` (`_route`, `generate`, `stream_transcribe`); Test: `tests/stt/mega_asr/test_generate.py`

**Step 1 (failing test, shapes, random weights):** `Model.generate(clip)` returns `STTOutput` with `text:str` and `segments`.
**Step 2:** FAIL.
**Step 3 (impl):**
```python
def generate(self, audio, **kw):
    route = self._router.route(self._load_wav_16k(audio))   # raw-wav frontend
    if route["use_lora"] and not self._lora_active:
        apply_deltas(self._asr, self._deltas); self._lora_active = True
    elif not route["use_lora"] and self._lora_active:
        remove_deltas(self._asr, self._deltas); self._lora_active = False
    try:
        return self._asr.generate(audio, **kw)   # reuse qwen3_asr path verbatim
    finally:
        pass  # leave state; reset lazily on next route
```
Mirror for `stream_transcribe`. Reuse the inner sampler/chunking — do not duplicate.
**Step 4:** PASS.
**Step 5 (slow parity, real weights, `@pytest.mark.requires_weights`):** transcription on `clean.wav` matches `reference.json["clean"]["text"]` (base path) and `degraded.wav` matches degraded text (lora path); assert chosen `use_lora` matches reference per clip.
**Step 6:** Commit.

---

## Phase 5 — Integration

### Task 5.1: CLI end-to-end

**Files:** Test: `tests/stt/mega_asr/test_cli.py`
**Steps:** failing test invoking `python -m mlx_audio.stt.generate --model <mlx-dir> --audio degraded.wav` exits 0 and prints non-empty text; ensure no new CLI code needed (inherited). Commit.

### Task 5.2: Conversion entrypoint + docs

**Files:**
- Create: `mlx_audio/stt/models/mega_asr/convert.py` (one command: HF `zhifeixie/Mega-ASR` → MLX dir: convert base via existing `mlx_audio.convert`, attach converted router + lora, write `config.json`)
- Create: `mlx_audio/stt/models/mega_asr/README.md` (Python + CLI usage, `--dtype`/quant notes, that clean→base & degraded→lora is automatic)
- Modify: root `README.md` STT table (add Mega-ASR row)
- Test: `tests/stt/mega_asr/test_convert.py`

**Step 1 (failing test):**
```python
def test_convert_produces_loadable_dir(tmp_path):
    from mlx_audio.stt.models.mega_asr.convert import convert
    # STUB_HF_DIR = tiny fixture: random base shard + 1-layer router.safetensors + small adapter
    out = convert(STUB_HF_DIR, tmp_path / "mlx")
    assert (out / "config.json").exists()
    assert list(out.glob("model*.safetensors"))      # base weight shard(s)
    assert (out / "router.safetensors").exists()
    assert (out / "lora.safetensors").exists()
    from mlx_audio.stt import load
    m = load(str(out))
    assert type(m).__module__.endswith("mega_asr")   # mega_asr.Model
    assert hasattr(m, "_router") and hasattr(m, "_deltas")
```
**Step 2 (run, expect FAIL):** `uv run pytest tests/stt/mega_asr/test_convert.py -v`
Expected: FAIL — `ImportError`/`AttributeError` (`convert` missing).
**Step 3 (impl):** `convert(hf_dir, out_dir, dtype="bfloat16")`: convert base via `mlx_audio.convert`, run `convert_router_weights` + `convert_lora`, write `router.safetensors`/`lora.safetensors` (layout per Task 4.2) and a `config.json` with `model_type="mega_asr"`; then write the model README and add the root README STT row.
**Step 4 (run, expect PASS):** `uv run pytest tests/stt/mega_asr/test_convert.py -v`
Expected: PASS — the converted dir contains `config.json` + base shard(s) + `router.safetensors` + `lora.safetensors`, and `mlx_audio.stt.load(<dir>)` returns a `mega_asr.Model` with `_router` and `_deltas`.
**Step 5 (docs QA — manual):** Run `grep -n "Mega-ASR" README.md`.
Expected: a new STT-table row is returned; and `mlx_audio/stt/models/mega_asr/README.md` contains both Python and CLI usage snippets.
**Step 6:** Commit.

### Task 5.3: Full verification

REQUIRED SUB-SKILL: superpowers:verification-before-completion.
**Steps:**
- `uv run pytest tests/stt/mega_asr -v` (mark slow/`requires_weights` separately) → all green.
- `uv run python -m mlx_audio.stt.generate --model <mlx-dir> --audio degraded.wav --verbose` → sensible transcription.
- Lint/diagnostics clean on changed files.
- Commit; then superpowers:finishing-a-development-branch (PR or merge).

---

## Acceptance criteria (Definition of Done)

- [ ] Router structure matches the **checkpoint state dict** (layer count, d_model, MLP head) — not metadata; Task 0.3 doc accounts for every tensor.
- [ ] Router `degraded_prob` parity vs PyTorch fixture (`atol≤2e-3`); clean<0.5, degraded≥0.5.
- [ ] LoRA delta math verified; apply→remove restores base (`atol≤1e-4`).
- [ ] `mega_asr` registered; `mlx_audio.stt.load()` loads base+router+adapter.
- [ ] Routed `generate`/`stream_transcribe` reuse the `qwen3_asr` decode path (no duplication).
- [ ] Real-weights parity: clean→base text, degraded→lora text match reference.
- [ ] `audio_tower` stays unquantized; merge done in fp32 before any quantization.
- [ ] CLI works; READMEs updated; tests green; diagnostics clean.

## Out of scope

- Training / RL recipe (DG-WGPO, A2S-SFT) — inference port only.
- `batch_infer` clean/degraded grouping (single-utterance routing first; batch optimization later).
- New quantization modes beyond existing `mlx_audio.convert`.
