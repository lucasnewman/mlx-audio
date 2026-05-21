# Mega-ASR → mlx-audio: Porting Research

**Date**: 2026-05-21
**Branch**: `feature/mega-asr` (proposed)
**Sources**:
- Paper: https://arxiv.org/html/2605.19833v1
- Project page: https://xzf-thu.github.io/Mega-ASR/
- Repo: https://github.com/xzf-thu/Mega-ASR (commit `29b14321ae60e320d4084e33759415bea80102cd`)
- Weights: https://huggingface.co/zhifeixie/Mega-ASR
- Base model: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Existing MLX backbone: `mlx_audio/stt/models/qwen3_asr/`

---

## 1. Mega-ASR Overview

| Property | Value |
|---|---|
| What it is | **Robustness post-training package on top of Qwen3-ASR-1.7B**, not a new architecture |
| Backbone | Qwen3-ASR-1.7B (AuT audio encoder + Qwen3 LLM decoder) |
| Novelty (paper) | Voices-in-the-wild-2M compound acoustic simulation (7 atomic → 54 compound), A2S-SFT progressive training, WER-gated RL (DG-WGPO) — **all training-time; irrelevant to inference** |
| Inference-time delta | (1) a 23.1M-param LoRA adapter, (2) a 1.17M-param raw-audio quality router, (3) in-place LoRA add/subtract switching |
| Languages | Backbone supports 52 languages/dialects; Mega-ASR eval focuses EN + Mandarin |
| Inference deps | Plain PyTorch/transformers/safetensors/soundfile/scipy/huggingface_hub/qwen_asr — **no vLLM, no FlashAttention, no CUDA kernels** in the repo |
| License | **Apache-2.0** (repo, HF weights, and base Qwen3-ASR) |

Headline robustness numbers (Mega-ASR vs Qwen3-ASR): CHiME-4/VOiCES/NOIZEUS avg **6.70 vs 7.93** (Whisper-Large-v3 10.72); NOIZEUS 0 dB **19.80 vs 23.97**; Voices-in-the-Wild-Bench **2.73/4.57 vs** Whisper **8.91/14.79**. Clean (router on): LibriSpeech **1.63/3.37**, FLEURS zh/en **3.86/3.17**.

---

## 2. The Delta over vanilla Qwen3-ASR

### 2.1 LoRA adapter (`mega-asr-merged/`)

PEFT LoRA — **adapter-only, not a standalone checkpoint**. Must be applied on top of base Qwen3-ASR-1.7B.

- Files: `adapter_config.json`, `adapter_model.safetensors`, `mega_lora_blocks.json`
- `adapter_config.json`: `peft_type=LORA`, `task_type=CAUSAL_LM`, `r=24`, `lora_alpha=24`, `lora_dropout=0.0`, `fan_in_fan_out=false`, `bias=none`, `inference_mode=true`, `target_modules=".*"`, plus `rank_pattern` (539 entries) and `alpha_pattern` (539 entries) — **per-module rank/alpha varies**.
- `adapter_model.safetensors`: 92,538,896 bytes (~88.3 MiB), **1078 tensors** (LoRA A/B pairs), 23,097,344 elements.
- Key shape: `base_model.model.thinker.<module>.lora_A.weight` / `lora_B.weight`.
- Covers **both encoder and decoder**, e.g.:
  - `base_model.model.thinker.audio_tower.conv_out.lora_{A,B}.weight`
  - `base_model.model.thinker.audio_tower.layers.0.self_attn.q_proj.lora_{A,B}.weight`
  - `base_model.model.thinker.model.layers.N.self_attn.q_proj.lora_{A,B}.weight`

### 2.2 Audio-quality router (`audio_quality_router/`)

Tiny **standalone** classifier — not LoRA, does **not** consume ASR encoder hidden states.

- Files (reference): `src/MegaASR/model/router.py`, `src/MegaASR/model/utils/audio_quality.py`; weights `audio_quality_router/best_acc_model.safetensors`.
- Input: raw waveform (soundfile) → mono → resample 16 kHz → LogMelSpectrogram.
- Frontend: `sample_rate=16000`, `n_mels=80`, `n_fft=400`, `hop_length=160`, `win_length=400`, **Slaney mel + Slaney norm**, log norm `(log10(mel)+4)/4`.
- Architecture (`transformer_mini`): metadata advertises `d_model=256`, `nhead=4`, `num_layers=4`, `dim_feedforward=1024`, `dropout=0.1`, `max_len=3000`, `num_classes=2`, `pooling="attention"`.
  - **⚠ Metadata is unreliable.** The released `best_acc_model.safetensors` state dict actually contains only `transformer.layers.0.*` + `transformer.norm.*` and a **2-layer MLP head** (`classifier.0.*`, `classifier.3.*`), not a single linear. **The state-dict tensor names/shapes are the source of truth** for the port (verified and reconciled in plan Task 0.3 — read dims/layer count from the checkpoint, never from metadata).
- Pipeline: `waveform → 80-bin log-mel → 2×Conv1d(stride=2) [4× downsample] → positional encoding → TransformerEncoder (layer count per checkpoint) → attention pooling → MLP head → 2 logits`.
- Output: 2 logits; `degraded_prob = softmax(logits)[1]`; threshold **0.5**.
- Size: 4,693,444 bytes, **1,172,229 params**, 35 tensors.

### 2.3 Routing / switching logic

One base model loaded once; LoRA deltas toggled in place.

- Reference: `src/MegaASR/model/megaASR.py` (`infer`, `batch_infer`), `utils/lora_switch.py` (`_set_lora`).
- Flow: `infer.py` builds `MegaASR(...)` → `_route(audio)` runs router → if `degraded_prob ≥ 0.5` then `use_lora=True` → `_set_lora(True)` **adds** adapter deltas to base weights → `asr.infer(...)` → for clean audio deltas are **subtracted** and the base path runs.
- Not two loaded ASR models; not a decode-parameter gate.

---

## 3. What already exists in mlx-audio (ground truth)

`mlx_audio/stt/models/qwen3_asr/qwen3_asr.py` (1425 lines) already implements the full backbone:

| Mega-ASR component | mlx-audio analog | Status |
|---|---|---|
| AuT audio encoder | `AudioEncoder` — Conv2d×3 stride-2 (8× downsample), 24 layers, 128 mel, block/window attention, `ln_post`/`proj1`/`proj2` | ✅ exists |
| Qwen3 LLM decoder | `TextModel` — GQA, q/k-norm, RoPE, SwiGLU, `KVCache` (mlx_lm) | ✅ exists |
| Audio frontend | `post_load_hook` → `WhisperFeatureExtractor` @16 kHz, 128 mel (matches Mega-ASR 8×→12.5 Hz) | ✅ exists |
| Weight sanitize | `Qwen3ASRModel.sanitize()` strips `thinker.`, drops tied `lm_head`, transposes conv2d NCHW→NHWC | ✅ exists |
| Quant policy | `model_quant_predicate` — never quantizes `audio_tower` | ✅ exists |
| Generate / stream | `generate`, `stream_transcribe`, chunking (`split_audio_into_chunks`) | ✅ exists |
| LoRA adapter | — | ❌ missing |
| LoRA add/subtract switch | — | ❌ missing |
| Audio-quality router (mel-80 + transformer_mini) | — | ❌ missing |

Loader: `mlx_audio/stt/utils.py::MODEL_REMAPPING` registers `qwen3_asr`; `mlx_audio/utils.py::base_load_model` resolves `config["model_type"]` → module → `Model(ModelConfig.from_dict(config))` → `sanitize` → quant → `load_weights` → `post_load_hook`. There is precedent for config-based model_type override (`llama`+`acoustic_dim` → `tada`).

---

## 4. Weight key mapping (near 1:1)

After stripping the PEFT wrapper prefix `base_model.model.thinker.`, adapter module paths align directly with our MLX module tree:

| Adapter module (stripped) | MLX path |
|---|---|
| `audio_tower.conv_out` | `audio_tower.conv_out` (nn.Linear) |
| `audio_tower.layers.N.self_attn.{q,k,v,out}_proj` | same |
| `audio_tower.layers.N.{fc1,fc2}` | same |
| `model.layers.N.self_attn.{q,k,v,o}_proj` | same |
| `model.layers.N.mlp.{gate,up,down}_proj` | same |

All LoRA targets are `nn.Linear` (bias-free) — LoRA delta is `scaling * (B @ A)` with `scaling = alpha_module / r_module` (per `rank_pattern`/`alpha_pattern`). This makes both offline merge and runtime switching mechanical.

---

## 5. Porting options

**Option A — robust-merged (minimal).** Offline-merge LoRA into the base (PEFT in PyTorch) → `python -m mlx_audio.convert` → load via the **existing** `qwen3_asr` path. Zero new model code; always runs the robust path. Good first milestone; slight expected regression on clean speech (the reason the router exists).

**Option B — full fidelity (chosen).** New `mlx_audio/stt/models/mega_asr/` module reusing `qwen3_asr` classes, plus the ported router and in-place LoRA add/subtract switching, registered as `mega_asr`. Reproduces the dynamic clean/degraded gating.

---

## 6. Recommended implementation plan (Option B)

See `docs/plans/2026-05-21-mega-asr-mlx-port.md` for the task-by-task plan. Phase summary:

```
P0  Worktree + branch + parity fixtures (PyTorch reference outputs) + router state-dict dump (truth)
P1  Weight conversion: adapter → MLX (respect rank_pattern), router → MLX
P2  Router port: LogMel-80 frontend + transformer_mini (layers/dims from checkpoint) + MLP head (+ parity)
P3  LoRA apply/switch in MLX: per-module scaling, add/subtract, rank_pattern (+ test)
P4  MegaASR model: wrap qwen3_asr + router + switch + generate; register "mega_asr"
P5  Integration: CLI, README, end-to-end + WER tests
```

Effort estimate: ~7–11 working days for full Option B; P1 (robust-merged) usable in 1–2 days.

---

## 7. Proposed directory structure

```
mlx_audio/stt/models/mega_asr/
├── __init__.py
├── config.py            # MegaASRConfig (wraps qwen3_asr ModelConfig + router cfg + lora cfg)
├── router.py            # AudioQualityRouter: LogMel-80 + transformer_mini + attention pooling + MLP head
├── lora.py              # build_deltas(), apply/remove (add/subtract) honoring rank_pattern
├── mega_asr.py          # Model: reuses Qwen3ASRModel, _route(), apply/remove deltas, generate()/stream
└── README.md

mlx_audio/stt/utils.py   # MODEL_REMAPPING += {"mega_asr": "mega_asr"}
tests/stt/mega_asr/      # router parity, lora math, end-to-end shapes
```

---

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| `rank_pattern` (539 modules, varying r/alpha) | High | Read per-module r/alpha programmatically; never hardcode rank; assert shapes |
| LoRA on encoder **and** decoder | Medium | Apply to all listed modules, not just decoder |
| Router mel ≠ ASR mel (80/Slaney vs 128/Whisper) | High | Implement a separate Slaney log-mel; parity-test `degraded_prob` against PyTorch |
| **Router metadata ≠ released state dict** (metadata says 4 layers + linear head; checkpoint has 1 layer + MLP head) | High | State dict is source of truth; dump keys/shapes first (plan Task 0.3); drive impl from tensors |
| Exact `transformer_mini` config (norm_first, activation, pos-enc, pooling form) | Medium | Read `utils/audio_quality.py` verbatim before coding; fixture-based parity |
| Merge/quant precision | Medium | Merge in float32, cast after; merge **before** quantization |
| Preserve `audio_tower` no-quant policy | Low | Reuse existing `model_quant_predicate` |
| Switching cost (add/subtract per utterance) | Low | Precompute deltas once; in-place tree update; batch by route in `batch_infer` analog |

---

## 9. License

Apache-2.0 across the board (Mega-ASR repo, `zhifeixie/Mega-ASR` weights, `Qwen/Qwen3-ASR-1.7B`). No licensing blocker for inclusion in mlx-audio (MIT).

---

## 10. Next steps

1. Dump the router checkpoint state dict (keys + shapes) and reconcile against `src/MegaASR/model/utils/audio_quality.py` — the state dict governs (plan Task 0.3).
2. Generate parity fixtures (router `degraded_prob` + transcriptions for clean & degraded clips) from the PyTorch reference.
3. Execute `docs/plans/2026-05-21-mega-asr-mlx-port.md` task-by-task.
