# Mega-ASR Audio-Quality Router — Reference Structure (source of truth)

**Purpose:** Authoritative spec for the MLX port of the Mega-ASR audio-quality router (Phase 2). Reconciles the reference PyTorch source with the **released checkpoint state dict** (the state dict + the checkpoint's embedded metadata config WIN over the source-code defaults).

**Reference source** (cloned `github.com/xzf-thu/Mega-ASR@29b14321`, on pc.lan):
- `src/MegaASR/model/utils/audio_quality.py` — `LogMelSpectrogram`, `PositionalEncoding`, `AttentionPooling`, `ConvFrontend`, `AudioQualityClassifier`, `create_audio_quality_model`.
- `src/MegaASR/model/router.py` — `AudioQualityRouter` (load + inference + threshold).

**Checkpoint:** `ckpt/Mega-ASR/audio_quality_router/best_acc_model.safetensors` (1,172,229 params, 35 tensors). State-dict keys/shapes captured in `tests/stt/mega_asr/fixtures/router_state_dict_keys.json`.

---

## ⚠ Config: checkpoint metadata wins over source defaults

`create_audio_quality_model(config)` is built from `config` read from the **safetensors metadata** `metadata["config"]` → `["model"]` (see `router.py::_load_model`). The source-code *default* args (`d_model=192`, `dim_feedforward=512`) are NOT what the released checkpoint uses. The **actual** values, confirmed by tensor shapes:

| Param | Value | Evidence (state dict) |
|---|---|---|
| `n_mels` | 80 | `frontend.conv.0.weight [128,80,3]` (in=80) |
| `d_model` | **256** | `transformer.norm.weight [256]`, `conv.4.weight [256,128,3]` |
| `nhead` | 4 | from metadata config (not shape-derivable) |
| `dim_feedforward` | **1024** | `transformer.layers.0.linear1.weight [1024,256]` |
| `num_layers` | **1** (hardcoded in `AudioQualityClassifier`, NOT config) | only `transformer.layers.0.*` present |
| `max_len` | 3000 | `pos_encoder.pe [1,850,256]` = `max_len//4+100` = 850 ✓ |
| `num_classes` | 2 | `classifier.3.weight [2,128]` |
| `dropout` | 0.1 | (eval: no-op) |
| `downsample_rate` | 4 | two stride-2 convs |

**MLX port:** read these from the checkpoint metadata `config.model` (fallback to the values above). Do NOT use 192/512.

---

## Pipeline (exact)

`waveform → LogMel(80) → ConvFrontend (4× downsample) → +PositionalEncoding → TransformerEncoder(1 layer, pre-norm) → AttentionPooling → MLP classifier → 2 logits → softmax[1]=degraded_prob → ≥0.5 ⇒ degraded`

### 1. Audio load (`router.py::_load_audio`)
- `soundfile.read(path, always_2d=True)` → mean over channels (mono).
- Resample to 16 kHz with `scipy.signal.resample_poly(audio, 16000//gcd, sr//gcd)` if `sr != 16000`.
- `waveform = torch.from_numpy(audio).float().unsqueeze(0)` → `[1, samples]`.

### 2. LogMelSpectrogram (`audio_quality.py`)
- `torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, win_length=400, n_mels=80, norm="slaney", mel_scale="slaney")`.
  - torchaudio defaults that matter: `power=2.0`, `center=True`, `pad_mode="reflect"`, `window=hann`, `f_min=0.0`, `f_max=8000`.
- `log_mel = clamp(mel, min=1e-10).log10()`; return `(log_mel + 4.0) / 4.0`.
- In `infer`: `mel.squeeze(0).transpose(0,1).unsqueeze(0)` → fed as `[1, T, 80]` (time-major).

**MLX parity:** STFT(n_fft=400, hop=160, win=400 Hann, center+reflect) → `|.|^2` → mel filterbank `librosa.filters.mel(sr=16000, n_fft=400, n_mels=80, fmin=0, fmax=8000, norm="slaney", htk=False)` (htk=False ⇒ slaney mel scale) → `log10(clamp(.,1e-10))` → `(x+4)/4`. Verify against fixture `router_logmel_clean_first10`.

### 3. ConvFrontend (`audio_quality.py::ConvFrontend`)
Input `[B,T,80]` → `transpose(1,2)` → `[B,80,T]`:
```
Conv1d(80, 128, kernel_size=3, stride=2, padding=1)   # d_model//2 = 128
BatchNorm1d(128)
GELU()
Dropout(0.1)
Conv1d(128, 256, kernel_size=3, stride=2, padding=1)  # d_model = 256
BatchNorm1d(256)
GELU()
Dropout(0.1)
```
→ `transpose(1,2)` → `[B, T//4, 256]`.
**MLX notes:** MLX `nn.Conv1d` expects `[B,T,C]` (no manual transpose needed, but match stride/pad). **BatchNorm1d in eval mode** uses `running_mean`/`running_var` (present in state dict as `conv.1.*`, `conv.5.*`); use `mlx.nn.BatchNorm` loaded in eval.

### 4. PositionalEncoding (`audio_quality.py::PositionalEncoding`)
- Standard sinusoidal `pe[1, max_len//4+100, d_model]` = `[1,850,256]` (buffer `pos_encoder.pe`). `x = x + pe[:, :T]`. Dropout (eval no-op).

### 5. TransformerEncoder (`audio_quality.py`)
- `nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)` × **1 layer**, with final `norm=nn.LayerNorm(256)`.
- **norm_first=True ⇒ PRE-norm**: `x = x + attn(norm1(x))`; `x = x + ff(norm2(x))`; then final `transformer.norm`.
- Attention: combined QKV `self_attn.in_proj_weight [768,256]` / `in_proj_bias [768]`, `out_proj [256,256]`. Split in_proj into Q,K,V each `[256,256]`. FFN: `linear1 [1024,256]` → GELU → `linear2 [256,1024]`.
- In eval with `mask=None`, `src_key_padding_mask=None` (single clip, full attention).

### 6. AttentionPooling (`audio_quality.py::AttentionPooling`)
- `query = nn.Linear(256, 1)`; `weights = query(x).squeeze(-1)` `[B,T]` → `softmax(dim=-1)` → `context = bmm(weights[:,None,:], x)` `[B,256]`.

### 7. Classifier (`audio_quality.py`)
```
Linear(256, 128) → GELU → Dropout(0.1) → Linear(128, 2)
```
(state dict: `classifier.0 [128,256]`, `classifier.3 [2,128]`; indices 1=GELU, 2=Dropout.)

### 8. Decision (`router.py::infer`)
- `logits [B,2]` → `softmax(dim=-1)` → `degraded_prob = probs[0,1]` → `is_degraded = degraded_prob >= threshold` (`threshold=0.5`).

---

## State-dict → MLX module map

| State-dict key | Shape | MLX target |
|---|---|---|
| `frontend.conv.0.{weight,bias}` | `[128,80,3]`,`[128]` | Conv1d #1 (transpose weight to MLX `[out,k,in]`) |
| `frontend.conv.1.{weight,bias,running_mean,running_var}` | `[128]`×4 | BatchNorm #1 |
| `frontend.conv.4.{weight,bias}` | `[256,128,3]`,`[256]` | Conv1d #2 |
| `frontend.conv.5.{weight,bias,running_mean,running_var}` | `[256]`×4 | BatchNorm #2 |
| `pos_encoder.pe` | `[1,850,256]` | sinusoidal buffer (recompute or load) |
| `transformer.layers.0.self_attn.in_proj_{weight,bias}` | `[768,256]`,`[768]` | split → q/k/v |
| `transformer.layers.0.self_attn.out_proj.{weight,bias}` | `[256,256]`,`[256]` | attn out |
| `transformer.layers.0.linear1.{weight,bias}` | `[1024,256]`,`[1024]` | FFN up |
| `transformer.layers.0.linear2.{weight,bias}` | `[256,1024]`,`[256]` | FFN down |
| `transformer.layers.0.norm1.{weight,bias}` / `norm2.*` | `[256]` | pre-norm LNs |
| `transformer.norm.{weight,bias}` | `[256]` | final LN |
| `pooling.query.{weight,bias}` | `[1,256]`,`[1]` | attention pooling Linear |
| `classifier.0.{weight,bias}` | `[128,256]`,`[128]` | MLP fc1 |
| `classifier.3.{weight,bias}` | `[2,128]`,`[2]` | MLP fc2 |

`num_batches_tracked` (conv.1/conv.5) are scalar BN counters — ignore on load.

---

## Parity anchors (fixtures, committed `9fc1c86`)
- `reference.json`: `router_logits_clean=[0.5060,-0.6112]` (degraded_prob 0.2465 <0.5 ⇒ clean→base); `router_logits_degraded=[-1.0212,1.3780]` (degraded_prob 0.9168 ≥0.5 ⇒ degraded→LoRA); `router_logmel_clean_first10`.
- Phase-2 acceptance: MLX `logits` match within `atol≤2e-3` and `use_lora` decision matches per clip.
