# Reverb audio benchmarks

Measured on 2026-11-19 on an Apple M5 Max with 128 GiB unified memory, macOS 27.2, Python 3.13.16, and MLX 0.32.4. These are measurements of this MLX port, not a comparison with the upstream ExecuTorch implementation.

## Method

- Each variant runs in a separate process, sequentially. One discarded warmup and five timed repetitions per workload; seed 42 for every run.
- Timing covers text preparation, prefix prefill, 16 unmasking steps (unless stated otherwise), and the full audio decode. MLX audio evaluation is synchronized before the timer stops. Loading, WAV writes, and ASR are excluded.
- The clone workload also includes loading and encoding the reference WAV. It re-encodes the distributed standard male voice prompt; it is not an independent speaker-similarity benchmark.
- All outputs use the same standard male reference, a 7.5-second requested duration (7.52 seconds / 180,480 samples after frame rounding), and 24 kHz mono audio.
- Memory is the maximum MLX active-allocation peak across measured runs, including the loaded text model and FP32 Higgs codec. It is not process RSS.
- Reverb source revision: `6957eceee689b1accf43e8b2a064587de8942743`; Higgs codec revision: `0e3a9177c2991016b8e5e17d23aa474ee665bef2`. The port is based on main `c4c3cb82f0d53dd3f175aea77eb66238497ad845`.

## English, 16 steps

| Variant | Model weights | Median latency (range) | Audio / wall time | RTF (wall / audio) | Peak MLX memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 594.5 MB | 126.5 ms (126.2–127.9) | 59.5× | 0.0168 | 3.24 GB |
| 8BIT | 315.9 MB | 168.0 ms (167.8–168.8) | 44.8× | 0.0223 | 2.97 GB |
| 4BIT | 167.3 MB | 167.3 ms (166.4–167.4) | 44.9× | 0.0222 | 2.82 GB |

Each bundle also contains the same **805.7 MB** FP32 codec. Quantization reduces model size and runtime memory here; dense BF16 is faster for this workload.

## English sampling-step comparison

| Steps | BF16 | 8-bit | 4-bit |
| --- | ---: | ---: | ---: |
| 4 | 51.1 ms | 63.3 ms | 63.3 ms |
| 8 | 76.2 ms | 99.2 ms | 98.6 ms |
| 16 | 126.5 ms | 168.0 ms | 167.3 ms |
| 32 | 226.4 ms | 310.1 ms | 306.5 ms |

All measured step settings returned the expected transcript on this single English sentence. This does not establish equivalent prosody or general quality at fewer steps. The implementation default remains 16.

## Reference WAV encoding + generation

| Variant | Median time | Audio / wall time |
| --- | ---: | ---: |
| BF16 | 145.6 ms | 51.7× |
| 8BIT | 187.4 ms | 40.1× |
| 4BIT | 186.5 ms | 40.3× |

## Multilingual intelligibility check

Parakeet TDT 0.6B v3 (revision `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`) transcribed the first measured waveform from each case. Text is NFKC-normalized, case-folded, and split into Unicode word tokens; punctuation is ignored.

| Language | Reference words | BF16 WER | 8-bit WER | 4-bit WER |
| --- | ---: | ---: | ---: | ---: |
| EN | 17 | 0.0% | 0.0% | 0.0% |
| DE | 16 | 0.0% | 0.0% | 0.0% |
| FR | 17 | 0.0% | 0.0% | 0.0% |
| ES | 17 | 0.0% | 0.0% | 0.0% |
| IT | 13 | 0.0% | 0.0% | 0.0% |
| PL | 14 | 0.0% | 0.0% | 0.0% |
| PT | 16 | 0.0% | 0.0% | 0.0% |

The seven default-step prompts contain 110 reference words per variant. All 33 saved generated utterances—including the English step sweep and WAV-cloning cases—had zero word errors. These are short, hand-selected smoke checks, not a standardized WER corpus, a listening study, or a voice-similarity measurement. Timing repetitions use the same seed and are not independent quality samples.

Exact parity with Reverb’s optimized upstream sampler remains unverified. The source link supplied by the model card returns 404; this port uses the configurable OmniVoice unmasking schedule with distilled guidance and cached prefix attention.

## Reproduce

Convert the three bundles using the [conversion instructions](README.md#conversion), then run:

```bash
python examples/reverb_benchmark.py \
  --model bf16=/path/to/Reverb-TTS-bf16 \
  --model 8bit=/path/to/Reverb-TTS-8bit \
  --model 4bit=/path/to/Reverb-TTS-4bit \
  --output output/reverb-benchmark \
  --repeats 5 \
  --asr-model mlx-community/parakeet-tdt-0.6b-v3
```

The output contains WAVs, per-variant JSON, combined `results.json`, and `results.csv`. The JSON records every timing, model-file hashes, loading times, hardware/runtime metadata, transcripts, and waveform statistics.
