"""Small deterministic tests; no checkpoint downloads or optional reference packages."""

import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.tts.models.zonos3 import Model, ModelConfig
from mlx_audio.tts.models.zonos3.backbone import Backbone, LMState
from mlx_audio.tts.models.zonos3.config import LMConfig
from mlx_audio.tts.models.zonos3.gdn_kernel import gated_delta, gated_delta_ops
from mlx_audio.tts.models.zonos3.generation import (
    TextTokenizer,
    aligned_codes,
    condition_ids,
    delay_codes,
    output_audio,
    sample_row,
)
from mlx_audio.tts.models.zonos3.weights import Linear, TensorStore, encode_name


def tiny_backbone():
    c = LMConfig(
        n_layers=4,
        max_seqlen=32,
        dim=8,
        head_dim=4,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=12,
        n_experts=3,
        moe_topk=2,
        n_codebooks=3,
        codebook_size=8,
        audio_vocab=10,
        eoa_id=8,
        audio_pad_id=9,
        span_end_id=None,
        text_vocab=20,
        text_extra_tokens=0,
        gdn_heads=2,
        gdn_head_dim=4,
        gdn_kernel=4,
        speaker_embedding_dim=4,
        condition_count=3,
        condition_bucket_counts=(2, 2, 3),
        condition_names=("loudness", "fmax", "speaking_rate"),
        condition_units={},
        condition_edges={
            "loudness": [-20.0],
            "fmax": [10000.0],
            "speaking_rate": [10.0, 20.0],
        },
        rope_theta=10000.0,
        norm_eps=1e-5,
        qk_eps=1e-6,
        loss_softcap=15.0,
        gdn_period=4,
        silence_frame=(0, 1, 2),
        text_tokenizer_spec={},
    )
    specs, tensors = {}, {}
    rng = np.random.default_rng(5)

    def add(name, shape, matrix=False, ones=False):
        specs[name] = {"shape": list(shape), "matrix": matrix, "dtype": "float32"}
        value = (
            np.ones(shape, np.float32)
            if ones
            else rng.normal(0, 0.15, shape).astype(np.float32)
        )
        key = (
            ("matrices." if matrix else "tables.")
            + encode_name(name)
            + (".weight" if matrix else "")
        )
        tensors[key] = mx.array(value)

    for cb in range(c.n_codebooks):
        add(f"audio_embd.{cb}.weight", (c.audio_vocab, c.dim))
    add("text_embd.weight", (21, 8))
    add("speaker_proj.weight", (8, 4), True)
    add("speaker_proj.bias", (8,))
    for i, count in enumerate(c.condition_bucket_counts):
        add(f"cond_embd.{i}.weight", (count + 1, 8))
    add("attnres_q.weight", (9, 8))
    add("output_norm.weight", (8,), ones=True)
    add("output.weight", (30, 8), True)
    for i in range(4):
        p = f"blk.{i}."
        add(p + "attn_norm.weight", (8,), ones=True)
        add(p + "ffn_norm.weight", (8,), ones=True)
        add(p + "attn_output.weight", (8, 8), True)
        if c.uses_gdn(i):
            add(p + "gdn_in.weight", (36, 8), True)
            add(p + "gdn_conv.weight", (24, 4))
            add(p + "gdn_A_log", (2,))
            add(p + "gdn_dt_bias", (2,))
            add(p + "gdn_norm.weight", (4,), ones=True)
        else:
            add(p + "attn_q.weight", (8, 8), True)
            add(p + "attn_k.weight", (4, 8), True)
            add(p + "attn_v.weight", (4, 8), True)
            add(p + "attn_gate.weight", (2, 8), True)
            add(p + "attn_temp", (2,), ones=True)
        if i == 0:
            for name, shape in [
                ("ffn_gate", (12, 8)),
                ("ffn_up", (12, 8)),
                ("ffn_down", (8, 12)),
            ]:
                add(p + name + ".weight", shape, True)
        else:
            add(p + "router_norm.weight", (8,), ones=True)
            add(p + "router.weight", (3, 8), True)
            add(p + "router_bias", (3,))
            add(p + "ffn_latent_down.weight", (4, 8), True)
            add(p + "ffn_latent_up.weight", (8, 4), True)
            for name, shape in [
                ("ffn_gate_exps", (3, 6, 4)),
                ("ffn_up_exps", (3, 6, 4)),
                ("ffn_down_exps", (3, 4, 6)),
            ]:
                add(p + name + ".weight", shape, True)
    backbone = Backbone(c, specs)
    backbone.params.load_weights(list(tensors.items()), strict=True)
    return c, specs, backbone


@pytest.fixture
def tiny():
    return tiny_backbone()


def test_loader_registration():
    from mlx_audio.registry import classify_model
    from mlx_audio.tts.utils import get_model_and_args

    arch, kind = get_model_and_args("zonos3", ["unrelated"])
    assert arch.Model is Model and kind == "zonos3"
    assert classify_model("zonos3") == "tts"


def test_cache_matches_full_forward_and_fork(tiny):
    c, _, m = tiny
    rng = np.random.default_rng(4)
    ids = np.column_stack(
        [rng.integers(0, 8, (10, 3)), rng.integers(0, 20, 10)]
    ).astype(np.int32)
    kw = {
        "speaker_rows": {0: np.ones(4, np.float32)},
        "condition_rows": {1: 0, 2: 1, 3: 2},
    }
    # CPU avoids GEMM/GEMV TF32 differences so this tests state and positions.
    with mx.stream(mx.cpu):
        full = m(ids, LMState(), **kw)
        state = LMState()
        pieces = [m(ids[:5], state, **kw)]
        mx.eval(pieces, state.arrays())
        fork = state.fork()
        pieces.append(m(ids[5:7], state))
        for row in ids[7:]:
            pieces.append(m(row[None], state))
        joined = mx.concatenate(pieces)
        mx.eval(full, joined, state.arrays())
        np.testing.assert_allclose(full, joined, atol=3e-6, rtol=3e-6)
        assert state.position == 10 and fork.position == 5
        other = m(ids[5:], fork)
        mx.eval(other, fork.arrays())
        np.testing.assert_allclose(full[5:], other, atol=3e-6, rtol=3e-6)
    with pytest.raises(ValueError, match="context"):
        m(ids, LMState(position=c.max_seqlen - 1))


def test_strict_native_roundtrip(tiny, tmp_path):
    c, specs, backbone = tiny
    config = ModelConfig(backbone=asdict(c), tensors=specs)
    weights = {"backbone." + k: v for k, v in tree_flatten(backbone.parameters())}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    restored = Model(ModelConfig.from_dict(asdict(config)))
    restored.load_weights(
        list(mx.load(str(tmp_path / "model.safetensors")).items()), strict=True
    )
    ids = mx.zeros((2, 4), mx.int32)
    np.testing.assert_allclose(restored(ids), backbone(ids, LMState()), atol=1e-6)
    with pytest.raises(ValueError):
        restored.load_weights(list(weights.items())[1:], strict=True)


@pytest.mark.parametrize("bits", [4, 8])
def test_packed_dense_and_expert_linear(bits):
    rng = np.random.default_rng(1)
    shape = (3, 8, 32)
    weight = mx.array(rng.normal(size=shape).astype(np.float32))
    packed, scales, biases = mx.quantize(weight, group_size=32, bits=bits)
    m = Linear({"shape": shape, "dtype": "uint32", "bits": bits})
    m.weight, m.scales, m.biases = packed, scales, biases
    dense = mx.dequantize(packed, scales, biases, group_size=32, bits=bits)
    x = mx.array(rng.normal(size=(2, 2, 1, 32)).astype(np.float32))
    choices = mx.array([[2, 0], [1, 2]])
    expected = x @ dense[choices].swapaxes(-1, -2)
    np.testing.assert_allclose(m(x, choices), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("head_dim", [4, 32])
def test_gdn_prefill_matches_sequential_and_state(head_dim):
    rng = np.random.default_rng(4)
    shape = (1, 7, 2, head_dim)
    q, k, v = [mx.array(rng.normal(0, 0.1, shape).astype(np.float32)) for _ in range(3)]
    g = mx.full(shape[:-1], 0.9)
    beta = mx.full(shape[:-1], 0.5)
    state = mx.zeros((1, 2, head_dim, head_dim))
    actual, actual_state = gated_delta(q, k, v, g, beta, state)
    expected, expected_state = gated_delta_ops(q, k, v, g, beta, state)
    mx.eval(actual, actual_state, expected, expected_state)
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    np.testing.assert_allclose(actual_state, expected_state, atol=2e-7)


def test_delay_alignment_and_eos():
    codes = np.arange(15, dtype=np.int32).reshape(5, 3)
    delayed = delay_codes(codes, 99)
    np.testing.assert_array_equal(aligned_codes(delayed, 3), codes)
    np.testing.assert_array_equal(aligned_codes(delayed, 3, start=2), codes[2:])
    np.testing.assert_array_equal(aligned_codes(delayed[:4], 3), codes[:2])
    np.testing.assert_array_equal(aligned_codes(delayed, 3, eos_frame=2), codes[:2])
    assert aligned_codes(delayed[:2], 3).shape == (0, 3)
    np.testing.assert_array_equal(delay_codes(codes, 99, flush=False), delayed[:5])


def test_condition_boundary_and_unknowns(tiny):
    c, _, _ = tiny
    assert condition_ids(c) == [2, 2, 3]
    assert condition_ids(c, values={"loudness": -20.0}) == [1, 2, 3]
    assert condition_ids(c, buckets=[None, 0, 3]) == [2, 0, 3]
    for kw in [
        {"values": {"other": 1}},
        {"values": [float("nan"), None, None]},
        {"buckets": [0, 0, 4]},
        {"values": [1]},
        {"values": [1, 2, 3], "buckets": [0, 0, 0]},
    ]:
        with pytest.raises(ValueError):
            condition_ids(c, **kw)


def test_sampling_masks_delay_and_codebook_repetition(tiny):
    c, _, _ = tiny
    logits = mx.zeros((3, 10))
    logits[:, 9] = 100.0
    logits[:, 8] = 90.0
    logits[:, 4] = 80.0
    token = sample_row(logits, c, step=0, history=[], temperature=0, min_frames=2)
    assert token.tolist() == [4, 9, 9]
    token = sample_row(
        logits,
        c,
        step=5,
        history=[],
        temperature=0,
        min_frames=2,
        has_audio_prefix=True,
    )
    assert token.tolist() == [8, 8, 8]
    logits = mx.zeros((3, 10))
    logits[:, 4] = 10.0
    logits[:, 5] = 8.0
    token = sample_row(
        logits,
        c,
        step=5,
        history=[[4, 4, 4]],
        temperature=0,
        repetition_penalty=2,
        repetition_codebooks=1,
        has_audio_prefix=True,
    )
    assert token.tolist() == [5, 4, 4]


def test_tokenizer_offsets_and_hash(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(
        models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    raw = tok.to_str().encode()
    (tmp_path / "tokenizer.json").write_bytes(raw)
    spec = {
        "kind": "superbpe",
        "offset": 4,
        "bos": 2,
        "eos": 3,
        "vocab_size": 3,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "path": "tokenizer.json",
    }
    assert TextTokenizer(tmp_path, spec, 7).encode("hello world") == [2, 5, 6, 3]
    (tmp_path / "tokenizer.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="SHA"):
        TextTokenizer(tmp_path, spec, 7)


class FakeDecoder:
    hop = 4

    def output_gain(self, db):
        return 1.0

    def decode(self, codes):
        assert isinstance(codes, mx.array)
        assert mx.all((codes >= 0) & (codes < 8)).item()
        return mx.repeat(codes[:, 0].astype(mx.float32) / 10, self.hop)

    def stream(self):
        return SimpleNamespace(
            decode=lambda codes: np.array(self.decode(codes)),
            flush=lambda: np.zeros(0, np.float32),
        )


@pytest.mark.parametrize("silence_frames", [0, 1, 3])
@pytest.mark.parametrize("stream", [False, True])
def test_generate_stops_on_delayed_eos_without_decoding_specials(
    tiny, silence_frames, stream
):
    c, specs, _ = tiny
    model = Model(ModelConfig(backbone=asdict(c), tensors=specs))
    model._tokenizer = SimpleNamespace(encode=lambda t: [2, 4, 3])
    model._decoder = FakeDecoder()

    def forward(ids, state, **kw):
        state.position += len(ids)
        return mx.zeros((len(ids), 3, 10))

    # Four ordinary frames followed by an EOA at the first codebook.
    sequence = np.array([[1, 2, 3]] * 4 + [[8, 2, 3]] + [[1, 2, 3]] * 3, np.int32)
    sequence = delay_codes(sequence, 9, flush=False)
    if silence_frames:
        # Existing silence prefix means generation is already warm.
        sequence = np.array([[1, 2, 3]] * 4 + [[8, 2, 3]] + [[1, 2, 3]] * 3, np.int32)
    with (
        patch.object(Model, "__call__", side_effect=forward),
        patch(
            "mlx_audio.tts.models.zonos3.model.sample_row",
            side_effect=[mx.array(row) for row in sequence],
        ),
    ):
        results = list(
            model.generate(
                "hello",
                voice="none",
                max_tokens=8,
                stream=stream,
                streaming_interval=0.067,
                silence_prefill_frames=silence_frames,
                text_normalization=False,
            )
        )
    audio = np.concatenate([np.array(r.audio) for r in results])
    assert len(audio) == (4 + silence_frames) * 4
    assert results[-1].is_final_chunk
    assert all(r.is_streaming_chunk == stream for r in results)


def test_speaker_embedding_validation(tiny):
    c, specs, _ = tiny
    model = Model(ModelConfig(backbone=asdict(c), tensors=specs))
    np.testing.assert_array_equal(model._embedding(None, None, np.ones(4)), np.ones(4))
    for value in [np.ones(3), np.full(4, np.nan)]:
        with pytest.raises(ValueError, match="finite"):
            model._embedding(None, None, value)


def test_generation_budget_drops_unfinished_delay_tail(tiny):
    c, specs, _ = tiny
    model = Model(ModelConfig(backbone=asdict(c), tensors=specs))
    model._tokenizer = SimpleNamespace(encode=lambda text: [2, 4, 3])
    model._decoder = FakeDecoder()
    codes = np.array([[1, 2, 3]] * 6, np.int32)
    sequence = delay_codes(codes, c.audio_pad_id, flush=False)

    def forward(ids, state, **kwargs):
        state.position += len(ids)
        return mx.zeros((len(ids), 3, 10))

    with (
        patch.object(Model, "__call__", side_effect=forward),
        patch(
            "mlx_audio.tts.models.zonos3.model.sample_row",
            side_effect=[mx.array(row) for row in sequence],
        ),
    ):
        results = list(
            model.generate(
                "hello",
                voice="none",
                max_tokens=6,
                silence_prefill_frames=0,
                text_normalization=False,
            )
        )
    assert results[0].samples == 4 * FakeDecoder.hop
    assert results[0].token_count == 6


def test_native_component_conversion_without_optional_runtime_imports(tmp_path):
    import sys

    from mlx_audio.tts.models.zonos3.convert import convert_component
    from mlx_audio.tts.models.zonos3.weights import NativeFile

    class Source:
        arch = "test"
        reader = SimpleNamespace(fields={"general.architecture": None})

        def names(self):
            return ["layer.weight", "layer.bias"]

        def shape(self, name):
            return (2, 4) if name.endswith("weight") else (2,)

        def meta(self, key):
            return "test"

        def f32(self, name):
            return mx.ones(self.shape(name))

        def weight(self, name):
            return SimpleNamespace(w=mx.ones(self.shape(name)), bits=None, scales=None)

    fake = SimpleNamespace(GGUFFile=lambda path: Source())
    with patch.dict(sys.modules, {"mlx_audio.tts.models.zonos3.gguf": fake}):
        config = convert_component("unused", tmp_path, "codec")
    store = TensorStore(config["tensors"])
    store.load_weights(
        list(mx.load(str(tmp_path / "model.safetensors")).items()), strict=True
    )
    assert store.w["layer.weight"](mx.ones((1, 4))).tolist() == [[4.0, 4.0]]
    assert NativeFile(tmp_path).arch == "test"


def test_output_manifest_uses_downloadable_voice_files(tmp_path):
    from fnmatch import fnmatch

    from mlx_audio.tts.models.zonos3.convert import convert
    from mlx_audio.utils import DEFAULT_ALLOW_PATTERNS

    root = tmp_path / "source" / "gguf"
    voices = root / "voices"
    voices.mkdir(parents=True)
    for name in ["zonos3-ptq.gguf", "zac-keep2.gguf"]:
        (root / name).touch()
    np.save(voices / "Jett_en.embedding.npy", np.ones(192, np.float32))
    out = tmp_path / "mlx"

    def save_component(source, dest, kind, **kwargs):
        dest.mkdir(exist_ok=True, parents=True)
        return {}

    with patch(
        "mlx_audio.tts.models.zonos3.convert.convert_component",
        side_effect=save_component,
    ):
        convert(str(root.parent), str(out), speaker_encoder=False)
    name = json.loads((out / "voices.json").read_text())["jett"]
    assert any(fnmatch("voices/" + name, pattern) for pattern in DEFAULT_ALLOW_PATTERNS)
    with np.load(out / "voices" / name, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["embedding"], np.ones(192))
    assert not (out / "LICENSE").exists()
    assert "Apache 2.0" in (out / "README.md").read_text()


@pytest.mark.parametrize("orig,new,width", [(441, 160, 17), (3, 5, 4)])
@pytest.mark.parametrize("length", [0, 1, 17, 440, 441, 442, 1000])
def test_polyphase_resample_matches_float64_reference(orig, new, width, length):
    from mlx_audio.tts.models.zonos3.audio import polyphase_resample

    rng = np.random.default_rng(length)
    pcm = rng.normal(0, 0.1, length).astype(np.float32)
    kernel = rng.normal(0, 0.01, (new, 2 * width + orig))
    total = -(-new * length // orig)
    expected = np.zeros(total, dtype=np.float64)
    # Independent scalar output indexing checks phase order and both padded edges.
    for i in range(total):
        block, phase = divmod(i, new)
        indices = block * orig + np.arange(kernel.shape[1]) - width
        valid = (indices >= 0) & (indices < length)
        expected[i] = kernel[phase, valid] @ pcm[indices[valid]].astype(np.float64)
    with mx.stream(mx.cpu):
        taps = mx.array(kernel, dtype=mx.float64)
    # Invoke from the ordinary GPU stream: float64 work must stay on the CPU.
    with mx.stream(mx.gpu):
        output = polyphase_resample(mx.array(pcm), taps, orig, new, width)
        consumed = output + 0.0
    mx.eval(consumed)
    assert isinstance(output, mx.array) and output.dtype == mx.float32
    assert output.shape == (total,)
    np.testing.assert_allclose(
        consumed, expected.astype(np.float32), rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("channels", [1, 3, 10])
@pytest.mark.parametrize("length", [0, 1, 7])
@pytest.mark.parametrize("flush", [False, True])
def test_mlx_delay_pattern_boundary_lengths(channels, length, flush):
    codes = np.arange(length * channels, dtype=np.int32).reshape(length, channels)
    expected = np.full(
        (length + (channels - 1 if flush else 0), channels), 999, np.int32
    )
    for t in range(len(expected)):
        for cb in range(channels):
            if 0 <= t - cb < length:
                expected[t, cb] = codes[t - cb, cb]
    actual = delay_codes(mx.array(codes), 999, flush=flush)
    assert isinstance(actual, mx.array) and actual.dtype == mx.int32
    np.testing.assert_array_equal(actual, expected)
    aligned = aligned_codes(actual, channels)
    assert isinstance(aligned, mx.array) and aligned.dtype == mx.int32
    np.testing.assert_array_equal(
        aligned, codes[: max(0, len(expected) - channels + 1)]
    )


def test_mlx_repetition_duplicates_invalid_ids_and_zero(tiny):
    c, _, _ = tiny
    logits = mx.full((3, 10), -20.0)
    logits[:, 0] = 10.0
    logits[:, 1] = 7.0
    history = mx.array([[0, -1, 8], [9, -1, 8], [0, 8, 7]])
    token = sample_row(
        logits,
        c,
        step=5,
        history=history,
        temperature=0,
        repetition_penalty=2,
        repetition_codebooks=1,
        has_audio_prefix=True,
    )
    assert token.tolist() == [1, 0, 0]
    # The negative-logit penalty multiplies once, regardless of duplicate count.
    logits = mx.full((3, 10), -20.0)
    logits[:, 0] = -2.0
    logits[:, 1] = -3.0
    history = mx.array([[0, 0, 0], [0, 0, 0]])
    token = sample_row(
        logits,
        c,
        step=5,
        history=history,
        temperature=0,
        repetition_penalty=2,
        repetition_codebooks=-1,
        has_audio_prefix=True,
    )
    assert token.tolist() == [1, 1, 1]


@pytest.mark.parametrize("gain", [0.0, 0.5, 1.0, 2.0])
def test_mlx_output_limiter_matches_numpy(gain):
    knee = 10 ** (-1 / 20)
    values = np.array(
        [-100.0, -1.0, -knee, -0.1, 0.0, 0.1, knee, 1.0, 100.0], np.float32
    )
    scaled = values * gain
    magnitude = np.abs(scaled)
    expected = np.where(
        magnitude > knee,
        np.sign(scaled)
        * (knee + (1 - knee) * np.tanh((magnitude - knee) / (1 - knee))),
        scaled,
    )
    actual = output_audio(mx.array(values), gain)
    assert isinstance(actual, mx.array) and actual.dtype == mx.float32
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
    assert output_audio(mx.zeros((0,)), gain).shape == (0,)


@pytest.mark.parametrize("silence_frames", [0, 1, 3])
@pytest.mark.parametrize("stop_column", [0, 1, 2])
def test_mlx_generation_aligns_stop_in_every_codebook(
    tiny, silence_frames, stop_column
):
    c, specs, _ = tiny
    model = Model(ModelConfig(backbone=asdict(c), tensors=specs))
    model._tokenizer = SimpleNamespace(encode=lambda text: [2, 4, 3])
    model._decoder = FakeDecoder()
    codes = np.array([c.silence_frame] * silence_frames + [[1, 2, 3]] * 8, np.int32)
    codes[silence_frames + 4, stop_column] = c.eoa_id
    delayed = delay_codes(codes, c.audio_pad_id)[silence_frames:]

    def forward(ids, state, **kwargs):
        assert isinstance(ids, mx.array) and ids.dtype == mx.int32
        state.position += len(ids)
        return mx.zeros((len(ids), c.n_codebooks, c.audio_vocab))

    def sample(*args, history, **kwargs):
        assert isinstance(history, mx.array) and history.dtype == mx.int32
        return delayed[kwargs["step"]]

    with (
        patch.object(Model, "__call__", side_effect=forward),
        patch("mlx_audio.tts.models.zonos3.model.sample_row", side_effect=sample),
    ):
        result = next(
            model.generate(
                "hello",
                voice="none",
                max_tokens=len(delayed),
                silence_prefill_frames=silence_frames,
                text_normalization=False,
            )
        )
    assert result.samples == (silence_frames + 4) * FakeDecoder.hop


@pytest.mark.parametrize("frames", [1, 3, 12])
def test_mlx_istft_envelope_matches_numpy_reference(frames):
    from mlx_audio.tts.models.zonos3.codec import CodecDecoder

    decoder = CodecDecoder.__new__(CodecDecoder)
    decoder.n_fft = [8, 24]
    decoder.head_hop = 2
    decoder.max_log_mag = 4.6
    decoder.windows = [
        mx.array(np.hanning(n + 1)[:-1].astype(np.float32)) for n in decoder.n_fft
    ]
    rng = np.random.default_rng(frames)
    for branch, n_fft in enumerate(decoder.n_fft):
        spec = rng.normal(0, 0.2, (frames, n_fft + 2)).astype(np.float32)
        bins = n_fft // 2 + 1
        spectrum = np.exp(spec[:, :bins]) * np.exp(1j * spec[:, bins:])
        window = np.array(decoder.windows[branch])
        inverse = np.fft.irfft(spectrum, n=n_fft, axis=-1) * window
        output = np.zeros((frames - 1) * decoder.head_hop + n_fft)
        envelope = np.zeros_like(output)
        for frame in range(frames):
            start = frame * decoder.head_hop
            output[start : start + n_fft] += inverse[frame]
            envelope[start : start + n_fft] += window**2
        expected = (output / np.maximum(envelope, 1e-12))[
            n_fft // 2 : n_fft // 2 + frames * decoder.head_hop
        ]
        with patch(
            "mlx_audio.tts.models.zonos3.codec.np.array",
            side_effect=AssertionError("NumPy in iSTFT"),
        ):
            actual = decoder._istft(mx.array(spec), branch)
            mx.eval(actual)
        assert isinstance(actual, mx.array) and actual.dtype == mx.float32
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_nonstreaming_codec_mlx_contract_and_invalid_codes():
    from mlx_audio.tts.models.zonos3.codec import CodecDecoder

    decoder = CodecDecoder.__new__(CodecDecoder)
    decoder.n_codebooks, decoder.codebook_size, decoder.hop = 3, 8, 4
    decoder._trunk = lambda ids: (ids, ids)
    decoder._istft = lambda spec, branch: mx.ones(
        (spec.shape[0] * decoder.hop - 1,), mx.float32
    )
    decoder._mix = lambda a, b: a
    decoder.rs = (1, 1, 0)
    decoder.rs_kernel = mx.ones((1, 1))
    with patch(
        "mlx_audio.tts.models.zonos3.codec.np.asarray",
        side_effect=AssertionError("NumPy in decode"),
    ):
        actual = decoder.decode(mx.zeros((3, 3), mx.int32))
        mx.eval(actual)
    assert isinstance(actual, mx.array) and actual.dtype == mx.float32
    assert actual.shape == (12,)
    np.testing.assert_array_equal(actual, [1.0] * 11 + [0.0])
    assert decoder.decode(mx.zeros((0, 3), mx.int32)).shape == (0,)
    for codes in [mx.full((1, 3), 8), mx.full((1, 3), -1)]:
        with pytest.raises(ValueError, match="non-codec"):
            decoder.decode(codes)
    with pytest.raises(ValueError, match="frames"):
        decoder.decode(mx.zeros((1, 2), mx.int32))
