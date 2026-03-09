"""Tests for DeepFilterNet STS model."""

from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import numpy as np


class TestDeepFilterNetConfig(unittest.TestCase):
    """Config serialization/default tests."""

    def test_defaults(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNetConfig

        cfg = DeepFilterNetConfig()
        self.assertEqual(cfg.sample_rate, 48000)
        self.assertEqual(cfg.fft_size, 960)
        self.assertEqual(cfg.hop_size, 480)
        self.assertEqual(cfg.nb_erb, 32)
        self.assertEqual(cfg.nb_df, 96)
        self.assertEqual(cfg.freq_bins, 481)

    def test_from_dict_and_to_dict(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNetConfig

        d = {"sample_rate": 44100, "nb_df": 64, "df_order": 3}
        cfg = DeepFilterNetConfig.from_dict(d)
        self.assertEqual(cfg.sample_rate, 44100)
        self.assertEqual(cfg.nb_df, 64)
        self.assertEqual(cfg.df_order, 3)

        out = cfg.to_dict()
        self.assertEqual(out["sample_rate"], 44100)
        self.assertEqual(out["nb_df"], 64)
        self.assertEqual(out["df_order"], 3)


class TestDeepFilterNetForward(unittest.TestCase):
    """Forward-pass shape tests for DF2/DF3 and DF1 backends."""

    def _run_forward(self, config, model_cls):
        model = model_cls(config)
        t = 8
        f = config.fft_size // 2 + 1
        spec = mx.zeros((1, 1, t, f, 2), dtype=mx.float32)
        feat_erb = mx.zeros((1, 1, t, config.nb_erb), dtype=mx.float32)
        feat_df = mx.zeros((1, 1, t, config.nb_df, 2), dtype=mx.float32)

        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_df)
        mx.eval(spec_e, m, lsnr, df_coefs)

        self.assertEqual(spec_e.shape, spec.shape)
        self.assertEqual(m.shape[0], 1)
        self.assertEqual(lsnr.shape[0], 1)
        self.assertEqual(df_coefs.shape[0], 1)

    def test_df2_like_forward(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNet2Config
        from mlx_audio.sts.models.deepfilternet.network import DfNet

        self._run_forward(DeepFilterNet2Config(), DfNet)

    def test_df3_like_forward(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNet3Config
        from mlx_audio.sts.models.deepfilternet.network import DfNet

        self._run_forward(DeepFilterNet3Config(), DfNet)

    def test_df1_forward(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNetConfig
        from mlx_audio.sts.models.deepfilternet.network_df1 import DfNetV1

        self._run_forward(DeepFilterNetConfig(), DfNetV1)


class TestDeepFilterNetRuntimeHelpers(unittest.TestCase):
    """Tests for runtime helper behavior without external weights."""

    def test_vorbis_window_shape(self):
        from mlx_audio.sts.models.deepfilternet.model import DeepFilterNetModel

        w = DeepFilterNetModel._vorbis_window(960)
        mx.eval(w)
        self.assertEqual(w.shape[0], 960)
        self.assertGreaterEqual(float(mx.min(w)), 0.0)
        self.assertLessEqual(float(mx.max(w)), 1.0)

    def test_resolve_model_dir_raises_for_missing(self):
        from mlx_audio.sts.models.deepfilternet.model import resolve_model_dir

        with self.assertRaises(FileNotFoundError):
            resolve_model_dir("/definitely/not/a/model/dir")

    def test_resolve_model_dir_from_existing_dir(self):
        from mlx_audio.sts.models.deepfilternet.model import resolve_model_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "my_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            self.assertEqual(resolve_model_dir(str(model_dir)), model_dir.resolve())

    def test_resolve_model_dir_rejects_file(self):
        from mlx_audio.sts.models.deepfilternet.model import resolve_model_dir

        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "model.safetensors"
            file_path.touch()
            with self.assertRaises(ValueError):
                resolve_model_dir(str(file_path))

    def test_streamer_rejects_df1_backend(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNetConfig
        from mlx_audio.sts.models.deepfilternet.model import DeepFilterNetModel
        from mlx_audio.sts.models.deepfilternet.network_df1 import DfNetV1

        cfg = DeepFilterNetConfig()
        runtime = DeepFilterNetModel(model=DfNetV1(cfg), config=cfg, model_dir=Path("."))
        with self.assertRaises(NotImplementedError):
            runtime.create_streamer()

    def test_streamer_df3_runs(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNet3Config
        from mlx_audio.sts.models.deepfilternet.model import DeepFilterNetModel
        from mlx_audio.sts.models.deepfilternet.network import DfNet

        cfg = DeepFilterNet3Config()
        runtime = DeepFilterNetModel(model=DfNet(cfg), config=cfg, model_dir=Path("."))
        streamer = runtime.create_streamer()

        # 20 ms of silence at 48 kHz.
        x = np.zeros((960,), dtype=np.float32)
        y1 = streamer.process_chunk(x, is_last=False)
        y2 = streamer.flush()
        self.assertIsInstance(y1, np.ndarray)
        self.assertIsInstance(y2, np.ndarray)


if __name__ == "__main__":
    unittest.main()
