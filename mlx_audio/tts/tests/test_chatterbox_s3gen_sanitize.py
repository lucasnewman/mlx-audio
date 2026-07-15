import re
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from ..models.chatterbox.s3gen.s3gen import S3Token2Wav


def _to_pytorch_key(mlx_key: str) -> str:
    """Turn an MLX conformer param key back into its PyTorch (nn.ModuleList) form.

    MLX exposes each conformer block as an attribute (encoders_0), whereas
    PyTorch stores them in an nn.ModuleList (encoders.0).
    """
    key = re.sub(r"\.up_encoders_(\d+)\.", r".up_encoders.\1.", mlx_key)
    key = re.sub(r"\.encoders_(\d+)\.", r".encoders.\1.", key)
    return key


class ChatterboxS3GenSanitizeTest(unittest.TestCase):
    """Regression test for the S3Gen conformer block naming.

    The flow-encoder conformer blocks are stored in PyTorch nn.ModuleLists, so
    their block index is dotted (flow.encoder.encoders.0). The MLX modules expose
    each block as an attribute (flow.encoder.encoders_0). sanitize() must rename
    the dotted index; otherwise every conformer weight is dropped by its
    should_keep filter and the blocks keep their random initialization, which
    corrupts the encoder conditioning (audible as distorted pronunciation).
    """

    def setUp(self):
        self.model = S3Token2Wav()
        self.model_params = dict(tree_flatten(self.model.parameters()))
        self.conformer_keys = [
            k
            for k in self.model_params
            if ".encoders_" in k or ".up_encoders_" in k
        ]
        # Guard: the model really does have conformer blocks to protect.
        self.assertGreater(len(self.conformer_keys), 0)

    def test_pytorch_conformer_keys_are_kept(self):
        # Build PyTorch-named weights (dotted block index) with model shapes.
        pytorch_weights = {
            _to_pytorch_key(k): mx.zeros(v.shape)
            for k, v in self.model_params.items()
            if k in self.conformer_keys
        }
        # Sanity: the input really is in the dropped-by-default dotted form.
        self.assertTrue(any(".encoders." in k for k in pytorch_weights))

        sanitized = self.model.sanitize(pytorch_weights)

        for k in self.conformer_keys:
            self.assertIn(
                k,
                sanitized,
                f"conformer weight {k} was dropped by sanitize()",
            )
            self.assertEqual(sanitized[k].shape, self.model_params[k].shape)

    def test_sanitize_is_idempotent_for_conformer(self):
        # Already-converted (underscore) keys must be preserved unchanged.
        mlx_weights = {
            k: mx.zeros(self.model_params[k].shape) for k in self.conformer_keys
        }
        sanitized = self.model.sanitize(mlx_weights)
        for k in self.conformer_keys:
            self.assertIn(k, sanitized)
            self.assertEqual(sanitized[k].shape, self.model_params[k].shape)


if __name__ == "__main__":
    unittest.main()
