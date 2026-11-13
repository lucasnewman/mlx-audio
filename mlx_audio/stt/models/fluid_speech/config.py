from dataclasses import dataclass

from mlx_audio.stt.models.parakeet.parakeet import ParakeetTDTArgs
from mlx_audio.utils import from_dict


@dataclass
class ModelConfig(ParakeetTDTArgs):
    fluid_speech: dict
    model_type: str = "fluid_speech"
    model_path: str | None = None

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        result = from_dict(cls, config)
        result.model_type = "fluid_speech"
        # NeMo normalizes only real frames and keeps one zero-padded frame for
        # subsampling. Fluid Speech's extra block sees that frame too.
        result.preprocessor.normalize_valid_frames = True
        result.encoder.mask_padding = True
        spec = result.fluid_speech
        if (
            spec.get("insert") != "pre"
            or not isinstance(spec.get("insert_dim"), int)
            or spec["insert_dim"] <= 0
            or not isinstance(spec.get("insert_kernel"), int)
            or spec["insert_kernel"] <= 0
            or spec["insert_kernel"] % 2 == 0
        ):
            raise ValueError(
                "Fluid Speech requires a pre-encoder block with an odd kernel"
            )
        if (
            result.encoder.subsampling != "dw_striding"
            or result.encoder.causal_downsampling
        ):
            raise ValueError("Fluid Speech requires noncausal dw_striding subsampling")
        if result.decoding.model_type != "tdt" or not (
            result.decoding.greedy and result.decoding.greedy.get("max_symbols", 0) > 0
        ):
            raise ValueError(
                "Fluid Speech requires TDT decoding with positive max_symbols"
            )
        durations = result.decoding.durations
        if (
            not durations
            or len(durations) != result.joint.num_extra_outputs
            or any(type(d) is not int or d < 0 for d in durations)
            or not any(durations)
        ):
            raise ValueError("Fluid Speech requires valid nonnegative TDT durations")
        return result
