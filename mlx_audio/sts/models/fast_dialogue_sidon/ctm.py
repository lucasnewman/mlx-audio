"""Two-time DiT head for the published t=999 to s=0 CTM update."""

from mlx_audio.sts.models.dialogue_sidon.diffusion import (
    DiffusionHead,
    TimestepEmbedder,
)


class CTMHead(DiffusionHead):
    def __init__(self, config, latent_size, cond_size):
        super().__init__(config, latent_size, cond_size)
        self.s_embedder = TimestepEmbedder(
            config.hidden_size, config.frequency_embedding_size
        )

    def __call__(self, noise, t, s, conditioning):
        x = self.latent_proj(noise)
        time = self.t_embedder(t) + self.s_embedder(s)
        c = self.cond_proj(conditioning) + time[:, None, :]
        for block in self.blocks:
            x = block(x, c)
        return self.final_layer(x, c)
