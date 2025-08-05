import mlx.core as mx
import mlx.nn as nn

from .matcha.flow_matching import BASECFM
from .configs import CFM_PARAMS


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        self.estimator = estimator

    def __call__(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, flow_cache=None):
        """Forward diffusion

        Args:
            mu (mx.array): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (mx.array): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (mx.array, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
            prompt_len: length of prompt
            flow_cache: cache for flow matching

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        if flow_cache is None:
            flow_cache = mx.zeros((1, 80, 0, 2))

        z = mx.random.normal(mu.shape, dtype=mu.dtype) * temperature
        cache_size = flow_cache.shape[2]

        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z = mx.where(mx.arange(z.shape[2]) < cache_size, flow_cache[:, :, :cache_size, 0], z)
            mu = mx.where(mx.arange(mu.shape[2]) < cache_size, flow_cache[:, :, :cache_size, 1], mu)

        z_cache = mx.concatenate([z[:, :, :prompt_len], z[:, :, -34:]], axis=2)
        mu_cache = mx.concatenate([mu[:, :, :prompt_len], mu[:, :, -34:]], axis=2)
        flow_cache = mx.stack([z_cache, mu_cache], axis=-1)

        t_span = mx.linspace(0, 1, n_timesteps + 1, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - mx.cos(t_span * 0.5 * mx.pi)

        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), flow_cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (mx.array): random noise
            t_span (mx.array): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (mx.array): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (mx.array): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (mx.array, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = mx.expand_dims(t, axis=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        # Prepare input tensors for classifier-free guidance
        x_in = mx.zeros((2, 80, x.shape[2]), dtype=x.dtype)
        mask_in = mx.zeros((2, 1, x.shape[2]), dtype=x.dtype)
        mu_in = mx.zeros((2, 80, x.shape[2]), dtype=x.dtype)
        t_in = mx.zeros((2,), dtype=x.dtype)
        spks_in = mx.zeros((2, 80), dtype=x.dtype)
        cond_in = mx.zeros((2, 80, x.shape[2]), dtype=x.dtype)

        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in = mx.broadcast_to(x, (2,) + x.shape[1:])
            mask_in = mx.broadcast_to(mask, (2,) + mask.shape[1:])
            mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
            t_in = mx.broadcast_to(mx.expand_dims(t, axis=0), (2,))
            spks_in = mx.concatenate([spks, mx.zeros_like(spks)], axis=0)
            cond_in = mx.concatenate([cond, mx.zeros_like(cond)], axis=0)

            dphi_dt = self.forward_estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

            dphi_dt_cond = dphi_dt[: x.shape[0]]
            dphi_dt_uncond = dphi_dt[x.shape[0] :]
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt_cond - self.inference_cfg_rate * dphi_dt_uncond

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].astype(mx.float32)

    def forward_estimator(self, x, mask, mu, t, spks, cond):
        return self.estimator(x, mask, mu, t, spks, cond)


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels=240, cfm_params=CFM_PARAMS, n_spks=1, spk_emb_dim=80, estimator=None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        self._rand_noise = mx.random.normal((1, 80, 50 * 300))

    def __call__(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (mx.array): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (mx.array): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (mx.array, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = self._rand_noise[:, :, : mu.shape[2]].astype(mu.dtype) * temperature

        t_span = mx.linspace(0, 1, n_timesteps + 1, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - mx.cos(t_span * 0.5 * mx.pi)

        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None
