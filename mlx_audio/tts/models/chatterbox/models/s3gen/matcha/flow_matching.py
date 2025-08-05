from abc import ABC

import mlx.core as mx
import mlx.nn as nn

from .decoder import Decoder


class BASECFM(nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

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
        z = mx.random.normal(mu.shape) * temperature
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

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

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (mx.array): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (mx.array): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (mx.array): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (mx.array, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = mx.random.uniform(shape=[b, 1, 1], dtype=mu.dtype)
        # sample noise p(x_0)
        z = mx.random.normal(x1.shape)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # Compute MSE loss manually
        estimator_output = self.estimator(y, mask, mu, t.squeeze(), spks)
        squared_diff = (estimator_output - u) ** 2
        masked_squared_diff = squared_diff * mask
        loss = mx.sum(masked_squared_diff) / (mx.sum(mask) * u.shape[1])

        return loss, y


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the architecture of the estimator here
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
