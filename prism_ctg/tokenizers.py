import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalTokenizer(nn.Module):
    def __init__(self, raw_patch_dim, proj_dim, n_tokens, seed=42):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        random_proj = torch.randn(raw_patch_dim, proj_dim, generator=generator)
        random_proj = F.normalize(random_proj, dim=0)
        self.register_buffer("random_projection", random_proj)
        codebook = torch.randn(n_tokens, proj_dim, generator=generator)
        codebook = F.normalize(codebook, dim=1)
        self.register_buffer("codebook", codebook)

    def forward(self, raw_patches):
        x_proj = torch.matmul(raw_patches, self.random_projection)
        x_proj = F.normalize(x_proj, dim=-1)
        similarities = torch.matmul(x_proj, self.codebook.t())
        return torch.argmax(similarities, dim=-1)


class FeatureTokenizer(nn.Module):
    def __init__(self, num_features=19, proj_dim=64, n_tokens=256, seed=123):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        random_proj = torch.randn(num_features, proj_dim, generator=generator)
        random_proj = F.normalize(random_proj, dim=0)
        self.register_buffer("random_projection", random_proj)
        codebook = torch.randn(n_tokens, proj_dim, generator=generator)
        codebook = F.normalize(codebook, dim=1)
        self.register_buffer("codebook", codebook)
        self.n_tokens = n_tokens

    def forward(self, features):
        x_proj = torch.matmul(features, self.random_projection)
        x_proj = F.normalize(x_proj, dim=-1)
        similarities = torch.matmul(x_proj, self.codebook.t())
        return torch.argmax(similarities, dim=-1)


class PatchFeatureExtractor(nn.Module):
    """Extracts hand-crafted FHR and TOCO features from raw signal patches."""

    def __init__(self, patch_size, input_channels, default_fhr=140.0, sampling_rate=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.default_fhr = default_fhr
        self.sampling_rate = sampling_rate
        self.num_features = 19

    def forward(self, raw_patches):
        batch_size, num_patches, patch_dim = raw_patches.shape
        device = raw_patches.device

        raw_patches = raw_patches.reshape(
            batch_size, num_patches, self.input_channels, self.patch_size
        )
        fhr_patches = raw_patches[:, :, 0, :]
        toco_patches = raw_patches[:, :, 1, :]

        fhr_features = self._extract_fhr_features(fhr_patches)
        toco_features = self._extract_toco_features(toco_patches)

        return torch.cat([fhr_features, toco_features], dim=-1)

    def _extract_fhr_features(self, fhr_patches):
        fhr_valid = (fhr_patches != 0) & (fhr_patches != -1)
        fhr_valid_count = fhr_valid.sum(dim=-1).clamp(min=1).float()
        no_valid_fhr = fhr_valid.sum(dim=-1) == 0

        # Mean
        fhr_sum = (fhr_patches * fhr_valid).sum(dim=-1)
        patch_mean = fhr_sum / fhr_valid_count
        patch_mean = torch.where(
            no_valid_fhr, torch.full_like(patch_mean, self.default_fhr), patch_mean
        )

        # Std
        fhr_sq_sum = (fhr_patches**2 * fhr_valid).sum(dim=-1)
        fhr_mean_sq = fhr_sq_sum / fhr_valid_count
        fhr_var = (fhr_mean_sq - patch_mean**2).clamp(min=0)
        patch_std = torch.sqrt(fhr_var)
        patch_std = torch.where(
            fhr_valid_count <= 1, torch.zeros_like(patch_std), patch_std
        )

        # Min
        fhr_for_min = fhr_patches.clone()
        fhr_for_min[~fhr_valid] = float("inf")
        patch_min = fhr_for_min.min(dim=-1).values
        patch_min = torch.where(
            no_valid_fhr, torch.full_like(patch_min, self.default_fhr), patch_min
        )

        # Max
        fhr_for_max = fhr_patches.clone()
        fhr_for_max[~fhr_valid] = float("-inf")
        patch_max = fhr_for_max.max(dim=-1).values
        patch_max = torch.where(
            no_valid_fhr, torch.full_like(patch_max, self.default_fhr), patch_max
        )

        patch_range = patch_max - patch_min
        valid_ratio = fhr_valid.sum(dim=-1).float() / self.patch_size

        # Clinical events
        above_mean = (
            (fhr_patches >= patch_mean.unsqueeze(-1) + 15)
            & (fhr_patches < 180)
            & fhr_valid
        )
        has_accel = above_mean.any(dim=-1).float()

        below_mean = (
            (fhr_patches <= patch_mean.unsqueeze(-1) - 15)
            & (fhr_patches > 110)
            & fhr_valid
        )
        has_decel = below_mean.any(dim=-1).float()

        has_tachy = ((fhr_patches > 180) & fhr_valid).any(dim=-1).float()
        has_brady = ((fhr_patches < 110) & fhr_valid).any(dim=-1).float()

        return torch.stack(
            [
                patch_mean, patch_std, patch_min, patch_max, patch_range,
                valid_ratio, has_accel, has_decel, has_tachy, has_brady,
            ],
            dim=-1,
        )

    def _extract_toco_features(self, toco_patches):
        toco_valid = (toco_patches != 0) & (toco_patches != -1)
        toco_valid_count = toco_valid.sum(dim=-1).clamp(min=1).float()
        no_valid_toco = toco_valid.sum(dim=-1) == 0

        # Mean
        toco_sum = (toco_patches * toco_valid).sum(dim=-1)
        toco_mean = toco_sum / toco_valid_count
        toco_mean = torch.where(no_valid_toco, torch.zeros_like(toco_mean), toco_mean)

        # Std
        toco_sq_sum = (toco_patches**2 * toco_valid).sum(dim=-1)
        toco_mean_sq = toco_sq_sum / toco_valid_count
        toco_var = (toco_mean_sq - toco_mean**2).clamp(min=0)
        toco_std = torch.sqrt(toco_var)
        toco_std = torch.where(
            toco_valid_count <= 1, torch.zeros_like(toco_std), toco_std
        )

        # Min
        toco_for_min = toco_patches.clone()
        toco_for_min[~toco_valid] = float("inf")
        toco_min = toco_for_min.min(dim=-1).values
        toco_min = torch.where(no_valid_toco, torch.zeros_like(toco_min), toco_min)

        # Max
        toco_for_max = toco_patches.clone()
        toco_for_max[~toco_valid] = float("-inf")
        toco_max = toco_for_max.max(dim=-1).values
        toco_max = torch.where(no_valid_toco, torch.zeros_like(toco_max), toco_max)

        toco_range = toco_max - toco_min
        toco_p95 = toco_mean + 1.645 * toco_std
        toco_p99 = toco_mean + 2.326 * toco_std

        # Spectral features
        toco_dominant_freq, toco_dominant_power, toco_spectral_energy, toco_periodicity = (
            self._compute_toco_fft(toco_patches, toco_valid, toco_mean)
        )

        return torch.stack(
            [
                toco_mean, toco_std, toco_range, toco_p95, toco_p99,
                toco_dominant_freq, toco_dominant_power, toco_spectral_energy, toco_periodicity,
            ],
            dim=-1,
        )

    def _compute_toco_fft(self, toco_patches, toco_valid, toco_mean):
        batch_size, num_patches, patch_size = toco_patches.shape
        device = toco_patches.device

        toco_filled = toco_patches.clone()
        toco_filled = torch.where(
            toco_valid, toco_filled, toco_mean.unsqueeze(-1).expand_as(toco_filled)
        )
        toco_centered = toco_filled - toco_filled.mean(dim=-1, keepdim=True)

        fft_result = torch.fft.rfft(toco_centered, dim=-1)
        power_spectrum = torch.abs(fft_result) ** 2
        power_spectrum_no_dc = power_spectrum[:, :, 1:]

        max_idx = torch.argmax(power_spectrum_no_dc, dim=-1)
        freqs = torch.fft.rfftfreq(patch_size, d=1.0 / self.sampling_rate).to(device)
        freqs_no_dc = freqs[1:]

        toco_dominant_freq = freqs_no_dc[max_idx]
        toco_dominant_power = torch.gather(
            power_spectrum_no_dc, -1, max_idx.unsqueeze(-1)
        ).squeeze(-1)
        toco_spectral_energy = power_spectrum_no_dc.sum(dim=-1)
        toco_periodicity = toco_dominant_power / (toco_spectral_energy + 1e-8)

        return toco_dominant_freq, toco_dominant_power, toco_spectral_energy, toco_periodicity
