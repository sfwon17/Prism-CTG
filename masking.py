import torch
import torch.nn as nn


class MAEMasking(nn.Module):
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        num_mask = int(num_patches * self.mask_ratio)
        num_visible = num_patches - num_mask

        noise = torch.rand(batch_size, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_visible]
        x_visible = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, embed_dim)
        )

        mask = torch.ones(batch_size, num_patches, device=x.device)
        mask[:, :num_visible] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_restore, ids_keep


class RawPatchify(nn.Module):
    def __init__(self, patch_size, input_channels):
        super().__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        num_patches = seq_len // self.patch_size
        x = x[:, :, : num_patches * self.patch_size]
        x = x.reshape(batch_size, channels, num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, num_patches, channels * self.patch_size)
        return x


class Patchify(nn.Module):
    def __init__(self, patch_size, input_channels):
        super().__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        num_patches = seq_len // self.patch_size
        x = x[:, :, : num_patches * self.patch_size]
        x = x.reshape(batch_size, channels, num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, num_patches, channels * self.patch_size)
