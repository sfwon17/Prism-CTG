import math

import torch
import torch.nn as nn

from attention import PatchCLSCrossAttention
from transformer import TransformerBlock


class DecoderWithHints(nn.Module):
    """
    MAE decoder that fuses two kinds of auxiliary information:

    - **Label hints**: injected via gated additive fusion.
    - **CLS reconstruction token**: injected via cross-attention
      (patches query the CLS token).
    """

    def __init__(
        self, embed_dim, decoder_embed_dim, patch_size, input_channels,
        num_heads=4, num_layers=2,
    ):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.hint_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.cls_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.cls_cross_attn = PatchCLSCrossAttention(decoder_embed_dim, num_heads=4)

        self.fusion_gate = nn.Sequential(
            nn.Linear(decoder_embed_dim * 2, decoder_embed_dim),
            nn.Sigmoid(),
        )

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(decoder_embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1)
                for _ in range(num_layers)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * input_channels)
        self.decoder_pos_embed = None
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x_encoded, label_embeds, ids_restore, num_patches, cls_recon=None):
        batch_size = x_encoded.size(0)
        x = self.decoder_embed(x_encoded)

        num_mask = num_patches - x.size(1)
        mask_tokens = self.mask_token.expand(batch_size, num_mask, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )

        # Label hints: gated additive fusion
        hints = self.hint_proj(label_embeds)
        gate = self.fusion_gate(torch.cat([x, hints], dim=-1))
        x = x + gate * hints

        # CLS info: cross-attention
        if cls_recon is not None:
            cls_info = self.cls_proj(cls_recon)
            x = self.cls_cross_attn(x, cls_info)

        if self.decoder_pos_embed is None or self.decoder_pos_embed.size(1) != num_patches:
            self.decoder_pos_embed = self._create_pos_embed(
                num_patches, x.size(-1), x.device
            )
        x = x + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def _create_pos_embed(self, num_patches, embed_dim, device):
        pe = torch.zeros(num_patches, embed_dim, device=device)
        position = torch.arange(0, num_patches, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, device=device).float()
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
