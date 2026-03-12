import torch
import torch.nn as nn


class CLSCrossAttention(nn.Module):
    """
    Cross-attention between CLS tokens.
    Each CLS token queries info from other CLS tokens.
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query_cls, other_cls):
        kv = torch.cat([query_cls, other_cls], dim=1)
        attn_out, _ = self.attn(
            query=self.norm1(query_cls), key=self.norm1(kv), value=kv
        )
        query_cls = query_cls + attn_out
        query_cls = query_cls + self.mlp(self.norm2(query_cls))
        return query_cls


class PatchCLSCrossAttention(nn.Module):
    """
    Patches query the CLS token to get task-specific information.

    Each patch can learn to attend differently to the CLS token,
    allowing for more expressive patch-level conditioning.
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, patches, cls_token):
        """
        Args:
            patches: [B, num_patches, embed_dim]
            cls_token: [B, 1, embed_dim]
        Returns:
            Patches enriched with CLS info: [B, num_patches, embed_dim]
        """
        attn_out, _ = self.cross_attn(
            query=self.norm1(patches),
            key=self.norm1(cls_token),
            value=cls_token,
        )
        patches = patches + attn_out
        patches = patches + self.mlp(self.norm2(patches))
        return patches
