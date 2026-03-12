import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from modules import (
    CNNBackbone,
    CLSCrossAttention,
    DecoderWithHints,
    FeatureLabelHead,
    FeatureTokenizer,
    MAEMasking,
    PatchCLSCrossAttention,
    PatchEmbedding,
    PatchFeatureExtractor,
    Patchify,
    PositionalEncoding,
    RawPatchify,
    SignalTokenizer,
    TransformerEncoder,
    VarPredictionHead,
    create_cls_isolation_mask,
)


class PRISMCTG(nn.Module):
    """
    PRISM-CTG: Pre-training with Reconstruction, Isolated Specialised tokens,
    and Multi-task learning for Cardiotocography.

    Masked Autoencoder with 3 specialised CLS tokens, label-hinted decoding,
    and cross-attention conditioning.

    CLS tokens:
        - cls_recon: drives patch reconstruction (cross-attention in decoder).
        - cls_var:   predicts clinical variables.
        - cls_feature: predicts per-patch feature labels (cross-attention to patches).

    During encoding, CLS tokens are isolated from each other via an attention
    mask but can attend to patches.  After encoding, CLS-to-CLS cross-attention
    allows information exchange between the three specialised tokens.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_cls = 3

        max_patches = config.max_seq_len // config.patch_size
        raw_patch_dim = config.input_channels * config.patch_size

        # --- Tokenizers ---
        self.raw_patchify = RawPatchify(config.patch_size, config.input_channels)
        self.signal_tokenizer = SignalTokenizer(
            raw_patch_dim=raw_patch_dim,
            proj_dim=config.signal_proj_dim,
            n_tokens=config.signal_n_tokens,
            seed=config.signal_tokenizer_seed,
        )
        self.label_embedding = nn.Embedding(config.signal_n_tokens, config.embed_dim)
        self.feature_extractor = PatchFeatureExtractor(
            config.patch_size, config.input_channels
        )
        self.feature_tokenizer = FeatureTokenizer(
            num_features=19,
            proj_dim=config.feature_proj_dim,
            n_tokens=config.feature_n_tokens,
            seed=config.feature_tokenizer_seed,
        )

        # --- CLS tokens ---
        self.cls_recon = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        self.cls_var = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        self.cls_feature = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        self.cls_pos_embed = nn.Parameter(torch.randn(1, 3, config.embed_dim) * 0.02)

        # --- Encoder ---
        self.cnn_backbone = CNNBackbone(config.input_channels, config.cnn_dim)
        self.patch_embed = PatchEmbedding(config.cnn_dim, config.patch_size, config.embed_dim)
        self.pos_enc = PositionalEncoding(max_patches, config.embed_dim)
        self.mae_masking = MAEMasking(config.mask_ratio)
        self.encoder = TransformerEncoder(
            config.embed_dim, config.num_heads, config.encoder_layers,
            config.mlp_ratio, config.dropout,
        )

        # --- CLS-to-CLS cross-attention ---
        self.cross_attn_cls_r = CLSCrossAttention(config.embed_dim, num_heads=4)
        self.cross_attn_cls_v = CLSCrossAttention(config.embed_dim, num_heads=4)
        self.cross_attn_cls_f = CLSCrossAttention(config.embed_dim, num_heads=4)

        # --- Patch-CLS cross-attention (feature prediction) ---
        self.patch_cls_cross_attn_f = PatchCLSCrossAttention(config.embed_dim, num_heads=4)

        # --- Decoder ---
        self.decoder = DecoderWithHints(
            config.embed_dim, config.decoder_embed_dim, config.patch_size,
            config.input_channels, config.decoder_num_heads, config.decoder_layers,
        )

        # --- Task heads ---
        self.var_head = VarPredictionHead(config.embed_dim, config.num_variables)
        self.feature_head = FeatureLabelHead(config.embed_dim, config.feature_n_tokens)
        self.patchify = Patchify(config.patch_size, config.input_channels)

        # --- Uncertainty weighting ---
        self.log_var_recon = nn.Parameter(torch.zeros(1))
        self.log_var_var = nn.Parameter(torch.zeros(1))
        self.log_var_feature = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Forward pass (training)
    # ------------------------------------------------------------------

    def forward(self, ctg_signal, clinical_vars):
        original_vars = clinical_vars.clone()
        B = ctg_signal.size(0)

        # Patchify & tokenize
        raw_patches = self.raw_patchify(ctg_signal)
        with torch.no_grad():
            signal_labels = self.signal_tokenizer(raw_patches)
        label_embeds = self.label_embedding(signal_labels)

        # CNN + patch embedding
        cnn_features = self.cnn_backbone(ctg_signal)
        x_full = self.patch_embed(cnn_features)
        x_full = self.pos_enc(x_full)
        num_patches = x_full.size(1)
        x_full = x_full + label_embeds

        # Mask 75%
        x_visible, mask, ids_restore, ids_keep = self.mae_masking(x_full)
        num_visible = x_visible.size(1)

        # Prepend CLS tokens
        cls_tokens = torch.cat(
            [
                self.cls_recon.expand(B, -1, -1),
                self.cls_var.expand(B, -1, -1),
                self.cls_feature.expand(B, -1, -1),
            ],
            dim=1,
        )
        cls_tokens = cls_tokens + self.cls_pos_embed
        x_with_cls = torch.cat([cls_tokens, x_visible], dim=1)

        # Encode with CLS isolation
        attn_mask = create_cls_isolation_mask(self.num_cls, num_visible, x_with_cls.device)
        x_encoded = self.encoder(x_with_cls, attn_mask=attn_mask)

        cls_r, cls_v, cls_f = x_encoded[:, 0:1], x_encoded[:, 1:2], x_encoded[:, 2:3]
        patches = x_encoded[:, 3:]

        # CLS-to-CLS cross-attention
        cls_r_enriched = self.cross_attn_cls_r(cls_r, torch.cat([cls_v, cls_f], dim=1))
        cls_v_enriched = self.cross_attn_cls_v(cls_v, torch.cat([cls_r, cls_f], dim=1))
        cls_f_enriched = self.cross_attn_cls_f(cls_f, torch.cat([cls_r, cls_v], dim=1))

        # --- Task outputs ---
        reconstructed = self.decoder(
            patches, label_embeds, ids_restore, num_patches, cls_recon=cls_r_enriched
        )
        predicted_vars = self.var_head(cls_v_enriched.squeeze(1))
        patches_for_features = self.patch_cls_cross_attn_f(patches, cls_f_enriched)
        feature_logits = self.feature_head(patches_for_features)

        # --- Targets ---
        target = self.patchify(ctg_signal)
        with torch.no_grad():
            visible_raw_patches = torch.gather(
                raw_patches, 1,
                ids_keep.unsqueeze(-1).expand(-1, -1, raw_patches.size(-1)),
            )
            visible_features = self.feature_extractor(visible_raw_patches)
            visible_feature_labels = self.feature_tokenizer(visible_features)

        # --- Losses ---
        recon_loss = self._mae_loss(reconstructed, target, mask)
        var_loss = self._var_loss(predicted_vars, original_vars)
        feature_loss = F.cross_entropy(
            feature_logits.view(-1, self.config.feature_n_tokens),
            visible_feature_labels.view(-1),
        )

        # Uncertainty-weighted total
        lv_r = torch.clamp(self.log_var_recon, -10, 10)
        lv_v = torch.clamp(self.log_var_var, -10, 10)
        lv_f = torch.clamp(self.log_var_feature, -10, 10)

        total_loss = (
            recon_loss * torch.exp(-lv_r) + lv_r / 2
            + var_loss * torch.exp(-lv_v) + lv_v / 2
            + feature_loss * torch.exp(-lv_f) + lv_f / 2
        )

        # --- Metrics ---
        with torch.no_grad():
            known_mask = (original_vars != -1).float()
            var_mae = (torch.abs(predicted_vars - original_vars) * known_mask).sum() / (
                known_mask.sum() + 1e-8
            )
            feature_acc = (feature_logits.argmax(dim=-1) == visible_feature_labels).float().mean()

        return total_loss, {
            "recon_loss": recon_loss.item(),
            "var_loss": var_loss.item(),
            "feature_loss": feature_loss.item(),
            "var_mae": var_mae.item(),
            "feature_acc": feature_acc.item(),
            "recon_weight": torch.exp(-lv_r).item(),
            "var_weight": torch.exp(-lv_v).item(),
            "feature_weight": torch.exp(-lv_f).item(),
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def encode(self, ctg_signal):
        """Encode without masking (for downstream tasks)."""
        B = ctg_signal.size(0)

        raw_patches = self.raw_patchify(ctg_signal)
        with torch.no_grad():
            signal_labels = self.signal_tokenizer(raw_patches)
        label_embeds = self.label_embedding(signal_labels)

        cnn_features = self.cnn_backbone(ctg_signal)
        x = self.patch_embed(cnn_features)
        x = self.pos_enc(x)
        num_patches = x.size(1)
        x = x + label_embeds

        cls_tokens = torch.cat(
            [
                self.cls_recon.expand(B, -1, -1),
                self.cls_var.expand(B, -1, -1),
                self.cls_feature.expand(B, -1, -1),
            ],
            dim=1,
        )
        cls_tokens = cls_tokens + self.cls_pos_embed
        x = torch.cat([cls_tokens, x], dim=1)

        attn_mask = create_cls_isolation_mask(self.num_cls, num_patches, x.device)
        x = self.encoder(x, attn_mask=attn_mask)

        cls_r, cls_v, cls_f = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        patches = x[:, 3:]

        cls_r_enriched = self.cross_attn_cls_r(cls_r, torch.cat([cls_v, cls_f], dim=1))
        cls_v_enriched = self.cross_attn_cls_v(cls_v, torch.cat([cls_r, cls_f], dim=1))
        cls_f_enriched = self.cross_attn_cls_f(cls_f, torch.cat([cls_r, cls_v], dim=1))

        return {
            "cls_recon": cls_r_enriched.squeeze(1),
            "cls_var": cls_v_enriched.squeeze(1),
            "cls_feature": cls_f_enriched.squeeze(1),
            "patches": patches,
        }

    def encode_pooled(self, ctg_signal):
        """Get a single pooled representation (mean of three CLS tokens)."""
        enc = self.encode(ctg_signal)
        return (enc["cls_recon"] + enc["cls_var"] + enc["cls_feature"]) / 3

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mae_loss(pred, target, mask):
        loss = ((pred - target) ** 2).mean(dim=-1)
        mask_sum = mask.sum()
        if mask_sum == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return (loss * mask).sum() / mask_sum

    @staticmethod
    def _var_loss(predicted, original):
        known_mask = (original != -1).float()
        if known_mask.sum() == 0:
            return torch.tensor(0.0, device=predicted.device, requires_grad=True)
        loss = (predicted - original) ** 2
        return (loss * known_mask).sum() / known_mask.sum()
