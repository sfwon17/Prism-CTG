from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_channels: int = 2
    num_variables: int = 3
    patch_size: int = 32
    max_seq_len: int = 1200
    cnn_dim: int = 64
    embed_dim: int = 256
    num_heads: int = 8
    encoder_layers: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    decoder_embed_dim: int = 128
    decoder_num_heads: int = 4
    decoder_layers: int = 2
    signal_proj_dim: int = 256
    signal_n_tokens: int = 512
    signal_tokenizer_seed: int = 42
    feature_proj_dim: int = 64
    feature_n_tokens: int = 256
    feature_tokenizer_seed: int = 123
    mask_ratio: float = 0.75
